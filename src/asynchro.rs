use crate::error::{ResampleError, ResampleResult};
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
use crate::interpolator_avx::AvxInterpolator;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
use crate::interpolator_neon::NeonInterpolator;
#[cfg(target_arch = "x86_64")]
use crate::interpolator_sse::SseInterpolator;
use crate::sinc::make_sincs;
use crate::windows::WindowFunction;
use crate::{InterpolationParameters, InterpolationType};
use crate::{Resampler, Sample};
use audio_core::Channel;

/// Functions for making the scalar product with a sinc
pub trait SincInterpolator<T> {
    /// Make the scalar product between the waveform starting at `index` and the sinc of `subindex`.
    fn get_sinc_interpolated(&self, wave: &[T], index: usize, subindex: usize) -> T;

    /// Get sinc length
    fn len(&self) -> usize;

    /// Check if sincs are empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get number of sincs used for oversampling
    fn nbr_sincs(&self) -> usize;
}

/// A plain scalar interpolator
pub struct ScalarInterpolator<T> {
    sincs: Vec<Vec<T>>,
    length: usize,
    nbr_sincs: usize,
}

impl<T> SincInterpolator<T> for ScalarInterpolator<T>
where
    T: Sample,
{
    /// Calculate the scalar produt of an input wave and the selected sinc filter
    fn get_sinc_interpolated(&self, wave: &[T], index: usize, subindex: usize) -> T {
        assert!(
            (index + self.length) < wave.len(),
            "Tried to interpolate for index {}, max for the given input is {}",
            index,
            wave.len() - self.length - 1
        );
        assert!(
            subindex < self.nbr_sincs,
            "Tried to use sinc subindex {}, max is {}",
            subindex,
            self.nbr_sincs - 1
        );
        let wave_cut = &wave[index..(index + self.sincs[subindex].len())];
        let sinc = &self.sincs[subindex];
        unsafe {
            let mut acc0 = T::zero();
            let mut acc1 = T::zero();
            let mut acc2 = T::zero();
            let mut acc3 = T::zero();
            let mut acc4 = T::zero();
            let mut acc5 = T::zero();
            let mut acc6 = T::zero();
            let mut acc7 = T::zero();
            let mut idx = 0;
            for _ in 0..wave_cut.len() / 8 {
                acc0 += *wave_cut.get_unchecked(idx) * *sinc.get_unchecked(idx);
                acc1 += *wave_cut.get_unchecked(idx + 1) * *sinc.get_unchecked(idx + 1);
                acc2 += *wave_cut.get_unchecked(idx + 2) * *sinc.get_unchecked(idx + 2);
                acc3 += *wave_cut.get_unchecked(idx + 3) * *sinc.get_unchecked(idx + 3);
                acc4 += *wave_cut.get_unchecked(idx + 4) * *sinc.get_unchecked(idx + 4);
                acc5 += *wave_cut.get_unchecked(idx + 5) * *sinc.get_unchecked(idx + 5);
                acc6 += *wave_cut.get_unchecked(idx + 6) * *sinc.get_unchecked(idx + 6);
                acc7 += *wave_cut.get_unchecked(idx + 7) * *sinc.get_unchecked(idx + 7);
                idx += 8;
            }
            acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7
        }
    }

    fn len(&self) -> usize {
        self.length
    }

    fn nbr_sincs(&self) -> usize {
        self.nbr_sincs
    }
}

impl<T> ScalarInterpolator<T>
where
    T: Sample,
{
    /// Create a new ScalarInterpolator
    ///
    /// Parameters are:
    /// - `sinc_len`: Length of sinc functions.
    /// - `oversampling_factor`: Number of intermediate sincs (oversampling factor).
    /// - `f_cutoff`: Relative cutoff frequency.
    /// - `window`: Window function to use.
    pub fn new(
        sinc_len: usize,
        oversampling_factor: usize,
        f_cutoff: f32,
        window: WindowFunction,
    ) -> Self {
        assert!(sinc_len % 8 == 0, "Sinc length must be a multiple of 8");
        let sincs = make_sincs(sinc_len, oversampling_factor, f_cutoff, window);
        Self {
            sincs,
            length: sinc_len,
            nbr_sincs: oversampling_factor,
        }
    }
}

/// A current set of sinc parameters. These might or might not be dynamic.
#[derive(Debug, Clone, Copy)]
pub struct SincParams {
    needed_input: usize,
    current_fill: usize,
    output_len: usize,
}

/// A currently internal only trait which governs how a sinc implementation
/// works.
pub trait SincKind {
    type Indexer: Iterator<Item = (usize, f64)>;

    /// Construct a new sinc kind with the specified parameters.
    fn new(chunk_size: usize, resample_ratio: f64, len: usize) -> Self;

    /// Get the current chunk size.
    fn chunk_size(&self) -> usize;

    /// Get the current sampling parameters.
    fn params(&self) -> SincParams;

    /// Construct a new indexer used to calculate frame indexes being resampled.
    fn indexer(&mut self, sinc_len: usize) -> Self::Indexer;

    /// Get the current number of frames needed.
    fn frames_needed(&self) -> usize;

    /// Perform an internal post-processing stage.
    fn post_process<O>(&mut self, indexer: Self::Indexer, sinc_len: usize, wave_out: O)
    where
        O: audio_core::ResizableBuf;

    /// Update the internal resample ratio.
    fn update_resample_ratio(&mut self, new_ratio: f64, sinc_len: usize);
}

#[derive(Default)]
pub struct FixedOut {
    resample_ratio: f64,
    last_index: f64,
    chunk_size: usize,
    needed_input_size: usize,
    current_buffer_fill: usize,
}

impl SincKind for FixedOut {
    type Indexer = crate::interpolation_type::RatioIndexer;

    fn new(chunk_size: usize, resample_ratio: f64, sinc_len: usize) -> Self {
        let needed_input_size =
            (chunk_size as f64 / resample_ratio).ceil() as usize + 2 + sinc_len / 2;

        Self {
            resample_ratio,
            last_index: -((sinc_len / 2) as f64),
            chunk_size,
            needed_input_size,
            current_buffer_fill: needed_input_size,
        }
    }

    fn chunk_size(&self) -> usize {
        3 * self.needed_input_size / 2
    }

    fn params(&self) -> SincParams {
        SincParams {
            needed_input: self.needed_input_size,
            current_fill: self.current_buffer_fill,
            output_len: self.chunk_size,
        }
    }

    fn frames_needed(&self) -> usize {
        self.needed_input_size
    }

    fn indexer(&mut self, _: usize) -> Self::Indexer {
        let t_ratio = 1.0 / self.resample_ratio as f64;

        crate::interpolation_type::RatioIndexer::new(self.last_index, t_ratio, self.chunk_size)
    }

    fn post_process<O>(&mut self, indexer: Self::Indexer, sinc_len: usize, _: O)
    where
        O: audio_core::ResizableBuf,
    {
        let idx = indexer.current;

        self.current_buffer_fill = self.needed_input_size;
        // store last index for next iteration
        self.last_index = idx - self.current_buffer_fill as f64;

        self.needed_input_size = (self.last_index as f32
            + self.chunk_size as f32 / self.resample_ratio as f32
            + sinc_len as f32)
            .ceil() as usize
            + 2;
    }

    fn update_resample_ratio(&mut self, new_ratio: f64, sinc_len: usize) {
        self.resample_ratio = new_ratio;

        self.needed_input_size = (self.last_index as f32
            + self.chunk_size as f32 / self.resample_ratio as f32
            + sinc_len as f32)
            .ceil() as usize
            + 2;
    }
}

#[derive(Default)]
pub struct FixedIn {
    resample_ratio: f64,
    last_index: f64,
    chunk_size: usize,
}

impl SincKind for FixedIn {
    type Indexer = crate::interpolation_type::SpanIndexer;

    fn new(chunk_size: usize, resample_ratio: f64, sinc_len: usize) -> Self {
        Self {
            resample_ratio,
            last_index: -((sinc_len / 2) as f64),
            chunk_size,
        }
    }

    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    fn params(&self) -> SincParams {
        let output_len = (self.chunk_size as f64 * self.resample_ratio + 10.0) as usize;

        SincParams {
            needed_input: self.chunk_size,
            current_fill: self.chunk_size,
            output_len,
        }
    }

    fn frames_needed(&self) -> usize {
        self.chunk_size
    }

    fn indexer(&mut self, sinc_len: usize) -> Self::Indexer {
        let t_ratio = 1.0 / self.resample_ratio as f64;
        let end_idx = self.chunk_size as isize - (sinc_len as isize + 1);

        crate::interpolation_type::SpanIndexer::new(self.last_index, t_ratio, end_idx as f64)
    }

    fn post_process<O>(&mut self, indexer: Self::Indexer, _: usize, mut wave_out: O)
    where
        O: audio_core::ResizableBuf,
    {
        let idx = indexer.current;
        let n = indexer.index;

        // store last index for next iteration
        self.last_index = idx - self.chunk_size as f64;
        wave_out.resize(n);
    }

    #[inline]
    fn update_resample_ratio(&mut self, new_ratio: f64, _: usize) {
        self.resample_ratio = new_ratio;
    }
}

/// An asynchronous resampler that accepts a fixed number of audio frames for input
/// and returns a variable number of frames.
///
/// The resampling is done by creating a number of intermediate points (defined by oversampling_factor)
/// by sinc interpolation. The new samples are then calculated by interpolating between these points.
pub type SincFixedIn<T> = Sinc<T, FixedIn>;

/// An asynchronous resampler that return a fixed number of audio frames.
/// The number of input frames required is given by the frames_needed function.
///
/// The resampling is done by creating a number of intermediate points (defined by oversampling_factor)
/// by sinc interpolation. The new samples are then calculated by interpolating between these points.
pub type SincFixedOut<T> = Sinc<T, FixedOut>;

pub struct Sinc<T, K> {
    nbr_channels: usize,
    resample_ratio_original: f64,
    interpolator: Box<dyn SincInterpolator<T> + Send + Sync>,
    buffer: audio::buf::Sequential<T>,
    interpolation: InterpolationType,
    kind: K,
}

pub fn make_interpolator<T>(
    sinc_len: usize,
    resample_ratio: f64,
    f_cutoff: f32,
    oversampling_factor: usize,
    window: WindowFunction,
) -> Box<dyn SincInterpolator<T> + Send + Sync>
where
    T: Sample,
{
    let sinc_len = 8 * (((sinc_len as f32) / 8.0).ceil() as usize);
    let f_cutoff = if resample_ratio >= 1.0 {
        f_cutoff
    } else {
        f_cutoff * resample_ratio as f32
    };

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if let Ok(interpolator) =
        AvxInterpolator::<T>::new(sinc_len, oversampling_factor, f_cutoff, window)
    {
        return Box::new(interpolator);
    }

    #[cfg(target_arch = "x86_64")]
    if let Ok(interpolator) =
        SseInterpolator::<T>::new(sinc_len, oversampling_factor, f_cutoff, window)
    {
        return Box::new(interpolator);
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    if let Ok(interpolator) =
        NeonInterpolator::<T>::new(sinc_len, oversampling_factor, f_cutoff, window)
    {
        return Box::new(interpolator);
    }

    Box::new(ScalarInterpolator::<T>::new(
        sinc_len,
        oversampling_factor,
        f_cutoff,
        window,
    ))
}

impl<T, K> Resampler<T> for Sinc<T, K>
where
    T: Sample,
    K: SincKind,
{
    /// Resample a chunk of audio. The input length is fixed, and the output varies in length.
    /// If the waveform for a channel is empty, this channel will be ignored and produce a
    /// corresponding empty output waveform.
    /// # Errors
    ///
    /// The function returns an error if the length of the input data is not equal
    /// to the number of channels and chunk size defined when creating the instance.
    fn process_with_buffer<B, O, M: ?Sized>(
        &mut self,
        wave_in: B,
        mut wave_out: O,
        mask: &M,
    ) -> ResampleResult<()>
    where
        B: audio_core::Buf<Sample = T>,
        O: audio_core::BufMut<Sample = T> + audio_core::ResizableBuf,
        M: bittle::Mask,
    {
        if wave_in.channels() != self.nbr_channels {
            return Err(ResampleError::WrongNumberOfChannels {
                expected: self.nbr_channels,
                actual: wave_in.channels(),
            });
        }

        let p = self.kind.params();

        for (chan, wave) in mask.join(wave_in.iter_channels().enumerate()) {
            if wave.len() != p.needed_input {
                return Err(ResampleError::WrongNumberOfFrames {
                    channel: chan,
                    expected: p.needed_input,
                    actual: wave.len(),
                });
            }
        }

        let sinc_len = self.interpolator.len();
        let oversampling_factor = self.interpolator.nbr_sincs();

        // Update buffer with new data.
        for mut wav in self.buffer.iter_channels_mut() {
            for idx in 0..(2 * sinc_len) {
                wav[idx] = wav[idx + p.current_fill];
            }
        }

        wave_out.resize_topology(self.nbr_channels, p.output_len);

        for (out, wave_in) in
            mask.join(self.buffer.iter_channels_mut().zip(wave_in.iter_channels()))
        {
            audio::channel::copy(wave_in, out.skip(2 * sinc_len));
        }

        let mut indexer = self.kind.indexer(sinc_len);

        self.interpolation.apply_to(
            &self.buffer,
            &mut indexer,
            oversampling_factor,
            sinc_len,
            self.interpolator.as_ref(),
            &mut wave_out,
            mask,
        );

        self.kind.post_process(indexer, sinc_len, wave_out);
        Ok(())
    }

    /// Query for the number of frames needed for the next call to "process".
    /// Will always return the chunk_size defined when creating the instance.
    fn nbr_frames_needed(&self) -> usize {
        self.kind.frames_needed()
    }

    /// Update the resample ratio. New value must be within +-10% of the original one
    fn set_resample_ratio(&mut self, new_ratio: f64) -> ResampleResult<()> {
        trace!("Change resample ratio to {}", new_ratio);
        if (new_ratio / self.resample_ratio_original > 0.9)
            && (new_ratio / self.resample_ratio_original < 1.1)
        {
            self.kind
                .update_resample_ratio(new_ratio, self.interpolator.len());
            Ok(())
        } else {
            Err(ResampleError::BadRatioUpdate)
        }
    }

    /// Update the resample ratio relative to the original one
    fn set_resample_ratio_relative(&mut self, rel_ratio: f64) -> ResampleResult<()> {
        let new_ratio = self.resample_ratio_original * rel_ratio;
        self.set_resample_ratio(new_ratio)
    }
}

impl<T, K> Sinc<T, K>
where
    T: Sample,
    K: SincKind,
{
    /// Create a new SincFixedOut
    ///
    /// Parameters are:
    /// - `resample_ratio`: Ratio between output and input sample rates.
    /// - `parameters`: Parameters for interpolation, see `InterpolationParameters`
    /// - `chunk_size`: size of output data in frames
    /// - `nbr_channels`: number of channels in input/output
    pub fn new(
        resample_ratio: f64,
        parameters: InterpolationParameters,
        chunk_size: usize,
        nbr_channels: usize,
    ) -> Self {
        debug!(
            "Create new Sinc, ratio: {}, chunk_size: {}, channels: {}, parameters: {:?}",
            resample_ratio, chunk_size, nbr_channels, parameters
        );

        let interpolator = make_interpolator(
            parameters.sinc_len,
            resample_ratio,
            parameters.f_cutoff,
            parameters.oversampling_factor,
            parameters.window,
        );

        Self::new_with_interpolator(
            resample_ratio,
            parameters.interpolation,
            interpolator,
            chunk_size,
            nbr_channels,
        )
    }

    /// Create a new SincFixedOut using an existing Interpolator
    ///
    /// Parameters are:
    /// - `resample_ratio`: Ratio between output and input sample rates.
    /// - `interpolation_type`: Parameters for interpolation, see `InterpolationParameters`
    /// - `interpolator`:  The interpolator to use
    /// - `chunk_size`: size of output data in frames
    /// - `nbr_channels`: number of channels in input/output
    pub fn new_with_interpolator(
        resample_ratio: f64,
        interpolation: InterpolationType,
        interpolator: Box<dyn SincInterpolator<T> + Send + Sync>,
        chunk_size: usize,
        nbr_channels: usize,
    ) -> Self {
        let kind = K::new(chunk_size, resample_ratio, interpolator.len());

        let buffer = audio::sequential![[T::zero(); kind.chunk_size() + 2 * interpolator.len()]; nbr_channels];

        Self {
            nbr_channels,
            resample_ratio_original: resample_ratio,
            interpolator,
            buffer,
            interpolation,
            kind,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::asynchro::ScalarInterpolator;
    use crate::asynchro::SincInterpolator;
    use crate::interpolation_type::{interp_cubic, interp_lin};
    use crate::InterpolationParameters;
    use crate::InterpolationType;
    use crate::Resampler;
    use crate::WindowFunction;
    use crate::{SincFixedIn, SincFixedOut};
    use num_traits::Float;
    use rand::Rng;

    fn get_sinc_interpolated<T: Float>(wave: &[T], index: usize, sinc: &[T]) -> T {
        let wave_cut = &wave[index..(index + sinc.len())];
        wave_cut
            .iter()
            .zip(sinc.iter())
            .fold(T::zero(), |acc, (x, y)| acc + *x * *y)
    }

    #[test]
    fn test_scalar_interpolator_64() {
        let mut rng = rand::thread_rng();
        let mut wave = Vec::new();
        for _ in 0..2048 {
            wave.push(rng.gen::<f64>());
        }
        let sinc_len = 256;
        let f_cutoff = 0.9473371669037001;
        let oversampling_factor = 256;
        let window = WindowFunction::BlackmanHarris2;

        let interpolator =
            ScalarInterpolator::<f64>::new(sinc_len, oversampling_factor, f_cutoff, window);
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &interpolator.sincs[123]);
        assert!((value - check).abs() < 1.0e-9);
    }

    #[test]
    fn test_scalar_interpolator_32() {
        let mut rng = rand::thread_rng();
        let mut wave = Vec::new();
        for _ in 0..2048 {
            wave.push(rng.gen::<f32>());
        }
        let sinc_len = 256;
        let f_cutoff = 0.9473371669037001;
        let oversampling_factor = 256;
        let window = WindowFunction::BlackmanHarris2;

        let interpolator =
            ScalarInterpolator::<f32>::new(sinc_len, oversampling_factor, f_cutoff, window);
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &interpolator.sincs[123]);
        assert!((value - check).abs() < 1.0e-6);
    }

    #[test]
    fn int_cubic() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let _resampler = SincFixedIn::<f64>::new(1.2, params, 1024, 2);
        let yvals = [0.0f64, 2.0f64, 4.0f64, 6.0f64];
        let interp = interp_cubic(0.5f64, &yvals);
        assert_eq!(interp, 3.0f64);
    }

    #[test]
    fn int_lin_32() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let _resampler = SincFixedIn::<f32>::new(1.2, params, 1024, 2);
        let yvals = [1.0f32, 5.0f32];
        let interp = interp_lin(0.25f32, &yvals);
        assert_eq!(interp, 2.0f32);
    }

    #[test]
    fn int_cubic_32() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let _resampler = SincFixedIn::<f32>::new(1.2, params, 1024, 2);
        let yvals = [0.0f32, 2.0f32, 4.0f32, 6.0f32];
        let interp = interp_cubic(0.5f32, &yvals);
        assert_eq!(interp, 3.0f32);
    }

    #[test]
    fn int_lin() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let _resampler = SincFixedIn::<f64>::new(1.2, params, 1024, 2);
        let yvals = [1.0f64, 5.0f64];
        let interp = interp_lin(0.25f64, &yvals);
        assert_eq!(interp, 2.0f64);
    }

    #[test]
    fn make_resampler_fi() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedIn::<f64>::new(1.2, params, 1024, 2);
        let waves = audio::dynamic![[0.0f64; 1024]; 2];
        let out = resampler.process(&waves, &bittle::all()).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert!(
            out[0].len() > 1150 && out[0].len() < 1229,
            "expected {} - {} samples, got {}",
            1150,
            1229,
            out[0].len()
        );
        let out2 = resampler.process(&waves, &bittle::all()).unwrap();
        assert_eq!(out2.len(), 2, "Expected {} channels, got {}", 2, out2.len());
        assert!(
            out2[0].len() > 1226 && out2[0].len() < 1232,
            "expected {} - {} samples, got {}",
            1226,
            1232,
            out2[0].len()
        );
    }

    #[test]
    fn make_resampler_fi_32() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedIn::<f32>::new(1.2, params, 1024, 2);
        let waves = audio::dynamic![[0.0f32; 1024]; 2];
        let out = resampler.process(&waves, &bittle::all()).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert!(
            out[0].len() > 1150 && out[0].len() < 1229,
            "expected {} - {} samples, got {}",
            1150,
            1229,
            out[0].len()
        );
        let out2 = resampler.process(&waves, &bittle::all()).unwrap();
        assert_eq!(out2.len(), 2, "Expected {} channels, got {}", 2, out2.len());
        assert!(
            out2[0].len() > 1226 && out2[0].len() < 1232,
            "expected {} - {} samples, got {}",
            1226,
            1232,
            out2[0].len()
        );
    }

    #[test]
    fn make_resampler_fi_skipped() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedIn::<f64>::new(1.2, params, 1024, 2);
        let waves = audio::wrap::dynamic(vec![vec![0.0f64; 1024], Vec::new()]);
        let mask: bittle::FixedSet<u128> = bittle::fixed_set![0];
        let out = resampler.process(&waves, &mask).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].len() > 1150 && out[0].len() < 1250);
        assert!(out[1].is_empty());
        let waves = audio::wrap::dynamic(vec![Vec::new(), vec![0.0f64; 1024]]);
        let mask: bittle::FixedSet<u128> = bittle::fixed_set![1];
        let out = resampler.process(&waves, &mask).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[1].len() > 1150 && out[0].len() < 1250);
        assert!(out[0].is_empty());
    }

    #[test]
    fn make_resampler_fi_downsample() {
        // Replicate settings from reported issue
        let params = InterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 160,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedIn::<f64>::new(16000 as f64 / 96000 as f64, params, 1024, 2);
        let waves = audio::dynamic![[0.0f64; 1024]; 2];
        let out = resampler.process(&waves, &bittle::all()).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert!(
            out[0].len() > 140 && out[0].len() < 200,
            "expected {} - {} samples, got {}",
            140,
            200,
            out[0].len()
        );
        let out2 = resampler.process(&waves, &bittle::all()).unwrap();
        assert_eq!(out2.len(), 2, "Expected {} channels, got {}", 2, out2.len());
        assert!(
            out2[0].len() > 167 && out2[0].len() < 173,
            "expected {} - {} samples, got {}",
            167,
            173,
            out2[0].len()
        );
    }

    #[test]
    fn make_resampler_fi_upsample() {
        // Replicate settings from reported issue
        let params = InterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 160,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedIn::<f64>::new(192000 as f64 / 44100 as f64, params, 1024, 2);
        let waves = audio::dynamic![[0.0f64; 1024]; 2];
        let out = resampler.process(&waves, &bittle::all()).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert!(
            out[0].len() > 3800 && out[0].len() < 4458,
            "expected {} - {} samples, got {}",
            3800,
            4458,
            out[0].len()
        );
        let out2 = resampler.process(&waves, &bittle::all()).unwrap();
        assert_eq!(out2.len(), 2, "Expected {} channels, got {}", 2, out2.len());
        assert!(
            out2[0].len() > 4455 && out2[0].len() < 4461,
            "expected {} - {} samples, got {}",
            4455,
            4461,
            out2[0].len()
        );
    }

    #[test]
    fn make_resampler_fo() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedOut::<f64>::new(1.2, params, 1024, 2);
        let frames = resampler.nbr_frames_needed();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let waves = audio::dynamic![[0.0f64; frames]; 2];
        let out = resampler.process(&waves, &bittle::all()).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }

    #[test]
    fn make_resampler_fo_32() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedOut::<f32>::new(1.2, params, 1024, 2);
        let frames = resampler.nbr_frames_needed();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let waves = audio::dynamic![[0.0f32; frames]; 2];
        let out = resampler.process(&waves, &bittle::all()).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }

    #[test]
    fn make_resampler_fo_skipped() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedOut::<f64>::new(1.2, params, 1024, 2);
        let frames = resampler.nbr_frames_needed();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let mut waves = audio::wrap::dynamic(vec![vec![0.0f64; frames], Vec::new()]);
        waves.as_mut()[0][100] = 3.0;
        let mask: bittle::FixedSet<u128> = bittle::fixed_set![0];
        let out = resampler.process(&waves, &mask).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
        assert!(out[1].is_empty());
        println!("{:?}", out[0]);
        let summed = out[0].iter().sum::<f64>();
        println!("sum: {}", summed);
        assert!(summed < 4.0);
        assert!(summed > 2.0);

        let frames = resampler.nbr_frames_needed();
        let mut waves = audio::wrap::dynamic(vec![Vec::new(), vec![0.0f64; frames]]);
        waves.as_mut()[1][10] = 3.0;
        let mask: bittle::FixedSet<u128> = bittle::fixed_set![1];
        let out = resampler.process(&waves, &mask).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].len(), 1024);
        assert!(out[0].is_empty());
        let summed = out[1].iter().sum::<f64>();
        assert!(summed < 4.0);
        assert!(summed > 2.0);
    }

    #[test]
    fn make_resampler_fo_downsample() {
        let params = InterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 160,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedOut::<f64>::new(0.125, params, 1024, 2);
        let frames = resampler.nbr_frames_needed();
        println!("{}", frames);
        assert!(
            frames > 8192 && frames < 9000,
            "expected {}..{} samples, got {}",
            8192,
            9000,
            frames
        );
        let waves = audio::dynamic![[0.0f64; frames]; 2];
        let out = resampler.process(&waves, &bittle::all()).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert_eq!(
            out[0].len(),
            1024,
            "Expected {} frames, got {}",
            1024,
            out[0].len()
        );
        let frames2 = resampler.nbr_frames_needed();
        assert!(
            frames2 > 8189 && frames2 < 8195,
            "expected {}..{} samples, got {}",
            8189,
            8195,
            frames2
        );
        let waves2 = audio::dynamic![[0.0f64; frames2]; 2];
        let out2 = resampler.process(&waves2, &bittle::all()).unwrap();
        assert_eq!(
            out2[0].len(),
            1024,
            "Expected {} frames, got {}",
            1024,
            out2[0].len()
        );
    }

    #[test]
    fn make_resampler_fo_upsample() {
        let params = InterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 160,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedOut::<f64>::new(8.0, params, 1024, 2);
        let frames = resampler.nbr_frames_needed();
        println!("{}", frames);
        assert!(
            frames > 128 && frames < 300,
            "expected {}..{} samples, got {}",
            140,
            200,
            frames
        );
        let waves = audio::dynamic![[0.0f64; frames]; 2];
        let out = resampler.process(&waves, &bittle::all()).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert_eq!(
            out[0].len(),
            1024,
            "Expected {} frames, got {}",
            1024,
            out[0].len()
        );
        let frames2 = resampler.nbr_frames_needed();
        assert!(
            frames2 > 125 && frames2 < 131,
            "expected {}..{} samples, got {}",
            125,
            131,
            frames2
        );
        let waves2 = audio::dynamic![[0.0f64; frames2]; 2];
        let out2 = resampler.process(&waves2, &bittle::all()).unwrap();
        assert_eq!(
            out2[0].len(),
            1024,
            "Expected {} frames, got {}",
            1024,
            out2[0].len()
        );
    }

    #[test]
    fn test_boxed_resampler() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };

        let mut resampler: Box<dyn crate::DynResampler<f64, audio::buf::Dynamic<f64>>> =
            Box::new(SincFixedOut::<f64>::new(1.2, params, 1024, 2));
        let frames = resampler.nbr_frames_needed();
        let waves = audio::dynamic![[0.0f64; frames]; 2];
        let _ = resampler.process(&waves, &bittle::fixed_set![]).unwrap();
    }
}
