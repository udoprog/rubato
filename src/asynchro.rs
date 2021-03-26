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
        assert!(sinc_len % 8 == 0);
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
    fn post_process<T>(
        &mut self,
        indexer: Self::Indexer,
        sinc_len: usize,
        channels: &[usize],
        wave_out: &mut [Vec<T>],
    );

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

    fn post_process<T>(
        &mut self,
        indexer: Self::Indexer,
        sinc_len: usize,
        _: &[usize],
        _: &mut [Vec<T>],
    ) {
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

    fn post_process<T>(
        &mut self,
        indexer: Self::Indexer,
        _: usize,
        channels: &[usize],
        wave_out: &mut [Vec<T>],
    ) {
        let idx = indexer.current;
        let n = indexer.index;

        // store last index for next iteration
        self.last_index = idx - self.chunk_size as f64;

        for chan in channels {
            //for w in wave_out.iter_mut() {
            wave_out[*chan].truncate(n);
        }
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

/// An asynchronous resampler that accepts a fixed number of audio frames for input
/// and returns a variable number of frames.
///
/// The resampling is done by creating a number of intermediate points (defined by oversampling_factor)
/// by sinc interpolation. The new samples are then calculated by interpolating between these points.
pub type SincFixedOut<T> = Sinc<T, FixedOut>;

pub struct Sinc<T, K> {
    nbr_channels: usize,
    resample_ratio_original: f64,
    interpolator: Box<dyn SincInterpolator<T>>,
    buffer: Vec<Vec<T>>,
    interpolation: InterpolationType,
    kind: K,
}

pub fn make_interpolator<T>(
    sinc_len: usize,
    resample_ratio: f64,
    f_cutoff: f32,
    oversampling_factor: usize,
    window: WindowFunction,
) -> Box<dyn SincInterpolator<T>>
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

/// Perform cubic polynomial interpolation to get value at x.
/// Input points are assumed to be at x = -1, 0, 1, 2
pub(crate) fn interp_cubic<T>(x: T, yvals: &[T; 4]) -> T
where
    T: Sample,
{
    let a0 = yvals[1];
    let a1 = -(T::one() / T::coerce(3.0)) * yvals[0] - T::coerce(0.5) * yvals[1] + yvals[2]
        - (T::one() / T::coerce(6.0)) * yvals[3];
    let a2 = T::coerce(0.5) * (yvals[0] + yvals[2]) - yvals[1];
    let a3 = T::coerce(0.5) * (yvals[1] - yvals[2])
        + (T::one() / T::coerce(6.0)) * (yvals[3] - yvals[0]);
    let x2 = x * x;
    let x3 = x2 * x;
    a0 + a1 * x + a2 * x2 + a3 * x3
}

/// Linear interpolation between two points at x=0 and x=1
pub(crate) fn interp_lin<T>(x: T, yvals: &[T; 2]) -> T
where
    T: Sample,
{
    (T::one() - x) * yvals[0] + x * yvals[1]
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
    fn process(&mut self, wave_in: &[Vec<T>]) -> ResampleResult<Vec<Vec<T>>> {
        if wave_in.len() != self.nbr_channels {
            return Err(ResampleError::WrongNumberOfChannels {
                expected: self.nbr_channels,
                actual: wave_in.len(),
            });
        }

        let p = self.kind.params();

        let mut used_channels = Vec::new();

        for (chan, wave) in wave_in.iter().enumerate() {
            if !wave.is_empty() {
                used_channels.push(chan);
                if wave.len() != p.needed_input {
                    return Err(ResampleError::WrongNumberOfFrames {
                        channel: chan,
                        expected: p.needed_input,
                        actual: wave.len(),
                    });
                }
            }
        }

        let sinc_len = self.interpolator.len();
        let oversampling_factor = self.interpolator.nbr_sincs();

        //update buffer with new data
        for wav in self.buffer.iter_mut() {
            for idx in 0..(2 * sinc_len) {
                wav[idx] = wav[idx + p.current_fill];
            }
        }

        let mut wave_out = vec![Vec::new(); self.nbr_channels];

        for chan in used_channels.iter() {
            for (idx, sample) in wave_in[*chan].iter().enumerate() {
                self.buffer[*chan][idx + 2 * sinc_len] = *sample;
            }

            wave_out[*chan] = vec![T::zero(); p.output_len];
        }

        let mut indexer = self.kind.indexer(sinc_len);

        self.interpolation.apply_to(
            &used_channels,
            &self.buffer,
            &mut indexer,
            oversampling_factor,
            sinc_len,
            self.interpolator.as_ref(),
            &mut wave_out,
        );

        self.kind
            .post_process(indexer, sinc_len, &used_channels, &mut wave_out);

        /*trace!(
            "Resampling channels {:?}, {} frames in, {} frames out",
            used_channels,
            self.chunk_size,
            n,
        );*/
        Ok(wave_out)
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
        interpolation_type: InterpolationType,
        interpolator: Box<dyn SincInterpolator<T>>,
        chunk_size: usize,
        nbr_channels: usize,
    ) -> Self {
        let kind = K::new(chunk_size, resample_ratio, interpolator.len());

        let buffer =
            vec![vec![T::zero(); kind.chunk_size() + 2 * interpolator.len()]; nbr_channels];

        Self {
            nbr_channels,
            resample_ratio_original: resample_ratio,
            interpolator,
            buffer,
            interpolation: interpolation_type,
            kind,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{interp_cubic, interp_lin};
    use crate::asynchro::ScalarInterpolator;
    use crate::asynchro::SincInterpolator;
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
        let waves = vec![vec![0.0f64; 1024]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].len() > 1150 && out[0].len() < 1250);
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
        let waves = vec![vec![0.0f32; 1024]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].len() > 1150 && out[0].len() < 1250);
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
        let waves = vec![vec![0.0f64; 1024], Vec::new()];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].len() > 1150 && out[0].len() < 1250);
        assert!(out[1].is_empty());
        let waves = vec![Vec::new(), vec![0.0f64; 1024]];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[1].len() > 1150 && out[0].len() < 1250);
        assert!(out[0].is_empty());
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
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves).unwrap();
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
        let waves = vec![vec![0.0f32; frames]; 2];
        let out = resampler.process(&waves).unwrap();
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
        let mut waves = vec![vec![0.0f64; frames], Vec::new()];
        waves[0][100] = 3.0;
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
        assert!(out[1].is_empty());
        println!("{:?}", out[0]);
        let summed = out[0].iter().sum::<f64>();
        println!("sum: {}", summed);
        assert!(summed < 4.0);
        assert!(summed > 2.0);

        let frames = resampler.nbr_frames_needed();
        let mut waves = vec![Vec::new(), vec![0.0f64; frames]];
        waves[1][10] = 3.0;
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].len(), 1024);
        assert!(out[0].is_empty());
        let summed = out[1].iter().sum::<f64>();
        assert!(summed < 4.0);
        assert!(summed > 2.0);
    }
}
