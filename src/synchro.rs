use crate::sinc::make_sincs;
use crate::windows::WindowFunction;
use audio_core::{Channel, ChannelMut};
use num_complex::Complex;
use num_integer as integer;
use num_traits::Zero;
use std::sync::Arc;

use crate::error::{ResampleError, ResampleResult};
use crate::{Resampler, Sample};
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// A helper for resampling a single chunk of data.
struct FftResampler<T> {
    fft_size_in: usize,
    fft_size_out: usize,
    filter_f: Vec<Complex<T>>,
    fft: Arc<dyn RealToComplex<T>>,
    ifft: Arc<dyn ComplexToReal<T>>,
    scratch_fw: Vec<Complex<T>>,
    scratch_inv: Vec<Complex<T>>,
    input_buf: Vec<T>,
    input_f: Vec<Complex<T>>,
    output_f: Vec<Complex<T>>,
    output_buf: Vec<T>,
}

/// A synchronous resampler that needs a fixed number of audio frames for input
/// and returns a variable number of frames.
///
/// The resampling is done by FFT:ing the input data. The spectrum is then extended or
/// truncated as well as multiplied with an antialiasing filter
/// before it's inverse transformed to get the resampled waveforms.
pub struct FftFixedIn<T>
where
    T: audio_core::Sample,
{
    nbr_channels: usize,
    chunk_size_in: usize,
    fft_size_in: usize,
    fft_size_out: usize,
    overlaps: audio::buf::Dynamic<T>,
    input_buffers: audio::buf::Dynamic<T>,
    saved_frames: usize,
    resampler: FftResampler<T>,
}

/// A synchronous resampler that needs a varying number of audio frames for input
/// and returns a fixed number of frames.
///
/// The resampling is done by FFT:ing the input data. The spectrum is then extended or
/// truncated as well as multiplied with an antialiasing filter
/// before it's inverse transformed to get the resampled waveforms.
pub struct FftFixedOut<T>
where
    T: Sample,
{
    nbr_channels: usize,
    chunk_size_out: usize,
    fft_size_in: usize,
    fft_size_out: usize,
    overlaps: audio::buf::Dynamic<T>,
    output_buffers: audio::buf::Dynamic<T>,
    saved_frames: usize,
    frames_needed: usize,
    resampler: FftResampler<T>,
}

/// A synchronous resampler that accepts a fixed number of audio frames for input
/// and returns a fixed number of frames.
///
/// The resampling is done by FFT:ing the input data. The spectrum is then extended or
/// truncated as well as multiplied with an antialiasing filter
/// before it's inverse transformed to get the resampled waveforms.
pub struct FftFixedInOut<T>
where
    T: Sample,
{
    nbr_channels: usize,
    chunk_size_in: usize,
    chunk_size_out: usize,
    fft_size_in: usize,
    overlaps: audio::buf::Dynamic<T>,
    resampler: FftResampler<T>,
}

impl<T> FftResampler<T>
where
    T: Sample,
{
    //
    pub fn new(fft_size_in: usize, fft_size_out: usize) -> Self {
        // calculate antialiasing cutoff
        let cutoff = if fft_size_in > fft_size_out {
            0.4f32.powf(16.0 / fft_size_in as f32) * fft_size_out as f32 / fft_size_in as f32
        } else {
            0.4f32.powf(16.0 / fft_size_in as f32)
        };
        debug!(
            "Create new FftResampler, fft_size_in: {}, fft_size_out: {}, cutoff: {}",
            fft_size_in, fft_size_out, cutoff
        );
        let sinc = make_sincs::<T>(fft_size_in, 1, cutoff, WindowFunction::BlackmanHarris2);
        let mut filter_t: Vec<T> = vec![T::zero(); 2 * fft_size_in];
        let mut filter_f: Vec<Complex<T>> = vec![Complex::zero(); fft_size_in + 1];
        for (n, f) in filter_t.iter_mut().enumerate().take(fft_size_in) {
            *f = sinc[0][n] / T::coerce(2 * fft_size_in);
        }

        let input_f: Vec<Complex<T>> = vec![Complex::zero(); fft_size_in + 1];
        let input_buf: Vec<T> = vec![T::zero(); 2 * fft_size_in];
        let output_f: Vec<Complex<T>> = vec![Complex::zero(); fft_size_out + 1];
        let output_buf: Vec<T> = vec![T::zero(); 2 * fft_size_out];
        let mut planner = RealFftPlanner::<T>::new();
        let fft = planner.plan_fft_forward(2 * fft_size_in);
        let ifft = planner.plan_fft_inverse(2 * fft_size_out);
        fft.process(&mut filter_t, &mut filter_f).unwrap();
        let scratch_fw = fft.make_scratch_vec();
        let scratch_inv = ifft.make_scratch_vec();

        FftResampler {
            fft_size_in,
            fft_size_out,
            filter_f,
            fft,
            ifft,
            scratch_fw,
            scratch_inv,
            input_buf,
            input_f,
            output_f,
            output_buf,
        }
    }

    /// Resample a small chunk
    fn resample_unit(
        &mut self,
        overlap: &mut [T],
        inp: impl Channel<Sample = T>,
        out: impl ChannelMut<Sample = T>,
    ) {
        audio::channel::copy(
            inp,
            audio::channel::LinearChannelMut::new(&mut self.input_buf[..self.fft_size_in]),
        );

        for item in self
            .input_buf
            .iter_mut()
            .skip(self.fft_size_in)
            .take(self.fft_size_in)
        {
            *item = T::zero();
        }

        // FFT and store result in history, update index
        self.fft
            .process_with_scratch(&mut self.input_buf, &mut self.input_f, &mut self.scratch_fw)
            .unwrap();

        // multiply with filter FT
        self.input_f
            .iter_mut()
            .take(self.fft_size_in + 1)
            .zip(self.filter_f.iter())
            .for_each(|(spec, filt)| *spec *= filt);
        let new_len = if self.fft_size_in < self.fft_size_out {
            self.fft_size_in + 1
        } else {
            self.fft_size_out
        };

        // copy to modified spectrum
        self.output_f[0..new_len].copy_from_slice(&self.input_f[0..new_len]);
        for val in self.output_f[new_len..].iter_mut() {
            *val = Complex::zero();
        }
        //self.output_f[self.fft_size_out] = self.input_f[self.fft_size_in];

        // IFFT result, store result and overlap
        self.ifft
            .process_with_scratch(
                &mut self.output_f,
                &mut self.output_buf,
                &mut self.scratch_inv,
            )
            .unwrap();

        let it = self
            .output_buf
            .iter()
            .zip(overlap.iter())
            .map(|(a, b)| *a + *b)
            .take(self.fft_size_out);
        audio::channel::copy_iter(it, out);
        overlap.copy_from_slice(&self.output_buf[self.fft_size_out..]);
    }
}

impl<T> FftFixedInOut<T>
where
    T: Sample,
{
    /// Create a new FftFixedInOut
    ///
    /// Parameters are:
    /// - `fs_in`: Input sample rate.
    /// - `fs_out`: Output sample rate.
    /// - `chunk_size_in`: desired length of input data in frames, actual value may be different.
    /// - `nbr_channels`: number of channels in input/output.
    pub fn new(fs_in: usize, fs_out: usize, chunk_size_in: usize, nbr_channels: usize) -> Self {
        debug!(
            "Create new FftFixedInOut, fs_in: {}, fs_out: {} chunk_size_in: {}, channels: {}",
            fs_in, fs_out, chunk_size_in, nbr_channels
        );

        let gcd = integer::gcd(fs_in, fs_out);
        let min_chunk_out = fs_out / gcd;
        let wanted = chunk_size_in;
        let fft_chunks = (wanted as f32 / min_chunk_out as f32).ceil() as usize;
        let fft_size_out = fft_chunks * fs_out / gcd;
        let fft_size_in = fft_chunks * fs_in / gcd;

        let resampler = FftResampler::<T>::new(fft_size_in, fft_size_out);

        let overlaps = audio::buf::Dynamic::with_topology(nbr_channels, fft_size_out);

        FftFixedInOut {
            nbr_channels,
            chunk_size_in: fft_size_in,
            chunk_size_out: fft_size_out,
            fft_size_in,
            overlaps,
            resampler,
        }
    }
}

impl<T> Resampler<T> for FftFixedInOut<T>
where
    T: Sample,
{
    /// Resample a chunk of audio. The input and output lengths are fixed.
    /// If the waveform for a channel is empty, this channel will be ignored and produce a
    /// corresponding empty output waveform.
    /// # Errors
    ///
    /// The function returns an error if the size of the input data is not equal
    /// to the number of channels and input size defined when creating the instance.
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

        for (chan, wave) in mask.join(wave_in.iter_channels().enumerate()) {
            if wave.len() != self.chunk_size_in {
                return Err(ResampleError::WrongNumberOfFrames {
                    channel: chan,
                    expected: self.chunk_size_in,
                    actual: wave.len(),
                });
            }
        }

        wave_out.resize_topology(wave_in.channels(), self.chunk_size_out);

        for ((wave_out, wave_in), mut overlap) in mask.join(
            wave_out
                .iter_channels_mut()
                .zip(wave_in.iter_channels())
                .zip(self.overlaps.iter_channels_mut()),
        ) {
            self.resampler
                .resample_unit(overlap.as_mut(), wave_in, wave_out)
        }

        Ok(())
    }

    /// Query for the number of frames needed for the next call to "process".
    fn nbr_frames_needed(&self) -> usize {
        self.fft_size_in
    }

    /// Update the resample ratio. This is not supported by this resampler and
    /// always returns an error.
    fn set_resample_ratio(&mut self, _new_ratio: f64) -> ResampleResult<()> {
        Err(ResampleError::SyncNotAdjustable)
    }

    /// Update the resample ratio relative to the original one. This is not
    /// supported by this resampler and always returns an error.
    fn set_resample_ratio_relative(&mut self, _rel_ratio: f64) -> ResampleResult<()> {
        Err(ResampleError::SyncNotAdjustable)
    }
}

impl<T> FftFixedOut<T>
where
    T: Sample,
{
    /// Create a new FftFixedOut
    ///
    /// Parameters are:
    /// - `fs_in`: Input sample rate.
    /// - `fs_out`: Output sample rate.
    /// - `chunk_size_out`: length of output data in frames.
    /// - `sub_chunks`: desired number of subchunks for processing, actual number may be different.
    /// - `nbr_channels`: number of channels in input/output.
    pub fn new(
        fs_in: usize,
        fs_out: usize,
        chunk_size_out: usize,
        sub_chunks: usize,
        nbr_channels: usize,
    ) -> Self {
        let gcd = integer::gcd(fs_in, fs_out);
        let min_chunk_out = fs_out / gcd;
        let wanted_subsize = chunk_size_out / sub_chunks;
        let fft_chunks = (wanted_subsize as f32 / min_chunk_out as f32).ceil() as usize;
        let fft_size_out = fft_chunks * fs_out / gcd;
        let fft_size_in = fft_chunks * fs_in / gcd;

        let resampler = FftResampler::<T>::new(fft_size_in, fft_size_out);

        debug!(
            "Create new FftFixedOut, fs_in: {}, fs_out: {} chunk_size_in: {}, channels: {}, fft_size_in: {}, fft_size_out: {}",
            fs_in, fs_out, chunk_size_out, nbr_channels, fft_size_in, fft_size_out
        );

        let overlaps = audio::buf::Dynamic::with_topology(nbr_channels, fft_size_out);
        let output_buffers =
            audio::buf::Dynamic::with_topology(nbr_channels, chunk_size_out + fft_size_out);

        let saved_frames = 0;
        let chunks_needed = (chunk_size_out as f32 / fft_size_out as f32).ceil() as usize;
        let frames_needed = chunks_needed * fft_size_in;

        FftFixedOut {
            nbr_channels,
            chunk_size_out,
            fft_size_in,
            fft_size_out,
            overlaps,
            output_buffers,
            saved_frames,
            frames_needed,
            resampler,
        }
    }
}

impl<T> Resampler<T> for FftFixedOut<T>
where
    T: Sample,
{
    /// Resample a chunk of audio. The required input length is provided by
    /// the "nbr_frames_needed" function, and the output length is fixed.
    /// If the waveform for a channel is empty, this channel will be ignored and produce a
    /// corresponding empty output waveform.
    /// # Errors
    ///
    /// The function returns an error if the length of the input data is not
    /// equal to the number of channels defined when creating the instance,
    /// and the number of audio frames given by "nbr_frames_needed".
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

        for (chan, wave) in mask.join(wave_in.iter_channels().enumerate()) {
            if wave.len() != self.frames_needed {
                return Err(ResampleError::WrongNumberOfFrames {
                    channel: chan,
                    expected: self.frames_needed,
                    actual: wave.len(),
                });
            }
        }

        wave_out.resize_topology(self.nbr_channels, self.output_buffers.frames());

        let it = wave_out
            .iter_channels_mut()
            .zip(self.output_buffers.iter_channels())
            .zip(wave_in.iter_channels())
            .zip(self.overlaps.iter_channels_mut());

        let fft_size_out = self.resampler.fft_size_out;

        for (((mut wave_out, output_buffer), wave_in), mut overlap) in mask.join(it) {
            audio::channel::copy(output_buffer, wave_out.as_channel_mut());

            let chunk = self.fft_size_out;
            let chunks = chunks(wave_in.len(), chunk);

            let mut wave_out = wave_out.skip(self.saved_frames);

            for n in 0..chunks {
                let wave_out = &mut wave_out;

                self.resampler.resample_unit(
                    overlap.as_mut(),
                    wave_in.as_channel().skip(n * chunk).limit(chunk),
                    wave_out
                        .as_channel_mut()
                        .skip(n * fft_size_out)
                        .limit(fft_size_out),
                );
            }
        }

        let processed_frames =
            self.saved_frames + self.fft_size_out * (self.frames_needed / self.fft_size_in);

        // save extra frames for next round
        self.saved_frames = processed_frames - self.chunk_size_out;

        if processed_frames > self.chunk_size_out {
            for (wave_out, out) in mask.join(
                wave_out
                    .iter_channels()
                    .zip(self.output_buffers.iter_channels_mut()),
            ) {
                audio::channel::copy(
                    wave_out.skip(self.chunk_size_out).limit(self.saved_frames),
                    out.limit(self.saved_frames),
                );
            }
        }

        wave_out.resize(self.chunk_size_out);

        //calculate number of needed frames from next round
        let frames_needed_out = if self.chunk_size_out > self.saved_frames {
            self.chunk_size_out - self.saved_frames
        } else {
            0
        };
        let chunks_needed = (frames_needed_out as f32 / self.fft_size_out as f32).ceil() as usize;
        self.frames_needed = chunks_needed * self.fft_size_in;
        Ok(())
    }

    /// Query for the number of frames needed for the next call to "process".
    fn nbr_frames_needed(&self) -> usize {
        self.frames_needed
    }

    /// Update the resample ratio. This is not supported by this resampler and
    /// always returns an error.
    fn set_resample_ratio(&mut self, _new_ratio: f64) -> ResampleResult<()> {
        Err(ResampleError::SyncNotAdjustable)
    }

    /// Update the resample ratio relative to the original one. This is not
    /// supported by this resampler and always returns an error.
    fn set_resample_ratio_relative(&mut self, _rel_ratio: f64) -> ResampleResult<()> {
        Err(ResampleError::SyncNotAdjustable)
    }
}

impl<T> FftFixedIn<T>
where
    T: Sample,
{
    /// Create a new FftFixedOut
    ///
    /// Parameters are:
    /// - `fs_in`: Input sample rate.
    /// - `fs_out`: Output sample rate.
    /// - `chunk_size_out`: length of output data in frames.
    /// - `sub_chunks`: desired number of subchunks for processing, actual number used may be different.
    /// - `nbr_channels`: number of channels in input/output.
    pub fn new(
        fs_in: usize,
        fs_out: usize,
        chunk_size_in: usize,
        sub_chunks: usize,
        nbr_channels: usize,
    ) -> Self {
        let gcd = integer::gcd(fs_in, fs_out);
        let min_chunk_in = fs_in / gcd;
        let wanted_subsize = chunk_size_in / sub_chunks;
        let fft_chunks = (wanted_subsize as f32 / min_chunk_in as f32).ceil() as usize;
        let fft_size_out = fft_chunks * fs_out / gcd;
        let fft_size_in = fft_chunks * fs_in / gcd;

        let resampler = FftResampler::<T>::new(fft_size_in, fft_size_out);
        debug!(
            "Create new FftFixedOut, fs_in: {}, fs_out: {} chunk_size_in: {}, channels: {}, fft_size_in: {}, fft_size_out: {}",
            fs_in, fs_out, chunk_size_in, nbr_channels, fft_size_in, fft_size_out
        );

        let overlaps = audio::buf::Dynamic::with_topology(nbr_channels, fft_size_out);
        let input_buffers =
            audio::buf::Dynamic::with_topology(nbr_channels, chunk_size_in + fft_size_out);

        let saved_frames = 0;

        FftFixedIn {
            nbr_channels,
            chunk_size_in,
            fft_size_in,
            fft_size_out,
            overlaps,
            input_buffers,
            saved_frames,
            resampler,
        }
    }
}

impl<T> Resampler<T> for FftFixedIn<T>
where
    T: Sample,
{
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

        for (chan, wave) in mask.join(wave_in.iter_channels().enumerate()) {
            if wave.len() != self.chunk_size_in {
                return Err(ResampleError::WrongNumberOfFrames {
                    channel: chan,
                    expected: self.chunk_size_in,
                    actual: wave.len(),
                });
            }
        }

        let mut input_temp = audio::buf::Sequential::with_topology(
            self.nbr_channels,
            self.saved_frames + self.chunk_size_in,
        );

        let it = input_temp
            .iter_channels_mut()
            .zip(self.input_buffers.iter_channels())
            .zip(wave_in.iter_channels());

        // copy new samples to input buffer
        for ((mut temp, input), wave) in mask.join(it) {
            audio::channel::copy(
                input.limit(self.saved_frames),
                temp.as_channel_mut().limit(self.saved_frames),
            );
            audio::channel::copy(wave, temp.skip(self.saved_frames).limit(self.chunk_size_in));
        }

        self.saved_frames += self.chunk_size_in;

        let nbr_chunks_ready =
            (self.saved_frames as f32 / self.fft_size_in as f32).floor() as usize;

        wave_out.resize_topology(wave_in.channels(), nbr_chunks_ready * self.fft_size_out);

        let it = wave_out
            .iter_channels_mut()
            .zip(input_temp.iter_channels())
            .zip(self.overlaps.iter_channels_mut());

        let fft_size_out = self.resampler.fft_size_out;

        for ((wave_out, input_temp), mut overlap) in mask.join(it) {
            let mut wave_out = wave_out.skip(self.saved_frames);
            let chunks = chunks(wave_out.len(), self.fft_size_out);

            for n in 0..chunks {
                self.resampler.resample_unit(
                    overlap.as_mut(),
                    input_temp
                        .skip(n * self.fft_size_in)
                        .limit(self.fft_size_in),
                    wave_out
                        .as_channel_mut()
                        .skip(n * fft_size_out)
                        .limit(fft_size_out),
                );
            }
        }

        // save extra frames for next round
        let frames_in_used = nbr_chunks_ready * self.fft_size_in;
        let extra = self.saved_frames.saturating_sub(frames_in_used);

        if self.saved_frames > frames_in_used {
            let it = input_temp
                .iter_channels()
                .zip(self.input_buffers.iter_channels_mut());

            for (inp, out) in mask.join(it) {
                audio::channel::copy(inp.skip(frames_in_used).limit(extra), out);
            }
        }

        self.saved_frames = extra;
        Ok(())
    }

    /// Query for the number of frames needed for the next call to "process".
    fn nbr_frames_needed(&self) -> usize {
        self.chunk_size_in
    }

    /// Update the resample ratio. This is not supported by this resampler and
    /// always returns an error.
    fn set_resample_ratio(&mut self, _new_ratio: f64) -> ResampleResult<()> {
        Err(ResampleError::SyncNotAdjustable)
    }

    /// Update the resample ratio relative to the original one. This is not
    /// supported by this resampler and always returns an error.
    fn set_resample_ratio_relative(&mut self, _rel_ratio: f64) -> ResampleResult<()> {
        Err(ResampleError::SyncNotAdjustable)
    }
}

fn chunks(len: usize, window: usize) -> usize {
    if len % window == 0 {
        len / window
    } else {
        len / window + 1
    }
}

#[cfg(test)]
mod tests {
    use crate::synchro::{FftFixedIn, FftFixedInOut, FftFixedOut, FftResampler};
    use crate::Resampler;

    #[test]
    fn resample_unit() {
        let mut resampler = FftResampler::<f64>::new(147, 1000);
        let mut wave_in = vec![0.0; 147];

        wave_in[0] = 0.3;
        wave_in[1] = 0.7;
        wave_in[2] = 1.0;
        wave_in[3] = 1.0;
        wave_in[4] = 0.7;
        wave_in[5] = 0.3;

        let mut wave_out = vec![0.0; 1000];
        let mut overlap = vec![0.0; 1000];
        resampler.resample_unit(
            &mut overlap,
            audio::channel::LinearChannel::new(&wave_in[..]),
            audio::channel::LinearChannelMut::new(&mut wave_out[..]),
        );
        let vecsum = wave_out.iter().sum::<f64>();
        let maxval = wave_out.iter().cloned().fold(0. / 0., f64::max);
        assert!((vecsum - 4.0 * 1000.0 / 147.0).abs() < 1.0e-6);
        assert!((maxval - 1.0).abs() < 0.1);
    }

    #[test]
    fn make_resampler_fio() {
        // asking for 1024 give the nearest which is 1029 -> 1120
        let mut resampler = FftFixedInOut::<f64>::new(44100, 48000, 1024, 2);
        let frames = resampler.nbr_frames_needed();
        let waves = audio::dynamic![[0.0f64; frames]; 2];
        let out = resampler.process(&waves, &bittle::all()).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1120);
    }

    #[test]
    fn make_resampler_fio_skipped() {
        // asking for 1024 give the nearest which is 1029 -> 1120
        let mut resampler = FftFixedInOut::<f64>::new(44100, 48000, 1024, 2);
        let frames = resampler.nbr_frames_needed();
        let waves = audio::wrap::dynamic(vec![vec![0.0f64; frames], Vec::new()]);
        let mask: bittle::FixedSet<u128> = bittle::fixed_set![0];
        let out = resampler.process(&waves, &mask).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1120);
        assert!(out[1].is_empty());
    }

    #[test]
    fn make_resampler_fo() {
        let mut resampler = FftFixedOut::<f64>::new(44100, 192000, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 294);
        let waves = audio::dynamic![[0.0f64; frames]; 2];
        let out = resampler.process(&waves, &bittle::all()).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }

    #[test]
    fn make_resampler_fo_skipped() {
        let mut resampler = FftFixedOut::<f64>::new(44100, 192000, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 294);
        let waves = audio::wrap::dynamic(vec![vec![0.0f64; frames], Vec::new()]);
        let mask: bittle::FixedSet<u128> = bittle::fixed_set![0];
        let out = resampler.process(&waves, &mask).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
        assert!(out[1].is_empty());
    }

    #[test]
    fn make_resampler_fo_empty() {
        let mut resampler = FftFixedOut::<f64>::new(44100, 192000, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 294);
        let waves = audio::wrap::dynamic(vec![Vec::new(); 2]);
        let out = resampler.process(&waves, &bittle::none()).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].is_empty());
        assert!(out[1].is_empty());
    }

    #[test]
    fn make_resampler_fi() {
        let mut resampler = FftFixedIn::<f64>::new(44100, 48000, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 1024);
        let waves = audio::dynamic![[0.0f64; frames]; 2];
        let out = resampler.process(&waves, &bittle::all()).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 640);
    }

    #[test]
    fn make_resampler_fi_downsample() {
        let mut resampler = FftFixedIn::<f64>::new(48000, 16000, 1200, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 1200);
        let waves = audio::dynamic![[0.0f64; frames]; 2];
        let out = resampler.process(&waves, &bittle::all()).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 400);
    }

    #[test]
    fn make_resampler_fi_skipped() {
        let mut resampler = FftFixedIn::<f64>::new(44100, 48000, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 1024);
        let waves = audio::wrap::dynamic(vec![vec![0.0f64; frames], Vec::new()]);
        let mask: bittle::FixedSet<u128> = bittle::fixed_set![0];
        let out = resampler.process(&waves, &mask).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 640);
        assert!(out[1].is_empty());
    }

    #[test]
    fn make_resampler_fi_empty() {
        let mut resampler = FftFixedIn::<f64>::new(44100, 48000, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 1024);
        let waves = audio::wrap::dynamic(vec![Vec::new(); 2]);
        let out = resampler.process(&waves, &bittle::none()).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].is_empty());
        assert!(out[1].is_empty());
    }

    #[test]
    fn make_resampler_fio_unusualratio() {
        // asking for 1024 give the nearest which is 1029 -> 1120
        let mut resampler = FftFixedInOut::<f64>::new(44100, 44110, 1024, 2);
        let frames = resampler.nbr_frames_needed();
        let waves = audio::dynamic![[0.0f64; frames]; 2];
        let out = resampler.process(&waves, &bittle::all()).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 4411);
    }

    #[test]
    fn make_resampler_fo_unusualratio() {
        let mut resampler = FftFixedOut::<f64>::new(44100, 44110, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 4410);
        let waves = audio::dynamic![[0.0f64; frames]; 2];
        let out = resampler.process(&waves, &bittle::all()).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }
}
