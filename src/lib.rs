//! An audio sample rate conversion library for Rust.
//!
//! This library provides resamplers to process audio in chunks.
//!
//! The ratio between input and output sample rates is completely free.
//! Implementations are available that accept a fixed length input
//! while returning a variable length output, and vice versa.
//!
//! ## Asynchronous resampling
//! The resampling is based on band-limited interpolation using sinc
//! interpolation filters. The sinc interpolation upsamples by an adjustable factor,
//! and then the new sample points are calculated by interpolating between these points.
//! The resampling ratio can be updated at any time.
//!
//! ## Synchronous resampling
//! Synchronous resampling is implemented via FFT. The data is FFT:ed, the spectrum modified,
//! and then inverse FFT:ed to get the resampled data.
//! This type of resampler is considerably faster but doesn't support changing the resampling ratio.
//!
//! ## Documentation
//!
//! The full documentation can be generated by rustdoc. To generate and view it run:
//! ```text
//! cargo doc --open
//! ```
//!
//! ## Example
//! Resample a single chunk of a dummy audio file from 44100 to 48000 Hz.
//! See also the "fixedin64" example that can be used to process a file from disk.
//! ```
//! use rubato::{Resampler, SincFixedIn, InterpolationType, InterpolationParameters, WindowFunction};
//! let params = InterpolationParameters {
//!     sinc_len: 256,
//!     f_cutoff: 0.95,
//!     interpolation: InterpolationType::Nearest,
//!     oversampling_factor: 160,
//!     window: WindowFunction::BlackmanHarris2,
//! };
//! let mut resampler = SincFixedIn::<f64>::new(
//!     48000 as f64 / 44100 as f64,
//!     params,
//!     1024,
//!     2,
//! );
//!
//! let waves_in = vec![vec![0.0f64; 1024];2];
//! let waves_out = resampler.process(&waves_in).unwrap();
//! ```
//!
//! ## Compatibility
//!
//! The `rubato` crate requires rustc version 1.40 or newer.

mod interpolation;
mod sinc;
#[cfg(target_arch = "x86_64")]
mod interpolator_sse;
mod synchro;
mod windows;
//mod sseasync;
mod asynchro;
pub use crate::synchro::{FftFixedIn, FftFixedInOut, FftFixedOut};
//pub use crate::sseasync::{SseSincFixedIn, SseSincFixedOut};
pub use crate::asynchro::{SincFixedIn, SincFixedOut};
pub use crate::windows::WindowFunction;

//use crate::interpolation::*;
//use crate::sinc::make_sincs;
//use num_traits::Float;
use std::error;
use std::fmt;

#[macro_use]
extern crate log;

type Res<T> = Result<T, Box<dyn error::Error>>;

/// Custom error returned by resamplers
#[derive(Debug)]
pub struct ResamplerError {
    desc: String,
}

impl fmt::Display for ResamplerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.desc)
    }
}

impl error::Error for ResamplerError {
    fn description(&self) -> &str {
        &self.desc
    }
}

impl ResamplerError {
    pub fn new(desc: &str) -> Self {
        ResamplerError {
            desc: desc.to_owned(),
        }
    }
}

/// A struct holding the parameters for interpolation.
#[derive(Debug)]
pub struct InterpolationParameters {
    /// Length of the windowed sinc interpolation filter.
    /// Higher values can allow a higher cut-off frequency leading to less high frequency roll-off
    /// at the expense of higher cpu usage. 256 is a good starting point.
    /// The value will be rounded up to the nearest multiple of 8.
    pub sinc_len: usize,
    /// Relative cutoff frequency of the sinc interpolation filter
    /// (relative to the lowest one of fs_in/2 or fs_out/2). Start at 0.95, and increase if needed.
    pub f_cutoff: f32,
    /// The number of intermediate points to use for interpolation.
    /// Higher values use more memory for storing the sinc filters.
    /// Only the points actually needed are calculated dusing processing
    /// so a larger number does not directly lead to higher cpu usage.
    /// But keeping it down helps in keeping the sincs in the cpu cache. Start at 128.
    pub oversampling_factor: usize,
    /// Interpolation type, see `InterpolationType`
    pub interpolation: InterpolationType,
    /// Window function to use.
    pub window: WindowFunction,
}

/// Interpolation methods that can be selected. For asynchronous interpolation where the
/// ratio between inut and output sample rates can be any number, it's not possible to
/// pre-calculate all the needed interpolation filters.
/// Instead they have to be computed as needed, which becomes impractical since the
/// sincs are very expensive to generate in terms of cpu time.
/// It's more efficient to combine the sinc filters with some other interpolation technique.
/// Then sinc filters are used to provide a fixed number of interpolated points between input samples,
/// and then the new value is calculated by interpolation between those points.

#[derive(Debug)]
pub enum InterpolationType {
    /// For cubic interpolation, the four nearest intermediate points are calculated
    /// using sinc interpolation.
    /// Then a cubic polynomial is fitted to these points, and is then used to calculate the new sample value.
    /// The computation time as about twice the one for linear interpolation,
    /// but it requires much fewer intermediate points for a good result.
    Cubic,
    /// With linear interpolation the new sample value is calculated by linear interpolation
    /// between the two nearest points.
    /// This requires two intermediate points to be calcuated using sinc interpolation,
    /// and te output is a weighted average of these two.
    /// This is relatively fast, but needs a large number of intermediate points to
    /// push the resampling artefacts below the noise floor.
    Linear,
    /// The Nearest mode doesn't do any interpolation, but simply picks the nearest intermediate point.
    /// This is useful when the nearest point is actually the correct one, for example when upsampling by a factor 2,
    /// like 48kHz->96kHz.
    /// Then setting the oversampling_factor to 2, and using Nearest mode,
    /// no unneccesary computations are performed and the result is the same as for synchronous resampling.
    /// This also works for other ratios that can be expressed by a fraction. For 44.1kHz -> 48 kHz,
    /// setting oversampling_factor to 160 gives the desired result (since 48kHz = 160/147 * 44.1kHz).
    Nearest,
}

/// A resampler that us used to resample a chunk of audio to a new sample rate.
/// The rate can be adjusted as required.
pub trait Resampler<T> {
    /// Resample a chunk of audio. Input and output data is stored in a vector,
    /// where each element contains a vector with all samples for a single channel.
    fn process(&mut self, wave_in: &[Vec<T>]) -> Res<Vec<Vec<T>>>;

    /// Update the resample ratio.
    fn set_resample_ratio(&mut self, new_ratio: f64) -> Res<()>;

    /// Update the resample ratio relative to the original one.
    fn set_resample_ratio_relative(&mut self, rel_ratio: f64) -> Res<()>;

    /// Query for the number of frames needed for the next call to "process".
    fn nbr_frames_needed(&self) -> usize;
}

