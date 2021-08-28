use crate::Sample;
use audio_core::{ChannelMut, LinearChannel};
use num_traits::Zero;

/// Interpolation methods that can be selected. For asynchronous interpolation where the
/// ratio between inut and output sample rates can be any number, it's not possible to
/// pre-calculate all the needed interpolation filters.
/// Instead they have to be computed as needed, which becomes impractical since the
/// sincs are very expensive to generate in terms of cpu time.
/// It's more efficient to combine the sinc filters with some other interpolation technique.
/// Then sinc filters are used to provide a fixed number of interpolated points between input samples,
/// and then the new value is calculated by interpolation between those points.
#[derive(Debug, Clone, Copy)]
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

impl InterpolationType {
    pub(crate) fn apply_to<'a, I, O, M: ?Sized>(
        self,
        buf: &'a I,
        indexer: impl Iterator<Item = (usize, f64)>,
        oversampling_factor: usize,
        sinc_len: usize,
        interpolator: &dyn crate::asynchro::SincInterpolator<I::Sample>,
        mut wave_out: O,
        mask: &M,
    ) where
        I: ?Sized + audio_core::Buf,
        I::Sample: Sample,
        I::Channel<'a>: audio_core::LinearChannel,
        O: audio_core::BufMut<Sample = I::Sample>,
        M: bittle::Mask,
    {
        use crate::interpolation::{get_nearest_time, get_nearest_times_2, get_nearest_times_4};

        match self {
            InterpolationType::Cubic => {
                let mut points = [I::Sample::zero(); 4];
                let mut nearest = [(0isize, 0isize); 4];

                for (n, idx) in indexer {
                    get_nearest_times_4(idx, oversampling_factor as isize, &mut nearest);
                    let frac = idx * oversampling_factor as f64
                        - (idx * oversampling_factor as f64).floor();
                    let frac_offset = I::Sample::coerce(frac);

                    for (mut wave_out, buf) in
                        mask.join(wave_out.iter_channels_mut().zip(buf.iter_channels()))
                    {
                        for (n, p) in nearest.iter().zip(points.iter_mut()) {
                            *p = interpolator.get_sinc_interpolated(
                                buf.as_linear_channel(),
                                (n.0 + 2 * sinc_len as isize) as usize,
                                n.1 as usize,
                            );
                        }

                        if let Some(f) = wave_out.get_mut(n) {
                            *f = interp_cubic(frac_offset, &points);
                        }
                    }
                }
            }
            InterpolationType::Linear => {
                let mut points = [I::Sample::zero(); 2];
                let mut nearest = [(0isize, 0isize); 2];

                for (n, idx) in indexer {
                    get_nearest_times_2(idx, oversampling_factor as isize, &mut nearest);

                    let frac = idx * oversampling_factor as f64
                        - (idx * oversampling_factor as f64).floor();
                    let frac_offset = I::Sample::coerce(frac);

                    for (mut wave_out, buf) in
                        mask.join(wave_out.iter_channels_mut().zip(buf.iter_channels()))
                    {
                        for (n, p) in nearest.iter().zip(points.iter_mut()) {
                            *p = interpolator.get_sinc_interpolated(
                                buf.as_linear_channel(),
                                (n.0 + 2 * sinc_len as isize) as usize,
                                n.1 as usize,
                            );
                        }

                        if let Some(f) = wave_out.get_mut(n) {
                            *f = interp_lin(frac_offset, &points);
                        }
                    }
                }
            }
            InterpolationType::Nearest => {
                for (n, idx) in indexer {
                    let nearest = get_nearest_time(idx, oversampling_factor as isize);

                    for (mut wave_out, buf) in
                        mask.join(wave_out.iter_channels_mut().zip(buf.iter_channels()))
                    {
                        let point = interpolator.get_sinc_interpolated(
                            buf.as_linear_channel(),
                            (nearest.0 + 2 * sinc_len as isize) as usize,
                            nearest.1 as usize,
                        );

                        if let Some(f) = wave_out.get_mut(n) {
                            *f = point;
                        }
                    }
                }
            }
        }
    }
}

/// Indexes from a given start, to
pub struct RatioIndexer {
    pub(crate) current: f64,
    t_ratio: f64,
    pub(crate) index: usize,
    end: usize,
}

impl RatioIndexer {
    pub(crate) fn new(current: f64, t_ratio: f64, end: usize) -> Self {
        Self {
            current,
            t_ratio,
            index: 0,
            end,
        }
    }
}

impl Iterator for RatioIndexer {
    type Item = (usize, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.end {
            return None;
        }

        let next = (self.index, self.current);
        self.current += self.t_ratio;
        self.index += 1;
        Some(next)
    }
}

/// Indexes from a given start, to
pub struct SpanIndexer {
    pub(crate) current: f64,
    t_ratio: f64,
    pub(crate) index: usize,
    end: f64,
}

impl SpanIndexer {
    pub(crate) fn new(current: f64, t_ratio: f64, end: f64) -> Self {
        Self {
            current,
            t_ratio,
            index: 0,
            end,
        }
    }
}

impl Iterator for SpanIndexer {
    type Item = (usize, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.end {
            return None;
        }

        let next = (self.index, self.current);
        self.current += self.t_ratio;
        self.index += 1;
        Some(next)
    }
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
