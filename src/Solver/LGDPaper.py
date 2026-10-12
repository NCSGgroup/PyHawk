"""Opt-in paper-style post-KBRR to LGD conversion.

The processing order follows the method used for the comparison study:

    range-rate residual -> zero-phase band-pass -> fourth-order derivative
    -> Ghobadi-Far transfer function

This module is deliberately independent of the production ComputeLGD class.
It operates on an already loaded continuous range-rate segment, so saved Hawk
and CSR products can be processed with exactly the same code.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt


class PaperLGDProcessor:
    """Convert range-rate residuals to LGD using the paper-style chain."""

    def __init__(
            self,
            dt=5.0,
            f_low=0.9e-3,
            f_high=11.0e-3,
            filter_order=3,
            transfer_min_frequency=1.0e-3,
            transfer_factor=3.45e-4,
            transfer_power=-1.04,
            edge_trim_seconds=60.0,
            detrend_input=True):
        self.dt = float(dt)
        self.f_low = float(f_low)
        self.f_high = float(f_high)
        self.filter_order = int(filter_order)
        self.transfer_min_frequency = float(transfer_min_frequency)
        self.transfer_factor = float(transfer_factor)
        self.transfer_power = float(transfer_power)
        self.edge_trim_seconds = float(edge_trim_seconds)
        self.detrend_input = bool(detrend_input)

        nyquist = 0.5 / self.dt
        if not 0.0 < self.f_low < self.f_high < nyquist:
            raise ValueError(
                "Invalid band %.6g-%.6g Hz for dt %.6g s"
                % (self.f_low, self.f_high, self.dt)
            )

        self._sos = butter(
            self.filter_order,
            [self.f_low, self.f_high],
            btype="bandpass",
            fs=1.0 / self.dt,
            output="sos",
        )

    def convert_segment(self, range_rate):
        """Return filtered rate, acceleration and LGD for one gap-free arc.

        The returned arrays retain the original length. The caller should
        remove edge_trim_points samples at both ends when mapping points.
        """
        rate = np.asarray(range_rate, dtype=np.float64).reshape(-1)
        if rate.size < 5:
            raise ValueError("At least five samples are required")
        if not np.all(np.isfinite(rate)):
            raise ValueError("The segment contains non-finite range-rate values")

        conditioned = detrend(rate, type="linear") if self.detrend_input else rate
        filtered_rate = sosfiltfilt(self._sos, conditioned)
        acceleration = self.diff4(filtered_rate, self.dt)
        lgd = self.transfer(acceleration, self.dt)
        return filtered_rate, acceleration, lgd

    @property
    def edge_trim_points(self):
        return max(0, int(np.ceil(self.edge_trim_seconds / self.dt)))

    @staticmethod
    def diff4(value, dt):
        """Fourth-order five-point first derivative with one-sided endpoints."""
        x = np.asarray(value, dtype=np.float64).reshape(-1)
        n = x.size
        if n < 5:
            return np.gradient(x, float(dt))

        y = np.empty_like(x)
        h = float(dt)
        y[2:-2] = (
            -x[4:] + 8.0 * x[3:-1] - 8.0 * x[1:-3] + x[:-4]
        ) / (12.0 * h)
        y[0] = (
            -25.0 * x[0] + 48.0 * x[1] - 36.0 * x[2]
            + 16.0 * x[3] - 3.0 * x[4]
        ) / (12.0 * h)
        y[1] = (
            -3.0 * x[0] - 10.0 * x[1] + 18.0 * x[2]
            - 6.0 * x[3] + x[4]
        ) / (12.0 * h)
        y[-1] = (
            25.0 * x[-1] - 48.0 * x[-2] + 36.0 * x[-3]
            - 16.0 * x[-4] + 3.0 * x[-5]
        ) / (12.0 * h)
        y[-2] = (
            3.0 * x[-1] + 10.0 * x[-2] - 18.0 * x[-3]
            + 6.0 * x[-4] - x[-5]
        ) / (12.0 * h)
        return y

    def transfer(self, acceleration, dt=None):
        """Apply the mirror-padded Ghobadi-Far transfer function."""
        x = np.asarray(acceleration, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return x.copy()
        sample_dt = self.dt if dt is None else float(dt)
        padded = np.concatenate((x, x[::-1]))
        frequency = np.fft.fftfreq(padded.size, d=sample_dt)
        magnitude = np.abs(frequency)
        band = magnitude >= self.transfer_min_frequency
        weight = np.zeros_like(magnitude)
        weight[band] = (
            1.0
            + self.transfer_factor
            * np.power(magnitude[band], self.transfer_power)
        )
        return np.real(np.fft.ifft(np.fft.fft(padded) * weight))[:x.size]


__all__ = ["PaperLGDProcessor"]
