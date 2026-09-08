import os
import pathlib
import h5py
import numpy as np

from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

from src.Interface.ArcOperation import ArcData
from src.Preference.EnumType import Payload, SatID
from src.Frame.EOP import EOP
from src.Frame.Frame import Frame
from src.Auxilary.GeoMathKit import GeoMathKit
from src.Auxilary.StandardLGD import StandardLGD
from src.Solver.AdjustSST import RangeRate


class ComputeLGD:
    """
    Compute LGD from KBRR/LRI range-rate residual for a single arc.

    Processing chain:

        KBRR residual
            ↓
        Butterworth band-pass filtering, 0.9–11 mHz
            ↓
        4th-order differentiation
            ↓
        Range acceleration residual
            ↓
        Transfer function
            ↓
        LGD
            ↓
        midpoint lon/lat

    Input:
        RangeRate_{arcNo}.hdf5

    Output:
        LGD_{arcNo}.hdf5

    Output datasets:
        time
        lon
        lat
        lgd
        range_acc
        kbrr_residual
        kbrr_filtered
            Diagnostic range-rate residual filtered in the LGD band.
    """

    def __init__(self, arcNo: int, date_span):
        self.__arcNo = arcNo
        self.__date_span = date_span

        self.__AdjustOrbitConfig = None
        self.__InterfaceConfig = None
        self.__FrameConfig = None

        self.__PathfileConfig = None
        self.__rd = None
        self.__fr = None

        # Han/Tang LGD signal band. The upper bound is payload dependent
        # when left as None: KBR uses 10 mHz, LRI uses 20 mHz.
        self.__f_low = 1.0e-3
        self.__f_high = None
        self.__resolved_f_low = None
        self.__resolved_f_high = None

        # Butterworth order used only for diagnostic range-rate filtering.
        self.__filter_order = 6

        # Gap threshold
        self.__gap_threshold = 20.0

        # Minimum segment length for stable filtering and FFT
        self.__min_segment_length = 300

        # Ghobadi-Far transfer function. It is only valid above 1 mHz.
        self.__transfer_min_frequency = 1.0e-3
        self.__tf_factor = 3.45e-4
        self.__tf_power = -1.04

        # KBRR preprocessing to reduce low-frequency leakage into the LGD
        # band for finite arcs with imperfect endpoints or isolated spikes.
        self.__kbrr_preprocess_enable = True
        self.__kbrr_despike_enable = True
        self.__kbrr_despike_window = 121
        self.__kbrr_despike_sigma = 6.0
        self.__kbrr_remove_low_frequency = True
        self.__kbrr_trend_degree = 2
        self.__kbrr_cpr_period = 5400.0
        self.__kbrr_cpr_harmonics = 2

        # Trim segment endpoints after numerical differentiation. One-sided
        # derivative points and finite-arc FFT endpoints are prone to leakage.
        self.__segment_edge_trim_seconds = 60.0

        # Output directory
        self.__LGDTemp = "../temp/LGD"
        self.__StandardLGDTemp = "../temp/StandardLGD"
        self.__sst_payload = "KBR"

    def configure(
            self,
            AdjustOrbitConfig,
            InterfaceConfig,
            FrameConfig,
            f_low=1.0e-3,
            f_high=None,
            filter_order=6,
            gap_threshold=20.0,
            min_segment_length=300,
            kbrr_preprocess_enable=True,
            kbrr_despike_enable=True,
            kbrr_despike_window=121,
            kbrr_despike_sigma=6.0,
            kbrr_remove_low_frequency=True,
            kbrr_trend_degree=2,
            kbrr_cpr_period=5400.0,
            kbrr_cpr_harmonics=2,
            segment_edge_trim_seconds=60.0):
        """
        Configure LGD computation.
        """

        self.__AdjustOrbitConfig = AdjustOrbitConfig
        self.__InterfaceConfig = InterfaceConfig
        self.__FrameConfig = FrameConfig

        self.__PathfileConfig = self.__AdjustOrbitConfig.PathOfFiles()
        self.__PathfileConfig.__dict__.update(
            self.__AdjustOrbitConfig.PathOfFilesConfig.copy()
        )
        self.__f_low = 1.0e-3 if f_low is None else float(f_low)
        self.__f_high = None if f_high is None else float(f_high)
        self.__resolved_f_low = None
        self.__resolved_f_high = None
        self.__filter_order = filter_order
        self.__gap_threshold = gap_threshold
        self.__min_segment_length = min_segment_length
        self.__kbrr_preprocess_enable = bool(kbrr_preprocess_enable)
        self.__kbrr_despike_enable = bool(kbrr_despike_enable)
        self.__kbrr_despike_window = int(kbrr_despike_window)
        self.__kbrr_despike_sigma = float(kbrr_despike_sigma)
        self.__kbrr_remove_low_frequency = bool(kbrr_remove_low_frequency)
        self.__kbrr_trend_degree = int(kbrr_trend_degree)
        self.__kbrr_cpr_period = float(kbrr_cpr_period)
        self.__kbrr_cpr_harmonics = int(kbrr_cpr_harmonics)
        self.__segment_edge_trim_seconds = float(segment_edge_trim_seconds)

        # Prefer LGDTemp from config if it exists.
        self.__LGDTemp = getattr(
            self.__PathfileConfig,
            "LGDTemp",
            self.__LGDTemp
        )
        self.__StandardLGDTemp = getattr(
            self.__PathfileConfig,
            "StandardLGDTemp",
            self.__StandardLGDTemp
        )

        self.__rd = ArcData(
            interfaceConfig=self.__InterfaceConfig
        )

        eop = EOP().configure(
            frameConfig=self.__FrameConfig
        ).load()

        self.__fr = Frame(eop).configure(
            frameConfig=self.__FrameConfig
        )

        return self

    def compute(self):
        """
        Main entrance.
        Compute LGD for current arcNo and save it.
        """

        time, kbrr_residual = self.__load_kbrr_residual()

        if len(time) < self.__min_segment_length:
            print(
                "Arc %d skipped: too few KBRR samples (%d)."
                % (self.__arcNo, len(time))
            )
            return None

        time_all = []
        residual_all = []
        kbrr_filtered_all = []
        range_acc_all = []
        lgd_raw_all = []
        lgd_all = []
        is_original_all = []

        segments = self.__split_by_gap(time)

        for index in segments:
            if len(index) < self.__min_segment_length:
                continue

            t_seg = time[index]
            rr_seg = kbrr_residual[index]

            rr_seg = self.__fill_nan_by_interp(
                x=rr_seg
            )

            if self.__kbrr_preprocess_enable:
                rr_seg = self.__preprocess_kbrr_segment(
                    time=t_seg,
                    residual=rr_seg
                )

            t_uniform, rr_uniform, is_original, dt = self.__uniform_sample(
                time=t_seg,
                value=rr_seg
            )

            if len(t_uniform) < self.__min_segment_length:
                continue

            range_acc_seg = self.__diff4(
                x=rr_uniform,
                dt=dt
            )

            keep = self.__segment_edge_keep_mask(
                n=len(t_uniform),
                dt=dt
            )

            if np.sum(keep) < self.__min_segment_length:
                continue

            t_uniform = t_uniform[keep]
            rr_uniform = rr_uniform[keep]
            is_original = is_original[keep]
            range_acc_seg = range_acc_seg[keep]

            f_low, f_high = self.__resolve_frequency_band(dt=dt)

            lgd_raw_seg = self.__transfer_function(
                x=range_acc_seg,
                dt=dt,
                f_low=self.__transfer_min_frequency,
                f_high=None
            )

            lgd_seg = self.__transfer_function(
                x=range_acc_seg,
                dt=dt,
                f_low=f_low,
                f_high=f_high
            )

            rr_filtered_seg = self.__bandpass_filter(
                x=rr_uniform,
                dt=dt,
                f_low=f_low,
                f_high=f_high
            )

            time_all.append(t_uniform)
            residual_all.append(rr_uniform)
            kbrr_filtered_all.append(rr_filtered_seg)
            range_acc_all.append(range_acc_seg)
            lgd_raw_all.append(lgd_raw_seg)
            lgd_all.append(lgd_seg)
            is_original_all.append(is_original)

        if len(lgd_all) == 0:
            print(
                "Arc %d skipped: no valid LGD segment."
                % self.__arcNo
            )
            return None

        time = np.concatenate(time_all)
        kbrr_residual = np.concatenate(residual_all)
        kbrr_filtered = np.concatenate(kbrr_filtered_all)
        range_acc = np.concatenate(range_acc_all)
        lgd_raw = np.concatenate(lgd_raw_all)
        lgd = np.concatenate(lgd_all)
        is_original_sample = np.concatenate(is_original_all)

        valid = np.isfinite(lgd)

        if np.sum(valid) == 0:
            print(
                "Arc %d skipped: no valid LGD samples."
                % self.__arcNo
            )
            return None

        time_valid = time[valid]

        lon, lat, pos_a_itrs, pos_b_itrs, los_itrs = self.__compute_midpoint_lonlat(
            time=time_valid
        )

        result = {
            "time": time_valid,
            "lon": lon,
            "lat": lat,
            "pos_a_itrs": pos_a_itrs,
            "pos_b_itrs": pos_b_itrs,
            "los_itrs": los_itrs,
            "lgd": lgd[valid],
            "lgd_unfiltered": lgd_raw[valid],
            "range_acc": range_acc[valid],
            "kbrr_residual": kbrr_residual[valid],
            "kbrr_filtered": kbrr_filtered[valid],
            "is_original_sample": is_original_sample[valid],
        }
        result["altitude"] = StandardLGD.midpoint_altitude(
            pos_a_itrs=result["pos_a_itrs"],
            pos_b_itrs=result["pos_b_itrs"]
        )

        self.__save_result(result)

        StandardLGD().configure(
            filedir=self.__StandardLGDTemp,
            date_span=self.__date_span,
            mission=self.__InterfaceConfig.mission
        ).save_arc(
            gps_time=result["time"],
            longitude=result["lon"],
            latitude=result["lat"],
            altitude=result["altitude"],
            lgd_full=result["lgd_unfiltered"],
            lgd_band=result["lgd"],
            arc_no=self.__arcNo
        )

        return result

    def __rolling_median(self, x, window):
        """
        Edge-padded rolling median used only for robust spike detection.
        """

        x = np.asarray(
            x,
            dtype=np.float64
        )

        n = len(x)
        window = int(window)
        if window < 3 or n < 3:
            return x.copy()

        if window % 2 == 0:
            window += 1

        if window > n:
            window = n if n % 2 == 1 else n - 1
        if window < 3:
            return x.copy()

        half = window // 2
        padded = np.pad(
            x,
            pad_width=(half, half),
            mode="edge"
        )
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            window_shape=window
        )

        return np.nanmedian(
            windows,
            axis=1
        )

    def __mad_sigma(self, x):
        """
        Robust sigma estimate from median absolute deviation.
        """

        x = np.asarray(
            x,
            dtype=np.float64
        )
        finite = np.isfinite(x)
        if np.sum(finite) < 5:
            return 0.0, 0.0

        center = np.nanmedian(
            x[finite]
        )
        mad = np.nanmedian(
            np.abs(x[finite] - center)
        )

        return center, 1.4826 * mad

    def __despike_series(self, time, value, window, sigma):
        """
        Replace isolated spikes by linear interpolation. This is intentionally
        mild and local; broad arc-scale signals are left to the low-frequency
        model below.
        """

        time = np.asarray(
            time,
            dtype=np.float64
        )
        value = np.asarray(
            value,
            dtype=np.float64
        )

        y = value.copy()
        if len(y) < 5:
            return y, np.zeros(
                len(y),
                dtype=bool
            )

        smooth = self.__rolling_median(
            x=y,
            window=window
        )
        residual = y - smooth
        center, robust_sigma = self.__mad_sigma(
            residual
        )
        if robust_sigma <= 0.0 or not np.isfinite(robust_sigma):
            return y, np.zeros(
                len(y),
                dtype=bool
            )

        outlier = (
            np.isfinite(residual)
            & (np.abs(residual - center) > float(sigma) * robust_sigma)
        )

        if not np.any(outlier):
            return y, outlier

        good = np.isfinite(y) & (~outlier)
        if np.sum(good) < 2:
            return y, outlier

        y[outlier] = np.interp(
            time[outlier],
            time[good],
            y[good]
        )

        return y, outlier

    def __remove_kbrr_low_frequency(self, time, residual):
        """
        Remove only the very-low-frequency empirical part of the KBRR residual:
        a low-order trend plus 1CPR/2CPR harmonics. Those components are below
        the LGD band, but imperfect finite-arc boundaries can leak their energy
        into 1-10 mHz during FFT processing.
        """

        time = np.asarray(
            time,
            dtype=np.float64
        )
        residual = np.asarray(
            residual,
            dtype=np.float64
        )

        if len(residual) < self.__min_segment_length:
            return residual.copy()

        t = time - np.nanmean(time)
        scale = max(
            np.nanmax(np.abs(t)),
            1.0
        )
        tau = t / scale

        columns = []
        trend_degree = max(
            0,
            int(self.__kbrr_trend_degree)
        )
        for degree in range(trend_degree + 1):
            columns.append(
                np.power(tau, degree)
            )

        period = float(self.__kbrr_cpr_period)
        harmonics = max(
            0,
            int(self.__kbrr_cpr_harmonics)
        )
        if period > 0.0:
            for harmonic in range(1, harmonics + 1):
                phase = 2.0 * np.pi * harmonic * t / period
                columns.append(
                    np.sin(phase)
                )
                columns.append(
                    np.cos(phase)
                )

        if len(columns) == 0:
            return residual.copy()

        design = np.vstack(columns).T
        finite = np.isfinite(residual) & np.all(
            np.isfinite(design),
            axis=1
        )
        if np.sum(finite) <= design.shape[1] + 5:
            return residual.copy()

        fit_mask = finite.copy()
        coeff = None
        for _ in range(3):
            if np.sum(fit_mask) <= design.shape[1]:
                break
            coeff = np.linalg.lstsq(
                design[fit_mask],
                residual[fit_mask],
                rcond=None
            )[0]
            model = design @ coeff
            fit_residual = residual - model
            center, robust_sigma = self.__mad_sigma(
                fit_residual[fit_mask]
            )
            if robust_sigma <= 0.0 or not np.isfinite(robust_sigma):
                break
            new_mask = finite & (
                np.abs(fit_residual - center) <= 4.0 * robust_sigma
            )
            if np.array_equal(
                    new_mask,
                    fit_mask):
                break
            fit_mask = new_mask

        if coeff is None:
            return residual.copy()

        low_frequency_model = design @ coeff
        return residual - low_frequency_model

    def __preprocess_kbrr_segment(self, time, residual):
        """
        Reduce KBRR-origin leakage before differentiation and FFT transfer.
        """

        y = np.asarray(
            residual,
            dtype=np.float64
        ).copy()

        if self.__kbrr_despike_enable:
            y, _ = self.__despike_series(
                time=time,
                value=y,
                window=self.__kbrr_despike_window,
                sigma=self.__kbrr_despike_sigma
            )

        if self.__kbrr_remove_low_frequency:
            y = self.__remove_kbrr_low_frequency(
                time=time,
                residual=y
            )

        return y

    def __segment_edge_keep_mask(self, n, dt):
        """
        Keep only the stable interior of a segment.

        Numerical differentiation uses one-sided formulas at the first/last
        samples, and the finite-arc FFT also treats segment endpoints as
        artificial boundaries. Trimming a short buffer from both ends prevents
        those samples from polluting the saved LGD product.
        """

        keep = np.ones(
            int(n),
            dtype=bool
        )

        if n <= 0 or self.__segment_edge_trim_seconds <= 0.0:
            return keep

        if dt <= 0.0 or not np.isfinite(dt):
            return keep

        trim_points = int(
            np.ceil(self.__segment_edge_trim_seconds / dt)
        )
        trim_points = max(
            2,
            trim_points
        )

        max_trim_points = max(
            0,
            (int(n) - int(self.__min_segment_length)) // 2
        )
        trim_points = min(
            trim_points,
            max_trim_points
        )

        if trim_points <= 0:
            return keep

        keep[:trim_points] = False
        keep[-trim_points:] = False

        return keep

    def __load_kbrr_residual(self):
        """
        Load postfit KBRR residual generated by AdjustSST.RangeRate.
        """

        res_dir = pathlib.Path(
            self.__PathfileConfig.RangeRateTemp
        ).joinpath(
            self.__date_span[0] + "_" + self.__date_span[1]
        )

        res_filename = res_dir.joinpath(
            "RangeRate_" + str(self.__arcNo) + ".hdf5"
        )

        if not res_filename.exists():
            raise FileNotFoundError(
                "RangeRate file does not exist: %s"
                % str(res_filename)
            )

        with h5py.File(res_filename, "r") as h5:
            payload = h5.attrs.get("sst_payload", "KBR")
            if isinstance(payload, bytes):
                payload = payload.decode("ascii", errors="ignore")
            self.__sst_payload = str(payload).upper()
        time, kbrr_residual = RangeRate.load_empirical_residual(
            res_filename, stage='Post'
        )

        time = np.asarray(
            time,
            dtype=np.float64
        ).reshape(-1)

        kbrr_residual = np.asarray(
            kbrr_residual,
            dtype=np.float64
        ).reshape(-1)

        if len(time) != len(kbrr_residual):
            raise ValueError(
                "Arc %d: time and KBRR residual length mismatch: %d vs %d"
                % (self.__arcNo, len(time), len(kbrr_residual))
            )

        return time, kbrr_residual

    def __split_by_gap(self, time):
        """
        Split KBRR time series by internal gaps.
        """

        if len(time) == 0:
            return []

        dt = np.diff(time)

        break_points = np.where(
            dt > self.__gap_threshold
        )[0] + 1

        starts = np.insert(
            break_points,
            0,
            0
        )

        ends = np.append(
            break_points,
            len(time)
        )

        segments = []

        for s, e in zip(starts, ends):
            if e > s:
                segments.append(
                    np.arange(s, e, dtype=np.int64)
                )

        return segments

    def __uniform_sample(self, time, value):
        """
        Match Code4LGDSlepian sampling handling.

        If sampling is non-uniform, interpolate to a uniform grid using the
        modal sample interval. Return a boolean mask for original epochs.
        """

        time = np.asarray(
            time,
            dtype=np.float64
        ).reshape(-1)

        value = np.asarray(
            value,
            dtype=np.float64
        ).reshape(-1)

        order = np.argsort(time)
        time = time[order]
        value = value[order]

        time, unique_idx = np.unique(
            time,
            return_index=True
        )
        value = value[unique_idx]

        dt = np.diff(time)

        if len(dt) == 0:
            raise ValueError(
                "Arc %d: cannot estimate dt from one epoch."
                % self.__arcNo
            )

        dt_mean = np.mean(dt)
        dt_std = np.std(dt)
        dt_rel_std = dt_std / dt_mean if dt_mean > 0 else np.inf
        dt_mode = self.__mode_step(dt)

        if dt_mode <= 0:
            raise ValueError(
                "Arc %d: invalid sampling interval."
                % self.__arcNo
            )

        if dt_rel_std <= 1.0e-6:
            return (
                time,
                value,
                np.ones(
                    len(time),
                    dtype=bool
                ),
                float(dt_mean)
            )

        time_uniform = np.arange(
            time[0],
            time[-1] + 0.5 * dt_mode,
            dt_mode,
            dtype=np.float64
        )

        if len(time) >= 4:
            interpolator = CubicSpline(
                time,
                value,
                bc_type="not-a-knot",
                extrapolate=False
            )
            value_uniform = interpolator(time_uniform)
            if np.any(~np.isfinite(value_uniform)):
                value_uniform = np.interp(
                    time_uniform,
                    time,
                    value
                )
        else:
            value_uniform = np.interp(
                time_uniform,
                time,
                value
            )

        tol = abs(dt_mode) * 1.0e-6
        nearest = np.searchsorted(
            time,
            time_uniform
        )
        left = np.clip(
            nearest - 1,
            0,
            len(time) - 1
        )
        right = np.clip(
            nearest,
            0,
            len(time) - 1
        )
        is_original = (
            (np.abs(time_uniform - time[left]) <= tol)
            |
            (np.abs(time_uniform - time[right]) <= tol)
        )

        return (
            time_uniform,
            np.asarray(
                value_uniform,
                dtype=np.float64
            ),
            is_original,
            float(dt_mode)
        )

    def __mode_step(self, dt):
        """
        MATLAB mode(diff(gpst)) equivalent for floating time tags.
        """

        dt = np.asarray(
            dt,
            dtype=np.float64
        )
        dt = dt[np.isfinite(dt)]

        if len(dt) == 0:
            return np.nan

        rounded = np.round(
            dt,
            decimals=9
        )
        values, counts = np.unique(
            rounded,
            return_counts=True
        )

        return float(
            values[np.argmax(counts)]
        )

    def __fill_nan_by_interp(self, x):
        """
        Fill isolated non-finite samples by linear interpolation.
        """

        x = np.asarray(
            x,
            dtype=np.float64
        ).copy()

        finite = np.isfinite(x)

        if np.all(finite):
            return x

        if np.sum(finite) < 2:
            return np.zeros_like(x)

        xp = np.arange(
            len(x),
            dtype=np.float64
        )

        x[~finite] = np.interp(
            xp[~finite],
            xp[finite],
            x[finite]
        )

        return x

    def __payload_default_f_high(self):
        """
        Han/Tang product convention: KBR-LGD keeps 1-10 mHz, while
        LRI-LGD keeps 1-20 mHz.
        """

        payload = str(self.__sst_payload).upper()
        if "LRI" in payload:
            return 20.0e-3
        return 10.0e-3

    def __resolve_frequency_band(self, dt):
        """
        Resolve the LGD signal band for the current payload and sample rate.
        """

        f_low = max(
            float(self.__f_low),
            self.__transfer_min_frequency
        )
        f_high = (
            self.__payload_default_f_high()
            if self.__f_high is None
            else float(self.__f_high)
        )

        nyq = 0.5 / dt
        if f_high >= nyq:
            f_high = nyq * 0.999

        if f_high <= f_low:
            raise ValueError(
                "Arc %d: invalid LGD band %.6e-%.6e Hz for dt %.3f s."
                % (self.__arcNo, f_low, f_high, dt)
            )

        self.__resolved_f_low = f_low
        self.__resolved_f_high = f_high

        return f_low, f_high

    def __bandpass_filter(self, x, dt, f_low=None, f_high=None):
        """
        Zero-phase Butterworth band-pass filter.
        """

        x = np.asarray(
            x,
            dtype=np.float64
        )

        fs = 1.0 / dt
        nyq = 0.5 * fs

        if f_low is None or f_high is None:
            f_low, f_high = self.__resolve_frequency_band(dt=dt)

        if f_high >= nyq:
            raise ValueError(
                "Arc %d: f_high %.6e Hz exceeds Nyquist %.6e Hz."
                % (self.__arcNo, f_high, nyq)
            )

        b, a = butter(
            N=self.__filter_order,
            Wn=[f_low, f_high],
            btype="bandpass",
            fs=fs,
        )

        x0 = self.__fill_nan_by_interp(x)

        padlen = min(
            len(x0) - 1,
            max(0, 3 * max(len(a), len(b)))
        )

        if padlen <= 0:
            return np.zeros_like(x0)

        y = filtfilt(
            b,
            a,
            x0,
            padlen=padlen
        )

        return y

    def __diff4(self, x, dt):
        """
        Fourth-order differentiation.

        Input:
            KBRR residual, delta rho_dot [m/s]

        Output:
            range acceleration residual, delta rho_ddot [m/s^2]
        """

        x = np.asarray(
            x,
            dtype=np.float64
        )

        n = len(x)

        y = np.zeros_like(
            x,
            dtype=np.float64
        )

        if n < 5:
            y[:] = np.gradient(x, dt)
            return y

        y[2:-2] = (
            -x[4:]
            + 8.0 * x[3:-1]
            - 8.0 * x[1:-3]
            + x[:-4]
        ) / (12.0 * dt)

        y[0] = (
            -25.0 * x[0]
            + 48.0 * x[1]
            - 36.0 * x[2]
            + 16.0 * x[3]
            - 3.0 * x[4]
        ) / (12.0 * dt)

        y[1] = (
            -3.0 * x[0]
            - 10.0 * x[1]
            + 18.0 * x[2]
            - 6.0 * x[3]
            + x[4]
        ) / (12.0 * dt)

        y[-1] = (
            25.0 * x[-1]
            - 48.0 * x[-2]
            + 36.0 * x[-3]
            - 16.0 * x[-4]
            + 3.0 * x[-5]
        ) / (12.0 * dt)

        y[-2] = (
            3.0 * x[-1]
            + 10.0 * x[-2]
            - 18.0 * x[-3]
            + 6.0 * x[-4]
            - x[-5]
        ) / (12.0 * dt)

        return y

    def __transfer_function(self, x, dt, f_low=None, f_high=None):
        """
        Apply the Ghobadi-Far transfer function only inside the selected
        frequency band.

        This follows the Han/Tang LGD processing logic: frequencies below
        1 mHz are not converted from L1B range-acceleration residuals, because
        the transfer relation is valid only above 1 mHz. For `lgd`, the band is
        1-10 mHz for KBR and 1-20 mHz for LRI. For `lgd_unfiltered`, f_high is
        None, so all frequencies >= 1 mHz up to Nyquist are retained.
        """

        x = np.asarray(
            x,
            dtype=np.float64
        )

        n = len(x)
        f_low = (
            self.__transfer_min_frequency
            if f_low is None
            else max(float(f_low), self.__transfer_min_frequency)
        )
        f_high = None if f_high is None else float(f_high)

        series_padded = np.concatenate(
            (
                x.reshape(-1),
                x.reshape(-1)[::-1]
            )
        )

        freq = np.fft.fftfreq(
            n=len(series_padded),
            d=dt
        )

        x_fft = np.fft.fft(series_padded)
        freq_abs = np.abs(freq)

        band = freq_abs >= f_low
        if f_high is not None:
            band = band & (freq_abs <= f_high)

        weight = np.zeros_like(
            freq_abs,
            dtype=np.float64
        )
        weight[band] = (
            1.0
            + self.__tf_factor
            * np.power(freq_abs[band], self.__tf_power)
        )

        lgd_padded = np.real(
            np.fft.ifft(
                x_fft * weight
            )
        )

        return lgd_padded[:n]

    def __compute_midpoint_lonlat(self, time):
        """
        Compute midpoint sub-satellite lon/lat.

        LGD is assigned to:

            r_mid = (r_A + r_B) / 2

        The ITRS positions and line-of-sight unit vector are also saved so
        later analyses can evaluate monthly reference fields strictly at the
        same satellite geometry.
        """

        gnv_a = self.__rd.getData(
            arc=self.__arcNo,
            kind=Payload.GNV,
            sat=SatID.A
        )

        gnv_b = self.__rd.getData(
            arc=self.__arcNo,
            kind=Payload.GNV,
            sat=SatID.B
        )

        t_a = gnv_a[:, 0].astype(np.float64)
        t_b = gnv_b[:, 0].astype(np.float64)

        r_a = gnv_a[:, 1:4].astype(np.float64)
        r_b = gnv_b[:, 1:4].astype(np.float64)

        r_a_interp = self.__interp_position(
            t_ref=t_a,
            r_ref=r_a,
            t_new=time
        )

        r_b_interp = self.__interp_position(
            t_ref=t_b,
            r_ref=r_b,
            t_new=time
        )

        r_mid_gcrs = 0.5 * (
            r_a_interp + r_b_interp
        )

        lon = np.zeros(
            len(time),
            dtype=np.float64
        )

        lat = np.zeros(
            len(time),
            dtype=np.float64
        )

        pos_a_itrs = np.zeros(
            (len(time), 3),
            dtype=np.float64
        )

        pos_b_itrs = np.zeros(
            (len(time), 3),
            dtype=np.float64
        )

        los_itrs = np.zeros(
            (len(time), 3),
            dtype=np.float64
        )

        for i in range(len(time)):
            self.__fr = self.__fr.setTime(
                float(time[i])
            )

            r_a_itrs = self.__fr.PosGCRS2ITRS(
                np.array(r_a_interp[i]).astype(float),
                self.__fr.getRotationMatrix
            )

            r_b_itrs = self.__fr.PosGCRS2ITRS(
                np.array(r_b_interp[i]).astype(float),
                self.__fr.getRotationMatrix
            )

            r_mid_itrs = 0.5 * (
                r_a_itrs + r_b_itrs
            )

            los = r_b_itrs - r_a_itrs
            los_norm = np.linalg.norm(los)

            pos_a_itrs[i, :] = r_a_itrs
            pos_b_itrs[i, :] = r_b_itrs

            if los_norm > 0:
                los_itrs[i, :] = los / los_norm
            else:
                los_itrs[i, :] = np.nan

            lon_i, lat_i, _ = GeoMathKit.CalcPolarAngles(
                r_mid_itrs
            )

            lon[i] = np.degrees(lon_i)
            lat[i] = np.degrees(lat_i)

            if lon[i] >= 180.0:
                lon[i] -= 360.0

        return lon, lat, pos_a_itrs, pos_b_itrs, los_itrs

    def __interp_position(self, t_ref, r_ref, t_new):
        """
        Linear interpolation of GNV positions to LGD epochs.
        """

        r_new = np.zeros(
            (len(t_new), 3),
            dtype=np.float64
        )

        for i in range(3):
            r_new[:, i] = np.interp(
                t_new,
                t_ref,
                r_ref[:, i]
            )

        return r_new

    def __save_result(self, result):
        """
        Save one-arc LGD product.
        """

        res_dir = pathlib.Path(
            self.__LGDTemp
        ).joinpath(
            self.__date_span[0] + "_" + self.__date_span[1]
        )

        if not res_dir.exists():
            os.makedirs(
                res_dir,
                exist_ok=True
            )

        res_filename = res_dir.joinpath(
            "LGD_" + str(self.__arcNo) + ".hdf5"
        )

        with h5py.File(res_filename, "w") as h5:
            h5.create_dataset(
                "time",
                data=result["time"]
            )

            h5.create_dataset(
                "lon",
                data=result["lon"]
            )

            h5.create_dataset(
                "lat",
                data=result["lat"]
            )

            h5.create_dataset(
                "altitude",
                data=result["altitude"]
            )

            h5.create_dataset(
                "pos_a_itrs",
                data=result["pos_a_itrs"]
            )

            h5.create_dataset(
                "pos_b_itrs",
                data=result["pos_b_itrs"]
            )

            h5.create_dataset(
                "los_itrs",
                data=result["los_itrs"]
            )

            h5.create_dataset(
                "lgd",
                data=result["lgd"]
            )

            h5.create_dataset(
                "lgd_unfiltered",
                data=result["lgd_unfiltered"]
            )

            h5.create_dataset(
                "range_acc",
                data=result["range_acc"]
            )

            h5.create_dataset(
                "kbrr_residual",
                data=result["kbrr_residual"]
            )

            h5.create_dataset(
                "kbrr_filtered",
                data=result["kbrr_filtered"]
            )

            h5.create_dataset(
                "is_original_sample",
                data=result["is_original_sample"]
            )

            h5.attrs["arcNo"] = self.__arcNo
            h5.attrs["date_start"] = self.__date_span[0]
            h5.attrs["date_end"] = self.__date_span[1]
            h5.attrs["input"] = "post_residual from RangeRate"
            h5.attrs["sst_payload"] = self.__sst_payload
            h5.attrs["LGD_position"] = "midpoint of GRACE-A and GRACE-B"
            h5.attrs["processing_order"] = (
                "range-rate residual -> uniform sampling -> finite difference "
                "range acceleration -> frequency-domain band selection -> "
                "admittance transfer inside selected band"
            )
            h5.attrs["filter"] = (
                "frequency-domain band-pass before transfer; diagnostic "
                "kbrr_filtered uses 6th-order zero-phase Butterworth"
            )
            h5.attrs["filter_order"] = self.__filter_order
            h5.attrs["kbrr_preprocess_enable"] = self.__kbrr_preprocess_enable
            h5.attrs["kbrr_despike_enable"] = self.__kbrr_despike_enable
            h5.attrs["kbrr_despike_window"] = self.__kbrr_despike_window
            h5.attrs["kbrr_despike_sigma"] = self.__kbrr_despike_sigma
            h5.attrs["kbrr_remove_low_frequency"] = self.__kbrr_remove_low_frequency
            h5.attrs["kbrr_trend_degree"] = self.__kbrr_trend_degree
            h5.attrs["kbrr_cpr_period"] = self.__kbrr_cpr_period
            h5.attrs["kbrr_cpr_harmonics"] = self.__kbrr_cpr_harmonics
            h5.attrs["segment_edge_trim_seconds"] = self.__segment_edge_trim_seconds
            h5.attrs["sampling"] = (
                "interpolate to uniform grid with modal dt when relative "
                "std(diff(time)) > 1e-6"
            )
            h5.attrs["f_low_Hz"] = (
                self.__resolved_f_low
                if self.__resolved_f_low is not None
                else self.__f_low
            )
            h5.attrs["f_high_Hz"] = (
                self.__resolved_f_high
                if self.__resolved_f_high is not None
                else np.nan
            )
            h5.attrs["transfer_min_frequency_Hz"] = self.__transfer_min_frequency
            h5.attrs["differentiation"] = "5-point 4th-order finite difference"
            h5.attrs["transfer_function"] = (
                "Z(f)=1+3.45e-4*f^(-1.04), applied only for f>=1 mHz "
                "inside the selected LGD band"
            )
            h5.attrs["lgd_unfiltered_definition"] = (
                "range-acceleration residual transformed for f>=1 mHz without "
                "an upper cutoff"
            )
            h5.attrs["unit_lgd"] = "m/s^2"
            h5.attrs["unit_lgd_nms2"] = "lgd * 1e9"

        print(
            "LGD arc %d saved to %s"
            % (self.__arcNo, str(res_filename))
        )
