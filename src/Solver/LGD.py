import os
import pathlib
import h5py
import numpy as np

from scipy.signal import butter, sosfiltfilt

from src.Interface.ArcOperation import ArcData
from src.Preference.EnumType import Payload, SatID
from src.Frame.EOP import EOP
from src.Frame.Frame import Frame
from src.Auxilary.GeoMathKit import GeoMathKit


class ComputeLGD:
    """
    Compute LGD from KBRR residual for a single arc.

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

        # Frequency band used in LGD studies
        self.__f_low = 0.9e-3
        self.__f_high = 11.0e-3

        # Butterworth order
        self.__filter_order = 3

        # Gap threshold
        self.__gap_threshold = 20.0

        # Minimum segment length for stable filtering and FFT
        self.__min_segment_length = 300

        # Ghobadi-Far transfer function
        self.__tf_factor = 3.45e-4
        self.__tf_power = -1.04

        # Output directory
        self.__LGDTemp = "../temp/LGD"

    def configure(
            self,
            AdjustOrbitConfig,
            InterfaceConfig,
            FrameConfig,
            f_low=0.9e-3,
            f_high=11.0e-3,
            filter_order=3,
            gap_threshold=20.0,
            min_segment_length=300):
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
        LGDTemp = self.__PathfileConfig.LGDTemp
        self.__f_low = f_low
        self.__f_high = f_high
        self.__filter_order = filter_order
        self.__gap_threshold = gap_threshold
        self.__min_segment_length = min_segment_length

        # Prefer LGDTemp from config if it exists.
        self.__LGDTemp = getattr(
            self.__PathfileConfig,
            "LGDTemp",
            LGDTemp
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

        kbrr_filtered = np.full_like(
            kbrr_residual,
            np.nan,
            dtype=np.float64
        )

        range_acc = np.full_like(
            kbrr_residual,
            np.nan,
            dtype=np.float64
        )

        lgd = np.full_like(
            kbrr_residual,
            np.nan,
            dtype=np.float64
        )

        segments = self.__split_by_gap(time)

        for index in segments:
            if len(index) < self.__min_segment_length:
                continue

            t_seg = time[index]
            rr_seg = kbrr_residual[index]

            dt = self.__estimate_dt(t_seg)

            rr_filtered_seg = self.__bandpass_filter(
                x=rr_seg,
                dt=dt
            )

            range_acc_seg = self.__diff4(
                x=rr_filtered_seg,
                dt=dt
            )

            lgd_seg = self.__transfer_function(
                x=range_acc_seg,
                dt=dt
            )

            kbrr_filtered[index] = rr_filtered_seg
            range_acc[index] = range_acc_seg
            lgd[index] = lgd_seg

        valid = np.isfinite(lgd)

        if np.sum(valid) == 0:
            print(
                "Arc %d skipped: no valid LGD samples."
                % self.__arcNo
            )
            return None

        time_valid = time[valid]

        lon, lat = self.__compute_midpoint_lonlat(
            time=time_valid
        )

        result = {
            "time": time_valid,
            "lon": lon,
            "lat": lat,
            "lgd": lgd[valid],
            "range_acc": range_acc[valid],
            "kbrr_residual": kbrr_residual[valid],
            "kbrr_filtered": kbrr_filtered[valid],
        }

        self.__save_result(result)

        return result

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
            time = h5["post_t"][:]
            kbrr_residual = h5["post_residual"][:]

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

    def __estimate_dt(self, time):
        """
        Estimate sampling interval.
        """

        dt = np.diff(time)

        if len(dt) == 0:
            raise ValueError(
                "Arc %d: cannot estimate dt from one epoch."
                % self.__arcNo
            )

        dt_median = np.median(dt)

        if dt_median <= 0:
            raise ValueError(
                "Arc %d: invalid sampling interval."
                % self.__arcNo
            )

        return float(dt_median)

    def __bandpass_filter(self, x, dt):
        """
        Zero-phase Butterworth band-pass filter.

        Default band:
            0.9 mHz - 11 mHz
        """

        x = np.asarray(
            x,
            dtype=np.float64
        )

        fs = 1.0 / dt
        nyq = 0.5 * fs

        if self.__f_high >= nyq:
            raise ValueError(
                "Arc %d: f_high %.6e Hz exceeds Nyquist %.6e Hz."
                % (self.__arcNo, self.__f_high, nyq)
            )

        sos = butter(
            N=self.__filter_order,
            Wn=[self.__f_low, self.__f_high],
            btype="bandpass",
            fs=fs,
            output="sos"
        )

        x0 = x - np.nanmean(x)

        padlen = min(
            len(x0) - 1,
            max(0, 3 * (2 * sos.shape[0] + 1))
        )

        if padlen <= 0:
            return np.zeros_like(x0)

        y = sosfiltfilt(
            sos,
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

    def __transfer_function(self, x, dt):
        """
        Apply Ghobadi-Far transfer function:

            Z(f) = 1 + 3.45e-4 * f^(-1.04)

        Input:
            range acceleration residual [m/s^2]

        Output:
            LGD [m/s^2]
        """

        x = np.asarray(
            x,
            dtype=np.float64
        )

        n = len(x)

        freq = np.fft.rfftfreq(
            n=n,
            d=dt
        )

        x_fft = np.fft.rfft(x)

        lgd_fft = np.zeros_like(
            x_fft,
            dtype=np.complex128
        )

        band = (
            (freq >= self.__f_low)
            &
            (freq <= self.__f_high)
        )

        Z = np.ones_like(
            freq,
            dtype=np.float64
        )

        Z[band] = (
            1.0
            + self.__tf_factor
            * np.power(freq[band], self.__tf_power)
        )

        lgd_fft[band] = x_fft[band] * Z[band]

        lgd = np.fft.irfft(
            lgd_fft,
            n=n
        )

        return lgd

    def __compute_midpoint_lonlat(self, time):
        """
        Compute midpoint sub-satellite lon/lat.

        LGD is assigned to:

            r_mid = (r_A + r_B) / 2
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

        for i in range(len(time)):
            self.__fr = self.__fr.setTime(
                float(time[i])
            )

            r_mid_itrs = self.__fr.PosGCRS2ITRS(
                np.array(r_mid_gcrs[i]).astype(float),
                self.__fr.getRotationMatrix
            )

            lon_i, lat_i, _ = GeoMathKit.CalcPolarAngles(
                r_mid_itrs
            )

            lon[i] = np.degrees(lon_i)
            lat[i] = np.degrees(lat_i)

            if lon[i] >= 180.0:
                lon[i] -= 360.0

        return lon, lat

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
                "lgd",
                data=result["lgd"]
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

            h5.attrs["arcNo"] = self.__arcNo
            h5.attrs["date_start"] = self.__date_span[0]
            h5.attrs["date_end"] = self.__date_span[1]
            h5.attrs["input"] = "post_residual from RangeRate"
            h5.attrs["LGD_position"] = "midpoint of GRACE-A and GRACE-B"
            h5.attrs["filter"] = "zero-phase Butterworth band-pass"
            h5.attrs["filter_order"] = self.__filter_order
            h5.attrs["f_low_Hz"] = self.__f_low
            h5.attrs["f_high_Hz"] = self.__f_high
            h5.attrs["differentiation"] = "4th-order finite difference"
            h5.attrs["transfer_function"] = "Z(f)=1+3.45e-4*f^(-1.04)"
            h5.attrs["unit_lgd"] = "m/s^2"
            h5.attrs["unit_lgd_nms2"] = "lgd * 1e9"

        print(
            "LGD arc %d saved to %s"
            % (self.__arcNo, str(res_filename))
        )