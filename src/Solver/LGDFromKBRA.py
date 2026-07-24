import os
import pathlib

import h5py
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

from src.Auxilary.GeoMathKit import GeoMathKit
from src.Frame.EOP import EOP
from src.Frame.Frame import Frame
from src.Interface.ArcOperation import ArcData
from src.Preference.EnumType import Payload, SatID
from src.Auxilary.StandardLGD import StandardLGD


class ComputeLGDFromKBRA:
    """
    Compute LGD from range-acceleration residuals.

    This module implements the second LGD route used in along-track LGD
    processing:

        official KBRA - dynamic reference KBRA -> KBRA residual -> LGD

    It is intentionally independent from ComputeLGD, whose default route is:

        KBRR residual -> numerical derivative -> KBRA residual -> LGD

    The default CalibrateOrbit workflow does not import or call this class.
    """

    def __init__(self, arcNo=None, date_span=None):
        self.__arcNo = arcNo
        self.__date_span = date_span

        self.__AdjustOrbitConfig = None
        self.__InterfaceConfig = None
        self.__FrameConfig = None

        self.__PathfileConfig = None
        self.__rd = None
        self.__fr = None

        self.__LGDTemp = "../temp/LGD"
        self.__StandardLGDTemp = "../temp/StandardLGD"
        self.__sst_payload = "KBR"

        # Han rr2lgd.f90 defaults.
        self.__fc = 0.01
        self.__f_low_outlier = 0.005
        self.__f_high = None
        self.__threshold_kbra = None
        self.__min_segment_duration = 10800.0
        self.__gap_tolerance = 0.1
        self.__filter_order = 6
        self.__transfer_min_frequency = 1.0e-3
        self.__tf_factor = 3.45e-4
        self.__tf_power = -1.04
        self.__allow_velocity_fallback = False

    def configure(
            self,
            AdjustOrbitConfig=None,
            InterfaceConfig=None,
            FrameConfig=None,
            fc=0.01,
            payload="KBR",
            f_high=None,
            threshold_kbra=None,
            min_segment_duration=10800.0,
            gap_tolerance=0.1,
            filter_order=6,
            allow_velocity_fallback=False):
        """
        Configure optional PyHawk context and Han-style processing constants.

        Parameters
        ----------
        fc : float
            Crossover frequency for model/observation combination. Frequencies
            below fc can be taken from model_lgd_tvg, while frequencies above
            fc come from observed KBRA residuals.
        payload : {"KBR", "LRI"}
            Controls Han defaults: KBR uses f_high=0.1 Hz and threshold=2e-7;
            LRI uses f_high=0.25 Hz and threshold=2e-8.
        """

        self.__AdjustOrbitConfig = AdjustOrbitConfig
        self.__InterfaceConfig = InterfaceConfig
        self.__FrameConfig = FrameConfig
        self.__fc = float(fc)
        self.__sst_payload = str(payload).upper()
        self.__min_segment_duration = float(min_segment_duration)
        self.__gap_tolerance = float(gap_tolerance)
        self.__filter_order = int(filter_order)
        self.__allow_velocity_fallback = bool(allow_velocity_fallback)

        if self.__sst_payload == "LRI":
            self.__f_high = 0.25 if f_high is None else float(f_high)
            self.__threshold_kbra = (
                2.0e-8 if threshold_kbra is None else float(threshold_kbra)
            )
        else:
            self.__f_high = 0.1 if f_high is None else float(f_high)
            self.__threshold_kbra = (
                2.0e-7 if threshold_kbra is None else float(threshold_kbra)
            )

        if self.__AdjustOrbitConfig is not None:
            self.__PathfileConfig = self.__AdjustOrbitConfig.PathOfFiles()
            self.__PathfileConfig.__dict__.update(
                self.__AdjustOrbitConfig.PathOfFilesConfig.copy()
            )
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

        if self.__InterfaceConfig is not None:
            self.__rd = ArcData(interfaceConfig=self.__InterfaceConfig)

        if self.__FrameConfig is not None:
            eop = EOP().configure(frameConfig=self.__FrameConfig).load()
            self.__fr = Frame(eop).configure(frameConfig=self.__FrameConfig)

        return self

    def compute_from_arc(
            self,
            reference_state_file=None,
            output_name=None,
            model_lgd_static=None,
            model_lgd_tvg=None):
        """
        Compute one arc by reading official KBRA from ArcData and deriving
        reference KBRA from the dynamic state-vector file.

        The state-vector file defaults to:

            {StateVectorDataTemp}/{date_start}_{date_end}/{arcNo}_AB.hdf5

        It must contain datasets `t`, `r`, and `v`, as produced by the orbit
        adjustment pipeline.
        """

        if self.__rd is None:
            raise RuntimeError("InterfaceConfig is required for compute_from_arc.")
        if self.__PathfileConfig is None:
            raise RuntimeError("AdjustOrbitConfig is required for compute_from_arc.")
        if self.__arcNo is None or self.__date_span is None:
            raise RuntimeError("arcNo and date_span are required for compute_from_arc.")

        kbr = self.__rd.getData(
            arc=self.__arcNo,
            kind=Payload.KBR,
            sat=SatID.A
        )
        if kbr is None or len(kbr) == 0:
            print("Arc %d skipped: no KBR/KBRA observations." % self.__arcNo)
            return None

        time_obs = np.asarray(kbr[:, 0], dtype=np.float64)
        observed_kbra = np.asarray(kbr[:, 3], dtype=np.float64)

        if reference_state_file is None:
            reference_state_file = pathlib.Path(
                self.__PathfileConfig.StateVectorDataTemp
            ).joinpath(
                self.__date_span[0] + "_" + self.__date_span[1],
                str(self.__arcNo) + "_AB.hdf5"
            )
        else:
            reference_state_file = pathlib.Path(reference_state_file)

        ref_time, reference_kbra = self.load_reference_kbra_from_state(
            reference_state_file,
            allow_velocity_fallback=self.__allow_velocity_fallback
        )

        reference_interp = self.__interpolate_series(
            t_ref=ref_time,
            value_ref=reference_kbra,
            t_new=time_obs
        )

        result = self.compute_from_kbra(
            time=time_obs,
            observed_kbra=observed_kbra,
            reference_kbra=reference_interp,
            model_lgd_static=model_lgd_static,
            model_lgd_tvg=model_lgd_tvg,
            range_residual=None,
        )

        if result is None:
            return None

        if self.__fr is not None:
            lon, lat, pos_a, pos_b, los = self.__compute_midpoint_lonlat(
                result["time"]
            )
            result["lon"] = lon
            result["lat"] = lat
            result["pos_a_itrs"] = pos_a
            result["pos_b_itrs"] = pos_b
            result["los_itrs"] = los
            result["altitude"] = StandardLGD.midpoint_altitude(
                pos_a_itrs=pos_a,
                pos_b_itrs=pos_b
            )

        self.save_result(result, output_name=output_name)
        if all(key in result for key in (
                "lon", "lat", "altitude", "lgd_unfiltered", "lgd")):
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

    def compute_from_hdf5(
            self,
            input_file,
            output_file=None,
            time_key="time",
            observed_key="observed_kbra",
            reference_key="reference_kbra",
            model_static_key=None,
            model_tvg_key=None,
            range_residual_key=None):
        """
        Compute LGD from an external HDF5 file containing observed/reference
        KBRA arrays. This is useful when the dynamic reference KBRA is generated
        outside PyHawk.
        """

        input_file = pathlib.Path(input_file)
        with h5py.File(input_file, "r") as h5:
            time = h5[time_key][:]
            observed_kbra = h5[observed_key][:]
            reference_kbra = h5[reference_key][:]
            model_static = (
                h5[model_static_key][:]
                if model_static_key is not None and model_static_key in h5
                else None
            )
            model_tvg = (
                h5[model_tvg_key][:]
                if model_tvg_key is not None and model_tvg_key in h5
                else None
            )
            range_residual = (
                h5[range_residual_key][:]
                if range_residual_key is not None and range_residual_key in h5
                else None
            )

        result = self.compute_from_kbra(
            time=time,
            observed_kbra=observed_kbra,
            reference_kbra=reference_kbra,
            model_lgd_static=model_static,
            model_lgd_tvg=model_tvg,
            range_residual=range_residual,
        )

        if output_file is not None and result is not None:
            self.save_result(result, filename=output_file)

        return result

    def compute_from_kbra(
            self,
            time,
            observed_kbra,
            reference_kbra,
            model_lgd_static=None,
            model_lgd_tvg=None,
            range_residual=None):
        """
        Core calculation from observed and reference range acceleration.

        All acceleration/LGD quantities are in m/s^2.
        """

        time = np.asarray(time, dtype=np.float64).reshape(-1)
        observed_kbra = np.asarray(observed_kbra, dtype=np.float64).reshape(-1)
        reference_kbra = np.asarray(reference_kbra, dtype=np.float64).reshape(-1)

        self.__require_same_length(
            time,
            observed_kbra,
            reference_kbra,
            names=("time", "observed_kbra", "reference_kbra")
        )

        model_lgd_static = self.__optional_array(
            model_lgd_static,
            len(time),
            "model_lgd_static"
        )
        model_lgd_tvg = self.__optional_array(
            model_lgd_tvg,
            len(time),
            "model_lgd_tvg"
        )
        range_residual = self.__optional_array(
            range_residual,
            len(time),
            "range_residual"
        )

        order = np.argsort(time)
        time = time[order]
        observed_kbra = observed_kbra[order]
        reference_kbra = reference_kbra[order]
        if model_lgd_static is not None:
            model_lgd_static = model_lgd_static[order]
        if model_lgd_tvg is not None:
            model_lgd_tvg = model_lgd_tvg[order]
        if range_residual is not None:
            range_residual = range_residual[order]

        valid = (
            np.isfinite(time)
            &
            np.isfinite(observed_kbra)
            &
            np.isfinite(reference_kbra)
        )
        if model_lgd_static is not None:
            valid &= np.isfinite(model_lgd_static)
        if model_lgd_tvg is not None:
            valid &= np.isfinite(model_lgd_tvg)
        if range_residual is not None:
            valid &= np.isfinite(range_residual)

        time = time[valid]
        observed_kbra = observed_kbra[valid]
        reference_kbra = reference_kbra[valid]
        if model_lgd_static is not None:
            model_lgd_static = model_lgd_static[valid]
        if model_lgd_tvg is not None:
            model_lgd_tvg = model_lgd_tvg[valid]
        if range_residual is not None:
            range_residual = range_residual[valid]

        kbra_residual = observed_kbra - reference_kbra

        if range_residual is not None and self.__sst_payload == "LRI":
            jump_mask = self.__lri_phase_jump_mask(
                range_residual=range_residual
            )
            time = time[jump_mask]
            observed_kbra = observed_kbra[jump_mask]
            reference_kbra = reference_kbra[jump_mask]
            kbra_residual = kbra_residual[jump_mask]
            if model_lgd_static is not None:
                model_lgd_static = model_lgd_static[jump_mask]
            if model_lgd_tvg is not None:
                model_lgd_tvg = model_lgd_tvg[jump_mask]

        if len(time) < 3:
            return None

        time_all = []
        kbra_all = []
        lgd_raw_all = []
        lgd_final_all = []
        lgd_high_all = []
        lgd_low_all = []
        idout_all = []
        observed_all = []
        reference_all = []

        segments = self.__split_by_gap(time)
        for index in segments:
            if len(index) < 3:
                continue

            t_seg = time[index]
            if t_seg[-1] - t_seg[0] <= self.__min_segment_duration:
                continue

            dt = self.__mode_step(np.diff(t_seg))
            if not np.isfinite(dt) or dt <= 0.0:
                continue

            kbra_seg = kbra_residual[index]
            observed_seg = observed_kbra[index]
            reference_seg = reference_kbra[index]

            t_uniform, kbra_uniform = self.__uniform_sample(t_seg, kbra_seg)
            _, observed_uniform = self.__uniform_sample(t_seg, observed_seg)
            _, reference_uniform = self.__uniform_sample(t_seg, reference_seg)

            lgd_raw = self.__transfer_function(kbra_uniform, dt)

            static_uniform = None
            tvg_uniform = None
            if model_lgd_static is not None:
                _, static_uniform = self.__uniform_sample(
                    t_seg,
                    model_lgd_static[index]
                )
            if model_lgd_tvg is not None:
                _, tvg_uniform = self.__uniform_sample(
                    t_seg,
                    model_lgd_tvg[index]
                )

            lgd_final, lgd_high, lgd_low, idout = self.__han_reconstruct(
                lgd_raw=lgd_raw,
                model_lgd_static=static_uniform,
                model_lgd_tvg=tvg_uniform,
                dt=dt,
            )

            time_all.append(t_uniform)
            kbra_all.append(kbra_uniform)
            lgd_raw_all.append(lgd_raw)
            lgd_final_all.append(lgd_final)
            lgd_high_all.append(lgd_high)
            lgd_low_all.append(lgd_low)
            idout_all.append(idout)
            observed_all.append(observed_uniform)
            reference_all.append(reference_uniform)

        if len(lgd_final_all) == 0:
            if self.__arcNo is not None:
                print("Arc %d skipped: no KBRA segment longer than 3 hours." % self.__arcNo)
            return None

        result = {
            "time": np.concatenate(time_all),
            "lgd": np.concatenate(lgd_final_all),
            "lgd_unfiltered": np.concatenate(lgd_raw_all),
            "lgd_high": np.concatenate(lgd_high_all),
            "lgd_low": np.concatenate(lgd_low_all),
            "idout": np.concatenate(idout_all),
            "range_acc": np.concatenate(kbra_all),
            "kbra_residual": np.concatenate(kbra_all),
            "observed_kbra": np.concatenate(observed_all),
            "reference_kbra": np.concatenate(reference_all),
        }

        finite = (
            np.isfinite(result["time"])
            &
            np.isfinite(result["lgd"])
            &
            np.isfinite(result["kbra_residual"])
        )
        for key in list(result.keys()):
            result[key] = result[key][finite]

        return result

    @staticmethod
    def load_reference_kbra_from_state(filename, allow_velocity_fallback=False):
        """
        Compute dynamic reference range acceleration from saved state vectors.

        The file must contain:

            t : (N,)
            r : (N, 6[, 1])  A xyz followed by B xyz
            v : (N, 6[, 1])  A xyz followed by B xyz

        If datasets `accA` and `accB` exist, they are used as the total
        force-model accelerations generated during orbit integration. This is
        the strict KBRA expression from Kim: rho_ddot is computed from relative
        position, velocity, and acceleration. Older state files without saved
        accelerations only fall back to differentiating integrated velocities
        when `allow_velocity_fallback=True`.
        """

        filename = pathlib.Path(filename)
        if not filename.exists():
            raise FileNotFoundError("State-vector file not found: %s" % str(filename))

        with h5py.File(filename, "r") as h5:
            t = np.asarray(h5["t"][:], dtype=np.float64).reshape(-1)
            r = np.asarray(h5["r"][:], dtype=np.float64)
            v = np.asarray(h5["v"][:], dtype=np.float64)
            acc_a = ComputeLGDFromKBRA.__read_saved_acceleration(h5, "accA", t)
            acc_b = ComputeLGDFromKBRA.__read_saved_acceleration(h5, "accB", t)

        r = np.squeeze(r)
        v = np.squeeze(v)
        if r.ndim != 2 or v.ndim != 2 or r.shape[1] < 6 or v.shape[1] < 6:
            raise ValueError(
                "State-vector datasets must have shape (N, 6[, 1]); "
                "got r=%s, v=%s" % (r.shape, v.shape)
            )

        pos_a = r[:, 0:3]
        pos_b = r[:, 3:6]
        vel_a = v[:, 0:3]
        vel_b = v[:, 3:6]

        if acc_a is None or acc_b is None:
            if not allow_velocity_fallback:
                raise ValueError(
                    "ReferenceKBRA LGD requires accA/accB in state-vector file: %s" % str(filename)
                )
            acc_a = ComputeLGDFromKBRA.__gradient_vector(vel_a, t)
            acc_b = ComputeLGDFromKBRA.__gradient_vector(vel_b, t)

        reference_kbra = ComputeLGDFromKBRA.range_acceleration_from_state(
            pos_a=pos_a,
            vel_a=vel_a,
            acc_a=acc_a,
            pos_b=pos_b,
            vel_b=vel_b,
            acc_b=acc_b,
        )

        return t, reference_kbra

    @staticmethod
    def __read_saved_acceleration(h5, key, time):
        if key not in h5:
            return None

        acc = np.asarray(h5[key][:], dtype=np.float64)
        acc = np.squeeze(acc)
        if acc.ndim == 1:
            if acc.size % 3 != 0:
                return None
            acc = acc.reshape((-1, 3))
        if acc.ndim != 2 or acc.shape[1] != 3:
            return None

        time = np.asarray(time, dtype=np.float64).reshape(-1)
        if len(acc) == len(time):
            return acc

        if "acc_time" not in h5:
            return None

        acc_time = np.asarray(h5["acc_time"][:], dtype=np.float64).reshape(-1)
        return ComputeLGDFromKBRA.__interpolate_vector(acc_time, acc, time)

    @staticmethod
    def __interpolate_vector(t_ref, value_ref, t_new):
        t_ref = np.asarray(t_ref, dtype=np.float64).reshape(-1)
        value_ref = np.asarray(value_ref, dtype=np.float64)
        t_new = np.asarray(t_new, dtype=np.float64).reshape(-1)

        if len(t_ref) == 0 or value_ref.size == 0:
            return None

        value_ref = np.squeeze(value_ref)
        if value_ref.ndim == 1:
            if value_ref.size % 3 != 0:
                return None
            value_ref = value_ref.reshape((-1, 3))
        if value_ref.ndim != 2 or value_ref.shape[1] != 3:
            return None

        count = min(len(t_ref), len(value_ref))
        t_ref = t_ref[:count]
        value_ref = value_ref[:count]

        finite = np.isfinite(t_ref) & np.all(np.isfinite(value_ref), axis=1)
        t_ref = t_ref[finite]
        value_ref = value_ref[finite]
        if len(t_ref) == 0:
            return None

        order = np.argsort(t_ref)
        t_ref = t_ref[order]
        value_ref = value_ref[order]

        rounded_time = np.round(t_ref, 9)
        _, last_reversed = np.unique(rounded_time[::-1], return_index=True)
        keep = len(rounded_time) - 1 - last_reversed
        keep = np.sort(keep)
        t_ref = t_ref[keep]
        value_ref = value_ref[keep]
        if len(t_ref) == 1:
            return np.repeat(value_ref[:1], len(t_new), axis=0)

        return np.column_stack([
            np.interp(t_new, t_ref, value_ref[:, i])
            for i in range(3)
        ])

    @staticmethod
    def range_acceleration_from_state(pos_a, vel_a, acc_a, pos_b, vel_b, acc_b):
        """
        Project relative position, velocity, and acceleration to rho_ddot.
        """

        pos_a = np.asarray(pos_a, dtype=np.float64)
        vel_a = np.asarray(vel_a, dtype=np.float64)
        acc_a = np.asarray(acc_a, dtype=np.float64)
        pos_b = np.asarray(pos_b, dtype=np.float64)
        vel_b = np.asarray(vel_b, dtype=np.float64)
        acc_b = np.asarray(acc_b, dtype=np.float64)

        dr = pos_a - pos_b
        dv = vel_a - vel_b
        da = acc_a - acc_b

        rho = np.linalg.norm(dr, axis=1)
        rho_dot = np.sum(dr * dv, axis=1) / rho
        rho_ddot = (
            np.sum(dv * dv, axis=1)
            +
            np.sum(dr * da, axis=1)
            -
            rho_dot * rho_dot
        ) / rho

        return rho_ddot

    def save_result(self, result, filename=None, output_name=None):
        """
        Save LGD product without overwriting the default LGD_{arc}.hdf5 route.
        """

        if filename is None:
            if output_name is None:
                if self.__arcNo is None:
                    output_name = "LGD_KBRA.hdf5"
                else:
                    output_name = "LGD_KBRA_" + str(self.__arcNo) + ".hdf5"

            if self.__date_span is not None:
                res_dir = pathlib.Path(self.__LGDTemp).joinpath(
                    self.__date_span[0] + "_" + self.__date_span[1]
                )
            else:
                res_dir = pathlib.Path(self.__LGDTemp)

            if not res_dir.exists():
                os.makedirs(res_dir, exist_ok=True)
            filename = res_dir.joinpath(output_name)
        else:
            filename = pathlib.Path(filename)
            if filename.parent and not filename.parent.exists():
                os.makedirs(filename.parent, exist_ok=True)

        with h5py.File(filename, "w") as h5:
            for key, value in result.items():
                h5.create_dataset(key, data=value)

            h5.attrs["arcNo"] = -1 if self.__arcNo is None else self.__arcNo
            if self.__date_span is not None:
                h5.attrs["date_start"] = self.__date_span[0]
                h5.attrs["date_end"] = self.__date_span[1]
            h5.attrs["input"] = "official KBRA minus dynamic reference KBRA"
            h5.attrs["sst_payload"] = self.__sst_payload
            h5.attrs["processing_order"] = (
                "observed range acceleration - dynamic reference range "
                "acceleration -> admittance transfer -> optional Han "
                "high-frequency observation / low-frequency model combination"
            )
            h5.attrs["filter"] = "6th-order zero-phase Butterworth"
            h5.attrs["filter_order"] = self.__filter_order
            h5.attrs["fc_Hz"] = self.__fc
            h5.attrs["f_outlier_low_Hz"] = self.__f_low_outlier
            h5.attrs["f_high_Hz"] = self.__f_high
            h5.attrs["threshold_kbra_mps2"] = self.__threshold_kbra
            h5.attrs["transfer_function"] = (
                "Z(f)=1+3.45e-4*f^(-1.04), applied for f>1 mHz"
            )
            h5.attrs["unit_lgd"] = "m/s^2"
            h5.attrs["unit_lgd_nms2"] = "lgd * 1e9"

        print("KBRA-route LGD saved to %s" % str(filename))

    def __han_reconstruct(
            self,
            lgd_raw,
            model_lgd_static,
            model_lgd_tvg,
            dt):
        """
        Han rr2lgd.f90 style reconstruction.

        If no TVG model is provided, return a band-passed observed LGD. If TVG
        is provided, output high-frequency observed LGD plus low-frequency TVG.
        """

        n = len(lgd_raw)
        idout = np.zeros(n, dtype=np.int8)
        nfc = max(1, int(round(1.0 / self.__fc / dt)))

        if model_lgd_static is not None and model_lgd_tvg is not None:
            diagnostic = lgd_raw - model_lgd_static - model_lgd_tvg
        elif model_lgd_tvg is not None:
            diagnostic = lgd_raw - model_lgd_tvg
        elif model_lgd_static is not None:
            diagnostic = lgd_raw - model_lgd_static
        else:
            diagnostic = lgd_raw

        diagnostic_filtered = self.__butterworth_filter(
            diagnostic,
            dt=dt,
            f_low=self.__f_low_outlier,
            f_high=self.__f_high
        )

        if n > 2 * nfc:
            middle = np.abs(diagnostic_filtered[nfc:-nfc]) < self.__threshold_kbra
            idout[nfc:-nfc] = middle.astype(np.int8)

        lgd_high = self.__butterworth_filter(
            lgd_raw,
            dt=dt,
            f_low=self.__fc,
            f_high=self.__f_high
        )

        if model_lgd_tvg is not None:
            tvg_high = self.__butterworth_filter(
                model_lgd_tvg,
                dt=dt,
                f_low=self.__fc,
                f_high=self.__f_high
            )
            lgd_low = model_lgd_tvg - tvg_high
        else:
            lgd_low = np.zeros_like(lgd_high)

        return lgd_low + lgd_high, lgd_high, lgd_low, idout

    def __transfer_function(self, x, dt):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        freq = np.fft.rfftfreq(len(x), d=dt)
        spectrum = np.fft.rfft(x)
        weight = np.ones_like(freq)
        index = freq > self.__transfer_min_frequency
        weight[index] = (
            1.0
            +
            self.__tf_factor
            *
            np.power(freq[index], self.__tf_power)
        )
        return np.fft.irfft(spectrum * weight, n=len(x))

    def __butterworth_filter(self, x, dt, f_low, f_high):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        fs = 1.0 / dt
        nyq = 0.5 * fs
        f_high = min(float(f_high), nyq * 0.999)
        if f_low <= 0.0 or f_high <= f_low:
            return np.zeros_like(x)

        b, a = butter(
            N=self.__filter_order,
            Wn=[f_low, f_high],
            btype="bandpass",
            fs=fs
        )

        x0 = self.__fill_nan_by_interp(x)
        padlen = min(len(x0) - 1, max(0, 3 * max(len(a), len(b))))
        if padlen <= 0:
            return np.zeros_like(x0)

        return filtfilt(b, a, x0, padlen=padlen)

    def __split_by_gap(self, time):
        dt = np.diff(time)
        if len(dt) == 0:
            return []
        dt_mode = self.__mode_step(dt)
        breaks = np.where(np.abs(dt - dt_mode) > self.__gap_tolerance)[0] + 1
        starts = np.insert(breaks, 0, 0)
        ends = np.append(breaks, len(time))
        return [
            np.arange(s, e, dtype=np.int64)
            for s, e in zip(starts, ends)
            if e > s
        ]

    def __uniform_sample(self, time, value):
        time = np.asarray(time, dtype=np.float64).reshape(-1)
        value = np.asarray(value, dtype=np.float64).reshape(-1)

        time, unique_idx = np.unique(time, return_index=True)
        value = value[unique_idx]

        dt_mode = self.__mode_step(np.diff(time))
        time_uniform = np.arange(
            time[0],
            time[-1] + 0.5 * dt_mode,
            dt_mode,
            dtype=np.float64
        )

        if len(time) == len(time_uniform) and np.allclose(
                time,
                time_uniform,
                atol=max(1.0e-9, dt_mode * 1.0e-9),
                rtol=0.0):
            return time, value

        if len(time) >= 4:
            interpolator = CubicSpline(
                time,
                value,
                bc_type="not-a-knot",
                extrapolate=False
            )
            value_uniform = interpolator(time_uniform)
            if np.any(~np.isfinite(value_uniform)):
                value_uniform = np.interp(time_uniform, time, value)
        else:
            value_uniform = np.interp(time_uniform, time, value)

        return time_uniform, np.asarray(value_uniform, dtype=np.float64)

    @staticmethod
    def __mode_step(dt):
        dt = np.asarray(dt, dtype=np.float64)
        dt = dt[np.isfinite(dt)]
        if len(dt) == 0:
            return np.nan
        rounded = np.round(dt, decimals=9)
        values, counts = np.unique(rounded, return_counts=True)
        return float(values[np.argmax(counts)])

    @staticmethod
    def __fill_nan_by_interp(x):
        x = np.asarray(x, dtype=np.float64).copy()
        finite = np.isfinite(x)
        if np.all(finite):
            return x
        if np.sum(finite) < 2:
            return np.zeros_like(x)
        xp = np.arange(len(x), dtype=np.float64)
        x[~finite] = np.interp(xp[~finite], xp[finite], x[finite])
        return x

    @staticmethod
    def __gradient_vector(value, time):
        value = np.asarray(value, dtype=np.float64)
        time = np.asarray(time, dtype=np.float64).reshape(-1)
        out = np.zeros_like(value)
        for col in range(value.shape[1]):
            out[:, col] = np.gradient(value[:, col], time)
        return out

    @staticmethod
    def __interpolate_series(t_ref, value_ref, t_new):
        t_ref = np.asarray(t_ref, dtype=np.float64).reshape(-1)
        value_ref = np.asarray(value_ref, dtype=np.float64).reshape(-1)
        t_new = np.asarray(t_new, dtype=np.float64).reshape(-1)
        order = np.argsort(t_ref)
        t_ref = t_ref[order]
        value_ref = value_ref[order]
        return np.interp(t_new, t_ref, value_ref)

    @staticmethod
    def __require_same_length(*arrays, names):
        lengths = [len(array) for array in arrays]
        if len(set(lengths)) != 1:
            raise ValueError(
                "Input length mismatch: "
                + ", ".join("%s=%d" % (name, length) for name, length in zip(names, lengths))
            )

    @staticmethod
    def __optional_array(value, length, name):
        if value is None:
            return None
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        if len(value) != length:
            raise ValueError(
                "%s length mismatch: expected %d, got %d"
                % (name, length, len(value))
            )
        return value

    @staticmethod
    def __lri_phase_jump_mask(range_residual):
        range_residual = np.asarray(range_residual, dtype=np.float64).reshape(-1)
        mask = np.ones(len(range_residual), dtype=bool)
        jumps = np.where(np.abs(np.diff(range_residual)) > 1.0e-3)[0] + 1
        for jump in jumps:
            start = max(0, jump - 25)
            end = min(len(mask), jump + 26)
            mask[start:end] = False
        return mask

    def __compute_midpoint_lonlat(self, time):
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

        r_a_interp = np.column_stack([
            np.interp(time, t_a, r_a[:, col])
            for col in range(3)
        ])
        r_b_interp = np.column_stack([
            np.interp(time, t_b, r_b[:, col])
            for col in range(3)
        ])

        lon = np.zeros(len(time), dtype=np.float64)
        lat = np.zeros(len(time), dtype=np.float64)
        pos_a_itrs = np.zeros((len(time), 3), dtype=np.float64)
        pos_b_itrs = np.zeros((len(time), 3), dtype=np.float64)
        los_itrs = np.zeros((len(time), 3), dtype=np.float64)

        for i in range(len(time)):
            self.__fr = self.__fr.setTime(float(time[i]))
            r_a_itrs = self.__fr.PosGCRS2ITRS(
                np.array(r_a_interp[i]).astype(float),
                self.__fr.getRotationMatrix
            )
            r_b_itrs = self.__fr.PosGCRS2ITRS(
                np.array(r_b_interp[i]).astype(float),
                self.__fr.getRotationMatrix
            )
            r_mid_itrs = 0.5 * (r_a_itrs + r_b_itrs)
            los = r_b_itrs - r_a_itrs
            los_norm = np.linalg.norm(los)

            pos_a_itrs[i, :] = r_a_itrs
            pos_b_itrs[i, :] = r_b_itrs
            los_itrs[i, :] = los / los_norm if los_norm > 0.0 else np.nan

            lon_i, lat_i, _ = GeoMathKit.CalcPolarAngles(r_mid_itrs)
            lon[i] = np.degrees(lon_i)
            lat[i] = np.degrees(lat_i)
            if lon[i] >= 180.0:
                lon[i] -= 360.0

        return lon, lat, pos_a_itrs, pos_b_itrs, los_itrs
