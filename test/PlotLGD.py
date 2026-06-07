# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2026/6/5 21:33
# @File    : PlotLGD.py
# @Software: PyCharm
import os
import h5py
import pathlib
import numpy as np
import matplotlib.pyplot as plt


class PlotLGDMap:
    """
    Plot spatial map of LGD using saved lon, lat and lgd.

    Expected input files:

        ../temp/LGD/{date_start}_{date_end}/LGD_{arcNo}.hdf5

    Each file should contain:

        time
        lon
        lat
        lgd

    lgd unit:
        m/s^2

    Plot unit:
        nm/s^2
    """

    def __init__(
            self,
            lgd_dir,
            date_span=None):
        """
        Parameters
        ----------
        lgd_dir : str
            LGD root directory, for example:
                "../temp/LGD"

            or direct date-span directory:
                "../temp/LGD/2021-03-01_2021-03-31"

        date_span : list or tuple or None
            Example:
                ["2021-03-01", "2021-03-31"]

            If None, lgd_dir is treated as the actual folder containing
            LGD_*.hdf5 files.
        """

        self.lgd_dir_root = pathlib.Path(lgd_dir)

        if date_span is None:
            self.lgd_dir = self.lgd_dir_root
            self.date_span = None
        else:
            self.date_span = date_span
            candidate = self.lgd_dir_root.joinpath(
                date_span[0] + "_" + date_span[1]
            )

            if candidate.exists():
                self.lgd_dir = candidate
            else:
                self.lgd_dir = self.lgd_dir_root

        self.data = None

    # ============================================================
    # Load LGD data
    # ============================================================

    def load(self):
        """
        Load all LGD_*.hdf5 files in lgd_dir.

        Returns
        -------
        self
        """

        files = sorted(
            self.lgd_dir.glob("LGD_*.hdf5")
        )

        if len(files) == 0:
            raise FileNotFoundError(
                "No LGD_*.hdf5 files found in: %s" % str(self.lgd_dir)
            )

        all_time = []
        all_lon = []
        all_lat = []
        all_lgd = []
        all_arc = []

        for file in files:
            arc_no = self._parse_arc_no(file)

            with h5py.File(file, "r") as h5:
                for key in ["time", "lon", "lat", "lgd"]:
                    if key not in h5:
                        raise KeyError(
                            "%s does not contain dataset: %s"
                            % (str(file), key)
                        )

                time = h5["time"][:]
                lon = h5["lon"][:]
                lat = h5["lat"][:]
                lgd = h5["lgd"][:]

            time = np.asarray(time, dtype=np.float64).reshape(-1)
            lon = np.asarray(lon, dtype=np.float64).reshape(-1)
            lat = np.asarray(lat, dtype=np.float64).reshape(-1)
            lgd = np.asarray(lgd, dtype=np.float64).reshape(-1)

            n = len(time)

            if not (
                len(lon) == n
                and len(lat) == n
                and len(lgd) == n
            ):
                raise ValueError(
                    "Length mismatch in %s: time=%d, lon=%d, lat=%d, lgd=%d"
                    % (
                        str(file),
                        len(time),
                        len(lon),
                        len(lat),
                        len(lgd)
                    )
                )

            valid = (
                np.isfinite(time)
                &
                np.isfinite(lon)
                &
                np.isfinite(lat)
                &
                np.isfinite(lgd)
            )

            all_time.append(time[valid])
            all_lon.append(self._lon_to_180(lon[valid]))
            all_lat.append(lat[valid])
            all_lgd.append(lgd[valid])
            all_arc.append(
                np.ones(np.sum(valid), dtype=np.int64) * int(arc_no)
            )

        self.data = {
            "time": np.concatenate(all_time),
            "lon": np.concatenate(all_lon),
            "lat": np.concatenate(all_lat),
            "lgd": np.concatenate(all_lgd),
            "arcNo": np.concatenate(all_arc),
        }

        index = np.argsort(self.data["time"])

        for key in self.data:
            self.data[key] = self.data[key][index]

        print(
            "Loaded LGD samples:",
            len(self.data["lgd"]),
            "from",
            len(files),
            "files"
        )

        return self

    def _parse_arc_no(self, file):
        """
        Parse arcNo from LGD_{arcNo}.hdf5.
        """

        stem = pathlib.Path(file).stem

        try:
            arc_no = int(stem.split("_")[-1])
        except Exception:
            arc_no = -1

        return arc_no

    # ============================================================
    # Select region
    # ============================================================

    def select_region(
            self,
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            arc_list=None,
            lgd_abs_limit=None):
        """
        Select LGD data within a geographic region.

        Parameters
        ----------
        lon_min, lon_max : float
            Longitude range in degrees, using [-180, 180].

        lat_min, lat_max : float
            Latitude range in degrees.

        arc_list : list or None
            If provided, only selected arcs are used.

        lgd_abs_limit : float or None
            If provided, remove points with abs(lgd * 1e9) larger than this.
            Unit: nm/s^2.

        Returns
        -------
        region : dict
        """

        if self.data is None:
            raise RuntimeError(
                "No data loaded. Please call load() first."
            )

        d = self.data

        lon = self._lon_to_180(d["lon"])

        if lon_min <= lon_max:
            idx_lon = (
                (lon >= lon_min)
                &
                (lon <= lon_max)
            )
        else:
            # Region crosses dateline
            idx_lon = (
                (lon >= lon_min)
                |
                (lon <= lon_max)
            )

        idx = (
            idx_lon
            &
            (d["lat"] >= lat_min)
            &
            (d["lat"] <= lat_max)
        )

        if arc_list is not None:
            idx_arc = np.isin(
                d["arcNo"],
                np.asarray(arc_list, dtype=np.int64)
            )

            idx = idx & idx_arc

        if lgd_abs_limit is not None:
            idx = idx & (
                np.abs(d["lgd"] * 1e9) <= lgd_abs_limit
            )

        region = {
            "time": d["time"][idx],
            "lon": lon[idx],
            "lat": d["lat"][idx],
            "lgd": d["lgd"][idx],
            "arcNo": d["arcNo"][idx],
        }

        print(
            "Selected regional LGD samples:",
            len(region["lgd"])
        )

        return region

    # ============================================================
    # Scatter map
    # ============================================================

    def plot_scatter_map(
            self,
            region,
            lon_lim=None,
            lat_lim=None,
            vmin=None,
            vmax=None,
            percentile_clip=98,
            point_size=5,
            cmap="seismic",
            title="LGD spatial map",
            save_fig=None,
            draw_tracks=True):
        """
        Plot LGD spatial scatter map.

        LGD is shown in nm/s^2.
        """

        self._check_region(region)

        lon = region["lon"]
        lat = region["lat"]
        lgd_nms2 = region["lgd"] * 1e9

        valid = (
            np.isfinite(lon)
            &
            np.isfinite(lat)
            &
            np.isfinite(lgd_nms2)
        )

        lon = lon[valid]
        lat = lat[valid]
        lgd_nms2 = lgd_nms2[valid]

        if len(lgd_nms2) == 0:
            raise RuntimeError("No valid LGD data to plot.")

        if vmin is None or vmax is None:
            vmax_auto = np.nanpercentile(
                np.abs(lgd_nms2),
                percentile_clip
            )

            if vmax is None:
                vmax = vmax_auto

            if vmin is None:
                vmin = -vmax

        if lon_lim is None:
            lon_lim = (
                np.nanmin(lon),
                np.nanmax(lon)
            )

        if lat_lim is None:
            lat_lim = (
                np.nanmin(lat),
                np.nanmax(lat)
            )

        fig, ax = plt.subplots(
            figsize=(7.5, 7.0),
            dpi=180
        )

        if draw_tracks:
            for arc_no in np.unique(region["arcNo"]):
                idx = region["arcNo"] == arc_no

                order = np.argsort(region["time"][idx])

                ax.plot(
                    region["lon"][idx][order],
                    region["lat"][idx][order],
                    color="0.75",
                    linewidth=0.5,
                    alpha=0.5,
                    zorder=1
                )

        sc = ax.scatter(
            lon,
            lat,
            c=lgd_nms2,
            s=point_size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors="none",
            zorder=2
        )

        cb = plt.colorbar(
            sc,
            ax=ax,
            shrink=0.82,
            pad=0.02
        )

        cb.set_label("LGD [nm/s²]")

        ax.set_xlim(*lon_lim)
        ax.set_ylim(*lat_lim)

        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_title(title)

        ax.grid(
            True,
            linewidth=0.4,
            alpha=0.35
        )

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        if save_fig is not None:
            self._make_parent_dir(save_fig)

            plt.savefig(
                save_fig,
                bbox_inches="tight"
            )

            print("Figure saved to:", save_fig)

        plt.show()

    # ============================================================
    # Binned map
    # ============================================================

    def plot_binned_map(
            self,
            region,
            lon_lim,
            lat_lim,
            grid_res=0.25,
            vmin=None,
            vmax=None,
            percentile_clip=98,
            min_count=1,
            cmap="seismic",
            title="Binned LGD spatial map",
            save_fig=None):
        """
        Plot gridded / binned LGD map.

        This is useful when the along-track scatter is too dense.

        Parameters
        ----------
        grid_res : float
            Grid resolution in degree.

        min_count : int
            Minimum number of samples in a grid cell.
        """

        self._check_region(region)

        lon = region["lon"]
        lat = region["lat"]
        lgd_nms2 = region["lgd"] * 1e9

        valid = (
            np.isfinite(lon)
            &
            np.isfinite(lat)
            &
            np.isfinite(lgd_nms2)
            &
            (lon >= lon_lim[0])
            &
            (lon <= lon_lim[1])
            &
            (lat >= lat_lim[0])
            &
            (lat <= lat_lim[1])
        )

        lon = lon[valid]
        lat = lat[valid]
        lgd_nms2 = lgd_nms2[valid]

        if len(lgd_nms2) == 0:
            raise RuntimeError("No valid LGD data to grid.")

        lon_edges = np.arange(
            lon_lim[0],
            lon_lim[1] + grid_res,
            grid_res
        )

        lat_edges = np.arange(
            lat_lim[0],
            lat_lim[1] + grid_res,
            grid_res
        )

        grid_sum = np.zeros(
            (len(lat_edges) - 1, len(lon_edges) - 1),
            dtype=np.float64
        )

        grid_count = np.zeros_like(
            grid_sum,
            dtype=np.int64
        )

        lon_idx = np.searchsorted(
            lon_edges,
            lon,
            side="right"
        ) - 1

        lat_idx = np.searchsorted(
            lat_edges,
            lat,
            side="right"
        ) - 1

        valid_idx = (
            (lon_idx >= 0)
            &
            (lon_idx < grid_sum.shape[1])
            &
            (lat_idx >= 0)
            &
            (lat_idx < grid_sum.shape[0])
        )

        lon_idx = lon_idx[valid_idx]
        lat_idx = lat_idx[valid_idx]
        values = lgd_nms2[valid_idx]

        for i in range(len(values)):
            grid_sum[lat_idx[i], lon_idx[i]] += values[i]
            grid_count[lat_idx[i], lon_idx[i]] += 1

        grid_mean = np.full_like(
            grid_sum,
            np.nan,
            dtype=np.float64
        )

        good = grid_count >= min_count

        grid_mean[good] = (
            grid_sum[good]
            /
            grid_count[good]
        )

        if vmin is None or vmax is None:
            vmax_auto = np.nanpercentile(
                np.abs(grid_mean[np.isfinite(grid_mean)]),
                percentile_clip
            )

            if vmax is None:
                vmax = vmax_auto

            if vmin is None:
                vmin = -vmax

        fig, ax = plt.subplots(
            figsize=(7.5, 7.0),
            dpi=180
        )

        mesh = ax.pcolormesh(
            lon_edges,
            lat_edges,
            grid_mean,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto"
        )

        cb = plt.colorbar(
            mesh,
            ax=ax,
            shrink=0.82,
            pad=0.02
        )

        cb.set_label("LGD [nm/s²]")

        # Overlay raw tracks lightly
        for arc_no in np.unique(region["arcNo"]):
            idx = region["arcNo"] == arc_no

            order = np.argsort(region["time"][idx])

            ax.plot(
                region["lon"][idx][order],
                region["lat"][idx][order],
                color="0.4",
                linewidth=0.35,
                alpha=0.25
            )

        ax.set_xlim(*lon_lim)
        ax.set_ylim(*lat_lim)

        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_title(title)

        ax.grid(
            True,
            linewidth=0.4,
            alpha=0.35
        )

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        if save_fig is not None:
            self._make_parent_dir(save_fig)

            plt.savefig(
                save_fig,
                bbox_inches="tight"
            )

            print("Figure saved to:", save_fig)

        plt.show()

        return {
            "lon_edges": lon_edges,
            "lat_edges": lat_edges,
            "grid_mean": grid_mean,
            "grid_count": grid_count
        }

    # ============================================================
    # Utilities
    # ============================================================

    def print_statistics(self, region):
        """
        Print basic statistics of selected LGD region.
        """

        self._check_region(region)

        lgd_nms2 = region["lgd"] * 1e9

        print("Number of samples:", len(lgd_nms2))
        print("Number of arcs:", len(np.unique(region["arcNo"])))
        print("Lon range:", np.nanmin(region["lon"]), np.nanmax(region["lon"]))
        print("Lat range:", np.nanmin(region["lat"]), np.nanmax(region["lat"]))
        print("LGD [nm/s²]:")
        print("  min   =", np.nanmin(lgd_nms2))
        print("  p01   =", np.nanpercentile(lgd_nms2, 1))
        print("  median=", np.nanmedian(lgd_nms2))
        print("  p99   =", np.nanpercentile(lgd_nms2, 99))
        print("  max   =", np.nanmax(lgd_nms2))

    def _check_region(self, region):
        required = [
            "time",
            "lon",
            "lat",
            "lgd",
            "arcNo"
        ]

        for key in required:
            if key not in region:
                raise KeyError(
                    "region does not contain key: %s" % key
                )

        n = len(region["time"])

        for key in required:
            if len(region[key]) != n:
                raise ValueError(
                    "region key %s has length %d, expected %d"
                    % (key, len(region[key]), n)
                )

    def _lon_to_180(self, lon):
        """
        Convert longitude to [-180, 180).
        """

        lon = np.asarray(
            lon,
            dtype=np.float64
        )

        return (
            (lon + 180.0)
            %
            360.0
        ) - 180.0

    def _make_parent_dir(self, filename):
        out_dir = pathlib.Path(filename).parent

        if str(out_dir) != "." and not out_dir.exists():
            os.makedirs(
                out_dir,
                exist_ok=True
            )


class PlotLGDSpatial:
    """
    Plot spatial distribution of LGD from saved PyHawk LGD products.

    Expected file structure:
        ../temp/LGD/{date_start}_{date_end}/LGD_{arcNo}.hdf5

    Required datasets in each file:
        time
        lon
        lat
        lgd

    LGD unit in file:
        m/s^2

    Plot unit:
        nm/s^2
    """

    def __init__(self, lgd_dir, date_span=None):
        """
        Parameters
        ----------
        lgd_dir : str
            LGD root directory, for example:
                "../temp/LGD"

            Or direct directory containing LGD_*.hdf5.

        date_span : list or tuple or None
            Example:
                ["2021-03-01", "2021-03-31"]

            If date_span is given, the code tries:
                lgd_dir / "2021-03-01_2021-03-31"
        """

        self.lgd_dir_root = pathlib.Path(lgd_dir)

        if date_span is None:
            self.lgd_dir = self.lgd_dir_root
            self.date_span = None
        else:
            self.date_span = date_span
            candidate = self.lgd_dir_root.joinpath(
                date_span[0] + "_" + date_span[1]
            )

            if candidate.exists():
                self.lgd_dir = candidate
            else:
                self.lgd_dir = self.lgd_dir_root

        self.data = None

    # ============================================================
    # Load data
    # ============================================================

    def load(self):
        """
        Load all LGD_*.hdf5 files.

        Returns
        -------
        self
        """

        files = sorted(self.lgd_dir.glob("LGD_*.hdf5"))

        if len(files) == 0:
            raise FileNotFoundError(
                "No LGD_*.hdf5 files found in: %s" % str(self.lgd_dir)
            )

        time_all = []
        lon_all = []
        lat_all = []
        lgd_all = []
        arc_all = []

        for file in files:

            arc_no = self._parse_arc_no(file)

            with h5py.File(file, "r") as h5:

                for key in ["time", "lon", "lat", "lgd"]:
                    if key not in h5:
                        raise KeyError(
                            "%s does not contain dataset: %s"
                            % (str(file), key)
                        )

                time = np.asarray(h5["time"][:], dtype=np.float64).reshape(-1)
                lon = np.asarray(h5["lon"][:], dtype=np.float64).reshape(-1)
                lat = np.asarray(h5["lat"][:], dtype=np.float64).reshape(-1)
                lgd = np.asarray(h5["lgd"][:], dtype=np.float64).reshape(-1)

            n = len(time)

            if not (len(lon) == n and len(lat) == n and len(lgd) == n):
                raise ValueError(
                    "Length mismatch in %s: time=%d, lon=%d, lat=%d, lgd=%d"
                    % (str(file), len(time), len(lon), len(lat), len(lgd))
                )

            valid = (
                np.isfinite(time)
                &
                np.isfinite(lon)
                &
                np.isfinite(lat)
                &
                np.isfinite(lgd)
            )

            time_all.append(time[valid])
            lon_all.append(self._lon_to_180(lon[valid]))
            lat_all.append(lat[valid])
            lgd_all.append(lgd[valid] * 1e9)  # m/s^2 -> nm/s^2
            arc_all.append(
                np.ones(np.sum(valid), dtype=np.int64) * int(arc_no)
            )

        self.data = {
            "time": np.concatenate(time_all),
            "lon": np.concatenate(lon_all),
            "lat": np.concatenate(lat_all),
            "lgd": np.concatenate(lgd_all),
            "arcNo": np.concatenate(arc_all),
        }

        index = np.argsort(self.data["time"])

        for key in self.data:
            self.data[key] = self.data[key][index]

        print("Loaded LGD files:", len(files))
        print("Loaded LGD samples:", len(self.data["lgd"]))
        print("LGD directory:", self.lgd_dir)

        return self

    def _parse_arc_no(self, file):
        """
        Parse arcNo from LGD_{arcNo}.hdf5.
        """

        stem = pathlib.Path(file).stem

        try:
            arc_no = int(stem.split("_")[-1])
        except Exception:
            arc_no = -1

        return arc_no

    # ============================================================
    # Region selection
    # ============================================================

    def select_region(
            self,
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            arc_list=None,
            lgd_abs_limit=None,
            percentile_limit=None):
        """
        Select regional LGD observations.

        Parameters
        ----------
        lon_min, lon_max : float
            Longitude range in degree, using [-180, 180].

        lat_min, lat_max : float
            Latitude range in degree.

        arc_list : list or None
            If not None, only selected arcs are used.

        lgd_abs_limit : float or None
            Remove points with abs(LGD) > lgd_abs_limit.
            Unit: nm/s^2.

        percentile_limit : float or None
            Remove points outside symmetric percentile range.
            For example 99 means keep values within the 99th percentile
            of abs(LGD).

        Returns
        -------
        region : dict
            time, lon, lat, lgd, arcNo
        """

        if self.data is None:
            raise RuntimeError("Please call load() first.")

        d = self.data

        lon = self._lon_to_180(d["lon"])

        if lon_min <= lon_max:
            idx_lon = (
                (lon >= lon_min)
                &
                (lon <= lon_max)
            )
        else:
            # Region crossing dateline
            idx_lon = (
                (lon >= lon_min)
                |
                (lon <= lon_max)
            )

        idx = (
            idx_lon
            &
            (d["lat"] >= lat_min)
            &
            (d["lat"] <= lat_max)
        )

        if arc_list is not None:
            idx = idx & np.isin(
                d["arcNo"],
                np.asarray(arc_list, dtype=np.int64)
            )

        if lgd_abs_limit is not None:
            idx = idx & (
                np.abs(d["lgd"]) <= lgd_abs_limit
            )

        if percentile_limit is not None:
            tmp = np.abs(d["lgd"][idx])
            if len(tmp) > 0:
                bound = np.nanpercentile(tmp, percentile_limit)
                idx = idx & (
                    np.abs(d["lgd"]) <= bound
                )

        region = {
            "time": d["time"][idx],
            "lon": lon[idx],
            "lat": d["lat"][idx],
            "lgd": d["lgd"][idx],
            "arcNo": d["arcNo"][idx],
        }

        print("Selected regional samples:", len(region["lgd"]))
        print("Selected arcs:", len(np.unique(region["arcNo"])))

        return region

    # ============================================================
    # Statistics
    # ============================================================

    def print_statistics(self, region):
        """
        Print basic LGD statistics.
        """

        self._check_region(region)

        lgd = region["lgd"]

        print("\n===== LGD Spatial Statistics =====")
        print("Samples:", len(lgd))
        print("Arcs:", len(np.unique(region["arcNo"])))
        print("Lon range:", np.nanmin(region["lon"]), np.nanmax(region["lon"]))
        print("Lat range:", np.nanmin(region["lat"]), np.nanmax(region["lat"]))
        print("LGD unit: nm/s^2")
        print("min    :", np.nanmin(lgd))
        print("p01    :", np.nanpercentile(lgd, 1))
        print("p05    :", np.nanpercentile(lgd, 5))
        print("median :", np.nanmedian(lgd))
        print("mean   :", np.nanmean(lgd))
        print("p95    :", np.nanpercentile(lgd, 95))
        print("p99    :", np.nanpercentile(lgd, 99))
        print("max    :", np.nanmax(lgd))
        print("std    :", np.nanstd(lgd))
        print("=================================\n")

    # ============================================================
    # Plot: spatial scatter map
    # ============================================================

    def plot_scatter(
            self,
            region,
            lon_lim=None,
            lat_lim=None,
            vmin=None,
            vmax=None,
            percentile_clip=98,
            point_size=8,
            cmap="seismic",
            draw_tracks=True,
            title="LGD spatial distribution",
            save_fig=None):
        """
        Draw LGD spatial scatter map.

        This function does NOT require any predefined grid.
        """

        self._check_region(region)

        lon = region["lon"]
        lat = region["lat"]
        lgd = region["lgd"]

        valid = (
            np.isfinite(lon)
            &
            np.isfinite(lat)
            &
            np.isfinite(lgd)
        )

        lon = lon[valid]
        lat = lat[valid]
        lgd = lgd[valid]

        if len(lgd) == 0:
            raise RuntimeError("No valid LGD data to plot.")

        if lon_lim is None:
            lon_lim = (
                np.nanmin(lon),
                np.nanmax(lon)
            )

        if lat_lim is None:
            lat_lim = (
                np.nanmin(lat),
                np.nanmax(lat)
            )

        if vmin is None or vmax is None:
            auto_v = np.nanpercentile(
                np.abs(lgd),
                percentile_clip
            )

            if vmax is None:
                vmax = auto_v

            if vmin is None:
                vmin = -auto_v

        fig, ax = plt.subplots(
            figsize=(8.0, 7.0),
            dpi=180
        )

        if draw_tracks:
            self._draw_tracks(
                ax=ax,
                region=region,
                lon_lim=lon_lim,
                lat_lim=lat_lim
            )

        sc = ax.scatter(
            lon,
            lat,
            c=lgd,
            s=point_size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors="none",
            alpha=0.95,
            zorder=3
        )

        cb = plt.colorbar(
            sc,
            ax=ax,
            shrink=0.82,
            pad=0.02
        )

        cb.set_label("LGD [nm/s²]")

        ax.set_xlim(*lon_lim)
        ax.set_ylim(*lat_lim)

        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_title(title)

        ax.grid(
            True,
            linewidth=0.4,
            alpha=0.35
        )

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        if save_fig is not None:
            self._make_parent_dir(save_fig)
            plt.savefig(
                save_fig,
                bbox_inches="tight"
            )
            print("Figure saved to:", save_fig)

        plt.show()

    # ============================================================
    # Plot: simple binned map generated from LGD itself
    # ============================================================

    def plot_self_binned(
            self,
            region,
            lon_lim,
            lat_lim,
            grid_res=0.25,
            statistic="median",
            min_count=1,
            vmin=None,
            vmax=None,
            percentile_clip=98,
            cmap="seismic",
            draw_tracks=True,
            title="Binned LGD spatial distribution",
            save_fig=None):
        """
        Create a simple grid from LGD itself and plot it.

        This does not need an external predefined grid.
        It only bins your saved lon/lat/LGD points.

        Parameters
        ----------
        statistic : str
            "median" or "mean"
        """

        self._check_region(region)

        grid = self.bin_to_grid(
            region=region,
            lon_lim=lon_lim,
            lat_lim=lat_lim,
            grid_res=grid_res,
            statistic=statistic,
            min_count=min_count
        )

        values = grid["grid"]

        if np.all(~np.isfinite(values)):
            raise RuntimeError("No valid grid cells after binning.")

        if vmin is None or vmax is None:
            auto_v = np.nanpercentile(
                np.abs(values[np.isfinite(values)]),
                percentile_clip
            )

            if vmax is None:
                vmax = auto_v

            if vmin is None:
                vmin = -auto_v

        fig, ax = plt.subplots(
            figsize=(8.0, 7.0),
            dpi=180
        )

        mesh = ax.pcolormesh(
            grid["lon_edges"],
            grid["lat_edges"],
            values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            zorder=2
        )

        cb = plt.colorbar(
            mesh,
            ax=ax,
            shrink=0.82,
            pad=0.02
        )

        cb.set_label("LGD [nm/s²]")

        if draw_tracks:
            self._draw_tracks(
                ax=ax,
                region=region,
                lon_lim=lon_lim,
                lat_lim=lat_lim,
                zorder=3
            )

        ax.set_xlim(*lon_lim)
        ax.set_ylim(*lat_lim)

        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_title(title)

        ax.grid(
            True,
            linewidth=0.4,
            alpha=0.35
        )

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        if save_fig is not None:
            self._make_parent_dir(save_fig)
            plt.savefig(
                save_fig,
                bbox_inches="tight"
            )
            print("Figure saved to:", save_fig)

        plt.show()

        return grid

    def bin_to_grid(
            self,
            region,
            lon_lim,
            lat_lim,
            grid_res=0.25,
            statistic="median",
            min_count=1):
        """
        Bin lon/lat/LGD points into a simple regular grid.

        This grid is generated only from LGD observations.
        It is not related to spherical harmonics or EWH.
        """

        lon = region["lon"]
        lat = region["lat"]
        lgd = region["lgd"]

        valid = (
            np.isfinite(lon)
            &
            np.isfinite(lat)
            &
            np.isfinite(lgd)
            &
            (lon >= lon_lim[0])
            &
            (lon <= lon_lim[1])
            &
            (lat >= lat_lim[0])
            &
            (lat <= lat_lim[1])
        )

        lon = lon[valid]
        lat = lat[valid]
        lgd = lgd[valid]

        lon_edges = np.arange(
            lon_lim[0],
            lon_lim[1] + grid_res,
            grid_res
        )

        lat_edges = np.arange(
            lat_lim[0],
            lat_lim[1] + grid_res,
            grid_res
        )

        nlon = len(lon_edges) - 1
        nlat = len(lat_edges) - 1

        cell_values = [[[] for _ in range(nlon)] for _ in range(nlat)]

        lon_idx = np.searchsorted(
            lon_edges,
            lon,
            side="right"
        ) - 1

        lat_idx = np.searchsorted(
            lat_edges,
            lat,
            side="right"
        ) - 1

        valid_idx = (
            (lon_idx >= 0)
            &
            (lon_idx < nlon)
            &
            (lat_idx >= 0)
            &
            (lat_idx < nlat)
        )

        for i, j, value in zip(
                lat_idx[valid_idx],
                lon_idx[valid_idx],
                lgd[valid_idx]):
            cell_values[i][j].append(value)

        grid = np.full(
            (nlat, nlon),
            np.nan,
            dtype=np.float64
        )

        count = np.zeros(
            (nlat, nlon),
            dtype=np.int64
        )

        for i in range(nlat):
            for j in range(nlon):
                if len(cell_values[i][j]) < min_count:
                    continue

                count[i, j] = len(cell_values[i][j])

                if statistic == "median":
                    grid[i, j] = np.nanmedian(cell_values[i][j])
                elif statistic == "mean":
                    grid[i, j] = np.nanmean(cell_values[i][j])
                else:
                    raise ValueError(
                        "Unsupported statistic: %s. Use 'median' or 'mean'."
                        % statistic
                    )

        lon_center = 0.5 * (
            lon_edges[:-1]
            +
            lon_edges[1:]
        )

        lat_center = 0.5 * (
            lat_edges[:-1]
            +
            lat_edges[1:]
        )

        return {
            "lon": lon_center,
            "lat": lat_center,
            "lon_edges": lon_edges,
            "lat_edges": lat_edges,
            "grid": grid,
            "count": count
        }

    # ============================================================
    # Internal utilities
    # ============================================================

    def _draw_tracks(
            self,
            ax,
            region,
            lon_lim,
            lat_lim,
            zorder=1):
        """
        Draw gray ground tracks.

        For global maps, this function splits tracks by:
            1. time gaps
            2. longitude jumps across ±180 deg

        This avoids artificial long lines across the map.
        """

        for arc_no in np.unique(region["arcNo"]):

            idx = region["arcNo"] == arc_no

            time = region["time"][idx]
            lon = region["lon"][idx]
            lat = region["lat"][idx]

            valid = (
                    np.isfinite(time)
                    &
                    np.isfinite(lon)
                    &
                    np.isfinite(lat)
                    &
                    (lon >= lon_lim[0])
                    &
                    (lon <= lon_lim[1])
                    &
                    (lat >= lat_lim[0])
                    &
                    (lat <= lat_lim[1])
            )

            time = time[valid]
            lon = lon[valid]
            lat = lat[valid]

            if len(time) < 2:
                continue

            order = np.argsort(time)

            time = time[order]
            lon = lon[order]
            lat = lat[order]

            dt = np.diff(time)

            dt_positive = dt[dt > 0]

            if len(dt_positive) == 0:
                continue

            dt_med = np.nanmedian(dt_positive)

            time_gap_threshold = max(
                60.0,
                5.0 * dt_med
            )

            time_breaks = np.where(
                dt > time_gap_threshold
            )[0] + 1

            # Important for global maps:
            # split when crossing the dateline.
            lon_jump = np.abs(np.diff(lon))

            lon_breaks = np.where(
                lon_jump > 180.0
            )[0] + 1

            breaks = np.unique(
                np.concatenate(
                    [
                        time_breaks,
                        lon_breaks
                    ]
                )
            )

            starts = np.insert(
                breaks,
                0,
                0
            )

            ends = np.append(
                breaks,
                len(time)
            )

            for s, e in zip(starts, ends):

                if e - s < 2:
                    continue

                ax.plot(
                    lon[s:e],
                    lat[s:e],
                    color="0.65",
                    linewidth=0.35,
                    alpha=0.35,
                    zorder=zorder
                )

    def _check_region(self, region):
        required = [
            "time",
            "lon",
            "lat",
            "lgd",
            "arcNo"
        ]

        for key in required:
            if key not in region:
                raise KeyError(
                    "region does not contain key: %s" % key
                )

        n = len(region["time"])

        for key in required:
            if len(region[key]) != n:
                raise ValueError(
                    "region key %s has length %d, expected %d"
                    % (key, len(region[key]), n)
                )

    def _lon_to_180(self, lon):
        """
        Convert longitude to [-180, 180).
        """

        lon = np.asarray(
            lon,
            dtype=np.float64
        )

        return (
            (lon + 180.0)
            %
            360.0
        ) - 180.0

    def _make_parent_dir(self, filename):
        out_dir = pathlib.Path(filename).parent

        if str(out_dir) != "." and not out_dir.exists():
            os.makedirs(
                out_dir,
                exist_ok=True
            )

    def remove_arc_median(self, region):
        """
        Remove median LGD value for each arc.

        This is useful for global LGD spatial visualization, because
        each arc may contain a residual bias after filtering and transfer.
        """

        self._check_region(region)

        new_region = {}

        for key in region:
            new_region[key] = np.copy(region[key])

        lgd = new_region["lgd"]

        for arc_no in np.unique(new_region["arcNo"]):
            idx = new_region["arcNo"] == arc_no

            if np.sum(idx) < 5:
                continue

            med = np.nanmedian(lgd[idx])

            lgd[idx] = lgd[idx] - med

        new_region["lgd"] = lgd

        return new_region

    def split_ascending_descending(self, region):
        """
        Split region into ascending and descending samples by arc.
        """

        self._check_region(region)

        asc_idx_all = np.zeros(len(region["time"]), dtype=bool)
        desc_idx_all = np.zeros(len(region["time"]), dtype=bool)

        for arc_no in np.unique(region["arcNo"]):
            idx = region["arcNo"] == arc_no

            time = region["time"][idx]
            lat = region["lat"][idx]

            if len(time) < 5:
                continue

            order = np.argsort(time)

            time_o = time[order]
            lat_o = lat[order]

            coef = np.polyfit(
                time_o - time_o[0],
                lat_o,
                1
            )

            idx_global = np.where(idx)[0]

            if coef[0] > 0:
                asc_idx_all[idx_global] = True
            else:
                desc_idx_all[idx_global] = True

        asc = {}
        desc = {}

        for key in region:
            asc[key] = region[key][asc_idx_all]
            desc[key] = region[key][desc_idx_all]

        return asc, desc

def demo1():

    plotter = PlotLGDMap(
        lgd_dir="../temp/LGD",
        date_span=["2021-03-01", "2021-03-31"]
    ).load()

    region = plotter.select_region(
        lon_min=-180,
        lon_max=180,
        lat_min=-90,
        lat_max=90,
        lgd_abs_limit=None
    )

    plotter.print_statistics(region)

    plotter.plot_scatter_map(
        region=region,
        lon_lim=(-180, 180),
        lat_lim=(-90, 90),
        vmin=-5,
        vmax=5,
        point_size=8,
        cmap="seismic",
        title="GRACE-FO LGD spatial map",
        save_fig="../result/img/LGD_spatial_scatter.png",
        draw_tracks=True
    )
    pass

def demo2():
    plotter = PlotLGDSpatial(
        lgd_dir="../temp/LGD",
        date_span=["2021-03-01", "2021-03-31"]
    ).load()

    region = plotter.select_region(
        lon_min=-180,
        lon_max=180,
        lat_min=-90,
        lat_max=90,
        lgd_abs_limit=None,
        percentile_limit=99
    )

    region_arc_demean = plotter.remove_arc_median(region)

    asc, desc = plotter.split_ascending_descending(region_arc_demean)


    plotter.print_statistics(region_arc_demean)
    plotter.plot_self_binned(
        region=asc,
        lon_lim=(-180, 180),
        lat_lim=(-90, 90),
        grid_res=1.0,
        statistic="median",
        min_count=1,
        vmin=-15,
        vmax=15,
        cmap="seismic",
        draw_tracks=False,
        title="Global arc-demeaned LGD, ascending tracks",
        save_fig="../result/img/LGD_global_ascending.png"
    )

    plotter.plot_self_binned(
        region=desc,
        lon_lim=(-180, 180),
        lat_lim=(-90, 90),
        grid_res=1.0,
        statistic="median",
        min_count=1,
        vmin=-15,
        vmax=15,
        cmap="seismic",
        draw_tracks=False,
        title="Global arc-demeaned LGD, descending tracks",
        save_fig="../result/img/LGD_global_descending.png"
    )
    # grid = plotter.plot_self_binned(
    #     region=region_arc_demean,
    #     lon_lim=(-180, 180),
    #     lat_lim=(-90, 90),
    #     grid_res=1.0,
    #     statistic="median",
    #     min_count=1,
    #     vmin=-15,
    #     vmax=15,
    #     cmap="seismic",
    #     draw_tracks=False,
    #     title="Global binned GRACE-FO LGD spatial distribution",
    #     save_fig="../result/img/LGD_global_binned.png"
    # )


if __name__ == '__main__':
    demo2()