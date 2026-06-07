import os
import csv
import h5py
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

import src.Frame.SOFA as sf
from src.Preference import Pre_Constants


class AnalyzeLGD:
    """
    Analyze along-track LGD products.

    Input files:
        {lgd_dir}/{date_start}_{date_end}/LGD_{arcNo}.hdf5

    Required datasets in each LGD file:
        time
        lon
        lat
        lgd

    Optional datasets:
        range_acc
        kbrr_residual
        kbrr_filtered

    Main workflow:
        1. load_all_arcs()
        2. select_region()
        3. summarize_tracks()
        4. diagnose_days()
        5. find_repeat_pairs()
        6. plot_repeat_tracks()
        7. compute_pair_difference()
    """

    def __init__(self, lgd_dir, date_span):
        """
        Parameters
        ----------
        lgd_dir : str
            Root directory of LGD products, for example:
                "../temp/LGD"

        date_span : list or tuple
            [date_start, date_end], for example:
                ["2021-03-01", "2021-03-31"]
        """

        self.lgd_dir_root = pathlib.Path(lgd_dir)
        self.date_span = date_span

        candidate_dir = self.lgd_dir_root.joinpath(
            date_span[0] + "_" + date_span[1]
        )

        # If lgd_dir itself already points to a date_span directory,
        # use it directly.
        if candidate_dir.exists():
            self.lgd_dir = candidate_dir
        else:
            self.lgd_dir = self.lgd_dir_root

        self.data = None

    # ============================================================
    # Basic loading
    # ============================================================

    def load_all_arcs(self):
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
                self._check_required_datasets(
                    h5=h5,
                    filename=str(file),
                    keys=["time", "lon", "lat", "lgd"]
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
            lgd_all.append(lgd[valid])
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

        return self

    def _parse_arc_no(self, file):
        """
        Parse arc number from LGD_{arcNo}.hdf5.
        """

        stem = pathlib.Path(file).stem

        try:
            arc_no = int(stem.split("_")[-1])
        except Exception:
            raise ValueError(
                "Cannot parse arc number from filename: %s" % str(file)
            )

        return arc_no

    def _check_required_datasets(self, h5, filename, keys):
        """
        Check required datasets in HDF5 file.
        """

        for key in keys:
            if key not in h5:
                raise KeyError(
                    "File %s does not contain dataset: %s"
                    % (filename, key)
                )

    # ============================================================
    # Region selection
    # ============================================================

    def select_region(
            self,
            lon_min,
            lon_max,
            lat_min,
            lat_max):
        """
        Select LGD observations within a lon-lat box.

        Parameters
        ----------
        lon_min, lon_max : float
            Longitude range in degrees, using [-180, 180].

        lat_min, lat_max : float
            Latitude range in degrees.

        Returns
        -------
        region : dict
            time, lon, lat, lgd, arcNo
        """

        if self.data is None:
            raise RuntimeError(
                "No LGD data loaded. Please call load_all_arcs() first."
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
            # Support crossing dateline, for example lon_min=170, lon_max=-170
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

        return {
            "time": d["time"][idx],
            "lon": lon[idx],
            "lat": d["lat"][idx],
            "lgd": d["lgd"][idx],
            "arcNo": d["arcNo"][idx],
        }

    # ============================================================
    # Track summary
    # ============================================================

    def summarize_tracks(
            self,
            region,
            lat_band=(-38.0, -22.0),
            lat_ref=-30.0,
            min_points=20):
        """
        Summarize each LGD ground-track segment.

        If region contains trackID, tracks are grouped by trackID.
        Otherwise, tracks are grouped by arcNo.
        """

        self._check_region(region)

        if "trackID" in region:
            group_key = "trackID"
        else:
            group_key = "arcNo"

        summaries = []

        for group_id in np.unique(region[group_key]):

            idx = region[group_key] == group_id

            time = np.asarray(region["time"][idx], dtype=np.float64)
            lon = np.asarray(region["lon"][idx], dtype=np.float64)
            lat = np.asarray(region["lat"][idx], dtype=np.float64)
            arc_no_arr = np.asarray(region["arcNo"][idx], dtype=np.int64)

            valid = (
                    np.isfinite(time)
                    &
                    np.isfinite(lon)
                    &
                    np.isfinite(lat)
            )

            time = time[valid]
            lon = self._lon_to_180(lon[valid])
            lat = lat[valid]
            arc_no_arr = arc_no_arr[valid]

            if len(time) < min_points:
                continue

            order_time = np.argsort(time)

            time = time[order_time]
            lon = lon[order_time]
            lat = lat[order_time]
            arc_no_arr = arc_no_arr[order_time]

            idx_band = (
                    (lat >= lat_band[0])
                    &
                    (lat <= lat_band[1])
            )

            if np.sum(idx_band) < min_points:
                continue

            time_b = time[idx_band]
            lon_b = lon[idx_band]
            lat_b = lat[idx_band]

            if np.nanmax(time_b) == np.nanmin(time_b):
                continue

            coef = np.polyfit(
                time_b - time_b[0],
                lat_b,
                1
            )

            if coef[0] > 0:
                direction = "ascending"
            else:
                direction = "descending"

            lon_ref = self._interp_lon_at_lat(
                lon=lon_b,
                lat=lat_b,
                lat_ref=lat_ref
            )

            if not np.isfinite(lon_ref):
                lon_ref = np.nanmedian(lon_b)

            mean_lon = np.nanmedian(lon_b)
            mean_time = np.nanmedian(time_b)

            date = self.gps_seconds_to_datetime(mean_time)

            summaries.append({
                "trackID": int(group_id),
                "arcNo": int(np.nanmedian(arc_no_arr)),
                "date": date.strftime("%Y-%m-%d"),
                "day": int(date.day),
                "mean_time": float(mean_time),
                "lon_ref": float(lon_ref),
                "mean_lon": float(mean_lon),
                "lat_min": float(np.nanmin(lat_b)),
                "lat_max": float(np.nanmax(lat_b)),
                "direction": direction,
                "n_points": int(np.sum(idx_band)),
            })

        summaries.sort(
            key=lambda x: (
                x["day"],
                x["direction"],
                x["lon_ref"]
            )
        )

        return summaries

    def diagnose_days(self, summaries, required_days=None):
        """
        Print available days and missing required days.
        """

        if len(summaries) == 0:
            print("No track summaries found.")
            return

        available_days = sorted(
            set([s["day"] for s in summaries])
        )

        print("Available days:", available_days)

        if required_days is not None:
            missing_days = [
                d for d in required_days
                if d not in available_days
            ]

            print("Required days:", required_days)
            print("Missing days:", missing_days)

        print("\nTrack summary:")

        for s in summaries:
            print(
                "arcNo:",
                s["arcNo"],
                "date:",
                s["date"],
                "day:",
                s["day"],
                "direction:",
                s["direction"],
                "lon_ref:",
                round(s["lon_ref"], 3),
                "mean_lon:",
                round(s["mean_lon"], 3),
                "lat_range:",
                round(s["lat_min"], 2),
                round(s["lat_max"], 2),
                "n:",
                s["n_points"]
            )

    # ============================================================
    # Repeat-track pairing
    # ============================================================

    def find_repeat_pairs(
            self,
            summaries,
            day_pairs=None,
            repeat_days=11.0,
            day_tolerance=1.0,
            lon_tolerance=0.8,
            direction=None,
            same_direction=True,
            one_pair_per_day_pair=True,
            verbose=True):
        """
        Find repeat-track pairs using trackID.
        """

        pairs = []

        if len(summaries) == 0:
            return pairs

        def get_track_id(s):
            return s["trackID"] if "trackID" in s else s["arcNo"]

        if day_pairs is not None:

            for after_day, before_day in day_pairs:

                after_tracks = [
                    s for s in summaries
                    if s["day"] == after_day
                ]

                before_tracks = [
                    s for s in summaries
                    if s["day"] == before_day
                ]

                if direction is not None:
                    after_tracks = [
                        s for s in after_tracks
                        if s["direction"] == direction
                    ]

                    before_tracks = [
                        s for s in before_tracks
                        if s["direction"] == direction
                    ]

                if verbose:
                    print(
                        "day pair:",
                        after_day,
                        before_day,
                        "| after:",
                        len(after_tracks),
                        "| before:",
                        len(before_tracks),
                        "| direction:",
                        direction
                    )

                candidates_all = []

                for a in after_tracks:
                    for b in before_tracks:

                        if same_direction and a["direction"] != b["direction"]:
                            continue

                        dlon = abs(
                            self._lon_to_180(
                                a["lon_ref"] - b["lon_ref"]
                            )
                        )

                        if dlon <= lon_tolerance:
                            candidates_all.append((dlon, a, b))

                if len(candidates_all) == 0:
                    continue

                candidates_all.sort(key=lambda x: x[0])

                if one_pair_per_day_pair:
                    candidates_all = [candidates_all[0]]

                for dlon, a, b in candidates_all:
                    pairs.append({
                        "after_track": get_track_id(a),
                        "before_track": get_track_id(b),
                        "after_arc": a["arcNo"],
                        "before_arc": b["arcNo"],
                        "after_day": a["day"],
                        "before_day": b["day"],
                        "after_date": a["date"],
                        "before_date": b["date"],
                        "lon_after": a["lon_ref"],
                        "lon_before": b["lon_ref"],
                        "dlon": float(dlon),
                        "direction": a["direction"],
                    })

        else:

            for a in summaries:

                if direction is not None and a["direction"] != direction:
                    continue

                candidates = []

                for b in summaries:

                    if get_track_id(a) == get_track_id(b):
                        continue

                    if direction is not None and b["direction"] != direction:
                        continue

                    if same_direction and a["direction"] != b["direction"]:
                        continue

                    dt_days = (
                                      a["mean_time"]
                                      -
                                      b["mean_time"]
                              ) / 86400.0

                    if abs(dt_days - repeat_days) > day_tolerance:
                        continue

                    dlon = abs(
                        self._lon_to_180(
                            a["lon_ref"] - b["lon_ref"]
                        )
                    )

                    if dlon > lon_tolerance:
                        continue

                    candidates.append(
                        (
                            dlon,
                            abs(dt_days - repeat_days),
                            a,
                            b
                        )
                    )

                if len(candidates) == 0:
                    continue

                candidates.sort(
                    key=lambda x: (
                        x[0],
                        x[1]
                    )
                )

                dlon, _, a_best, b_best = candidates[0]

                pairs.append({
                    "after_track": get_track_id(a_best),
                    "before_track": get_track_id(b_best),
                    "after_arc": a_best["arcNo"],
                    "before_arc": b_best["arcNo"],
                    "after_day": a_best["day"],
                    "before_day": b_best["day"],
                    "after_date": a_best["date"],
                    "before_date": b_best["date"],
                    "lon_after": a_best["lon_ref"],
                    "lon_before": b_best["lon_ref"],
                    "dlon": float(dlon),
                    "direction": a_best["direction"],
                })

        pairs.sort(
            key=lambda p: 0.5 * (
                    p["lon_after"] + p["lon_before"]
            )
        )

        return pairs

    def print_pairs(self, pairs):
        """
        Print repeat-track pairs.
        """

        if len(pairs) == 0:
            print("No repeat-track pairs found.")
            return

        for p in pairs:
            print(
                "after:",
                p["after_date"],
                "arc",
                p["after_arc"],
                "| before:",
                p["before_date"],
                "arc",
                p["before_arc"],
                "| direction:",
                p["direction"],
                "| lon_after:",
                round(p["lon_after"], 3),
                "| lon_before:",
                round(p["lon_before"], 3),
                "| dlon:",
                round(p["dlon"], 3)
            )

    # ============================================================
    # Plotting
    # ============================================================

    def plot_repeat_tracks(
            self,
            region,
            pairs,
            lon_lim=(135, 165),
            lat_lim=(-65, 5),
            lat_band=(-38, -22),
            offset_step=4.5,
            save_fig=None,
            clean_lgd=True,
            lgd_mad_threshold=8.0,
            lgd_abs_limit=None,
            lgd_jump_limit=4.0):
        """
        Plot ground tracks and along-track LGD curves by trackID.

        Black = after event
        Gray  = before event

        The LGD curves are plotted in time order, not latitude order.
        """

        self._check_region(region)

        if "trackID" not in region:
            raise KeyError(
                "region has no trackID. Please call split_region_into_tracks() first."
            )

        if len(pairs) == 0:
            raise ValueError("pairs is empty. Cannot plot repeat tracks.")

        fig = plt.figure(
            figsize=(13.5, 6.2),
            dpi=180
        )

        ax_map = fig.add_axes([0.07, 0.14, 0.25, 0.76])
        ax_lgd = fig.add_axes([0.39, 0.14, 0.57, 0.76])

        # ============================================================
        # Left panel: ground tracks
        # ============================================================

        for pair in pairs:

            for track_id, color, linewidth, alpha in [
                (pair["before_track"], "0.65", 1.0, 0.9),
                (pair["after_track"], "k", 1.2, 0.95)
            ]:
                track = self._extract_track(
                    region=region,
                    trackID=track_id,
                    lat_lim=lat_lim,
                    clean_lgd=False
                )

                if track is None:
                    continue

                ax_map.plot(
                    track["lon"],
                    track["lat"],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha
                )

        ax_map.axhline(
            lat_band[0],
            color="0.35",
            linewidth=0.8,
            linestyle="--"
        )

        ax_map.axhline(
            lat_band[1],
            color="0.35",
            linewidth=0.8,
            linestyle="--"
        )

        ax_map.set_xlim(*lon_lim)
        ax_map.set_ylim(*lat_lim)
        ax_map.set_xlabel("Longitude [deg]")
        ax_map.set_ylabel("Latitude [deg]")
        ax_map.set_title("(a) Ground tracks", fontsize=13, pad=10)
        ax_map.grid(True, linewidth=0.4, alpha=0.35)

        # ============================================================
        # Right panel: LGD profiles
        # ============================================================

        plotted_x = []

        for i, pair in enumerate(pairs):

            offset = i * offset_step

            for track_id, color, linewidth, alpha in [
                (pair["before_track"], "0.65", 1.0, 0.85),
                (pair["after_track"], "k", 1.2, 0.95)
            ]:
                track = self._extract_track(
                    region=region,
                    trackID=track_id,
                    lat_lim=lat_lim,
                    clean_lgd=clean_lgd,
                    lgd_mad_threshold=lgd_mad_threshold,
                    lgd_abs_limit=lgd_abs_limit,
                    lgd_jump_limit=lgd_jump_limit
                )

                if track is None:
                    continue

                lat = track["lat"]
                lgd = track["lgd"] * 1e9

                x = lgd + offset

                plotted_x.append(x)

                # Important:
                # Do NOT sort by latitude here.
                # The track has already been sorted by time in _extract_track().
                ax_lgd.plot(
                    x,
                    lat,
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha
                )

                ax_lgd.scatter(
                    x,
                    lat,
                    color=color,
                    s=3,
                    alpha=0.55
                )

            label_text = "%d / %d" % (
                pair["after_day"],
                pair["before_day"]
            )

            ax_lgd.text(
                offset,
                lat_lim[1] - 4.0,
                label_text,
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold"
            )

        ax_lgd.axhline(
            lat_band[0],
            color="0.35",
            linewidth=0.8,
            linestyle="--"
        )

        ax_lgd.axhline(
            lat_band[1],
            color="0.35",
            linewidth=0.8,
            linestyle="--"
        )

        ax_lgd.set_ylim(*lat_lim)

        if len(plotted_x) > 0:
            x_all = np.concatenate(plotted_x)
            x_all = x_all[np.isfinite(x_all)]

            if len(x_all) > 0:
                x_min = np.nanpercentile(x_all, 1) - 0.5
                x_max = np.nanpercentile(x_all, 99) + 0.5

                ax_lgd.set_xlim(
                    x_min,
                    x_max
                )

        ax_lgd.set_xlabel("LGD [nm/s²] + track offset")
        ax_lgd.set_ylabel("Latitude [deg]")
        ax_lgd.set_title("(b) Along-track LGD", fontsize=13, pad=14)
        ax_lgd.grid(True, linewidth=0.4, alpha=0.35)

        ax_lgd.text(
            0.02,
            0.02,
            "Black = after event; gray = before event",
            transform=ax_lgd.transAxes,
            fontsize=10,
            ha="left",
            va="bottom"
        )

        if save_fig is not None:
            out_dir = pathlib.Path(save_fig).parent

            if str(out_dir) != "." and not out_dir.exists():
                os.makedirs(out_dir, exist_ok=True)

            plt.savefig(
                save_fig,
                bbox_inches="tight"
            )

        plt.show()

    def plot_single_tracks(
            self,
            region,
            arcs,
            lat_lim=(-65, 5),
            save_fig=None):
        """
        Plot selected individual LGD tracks.
        """

        self._check_region(region)

        plt.figure(
            figsize=(5, 7),
            dpi=200
        )

        for arc in arcs:
            track = self._extract_arc_track(
                region=region,
                arcNo=arc,
                lat_lim=lat_lim
            )

            if track is None:
                continue

            order = np.argsort(track["lat"])

            plt.plot(
                track["lgd"][order] * 1e9,
                track["lat"][order],
                linewidth=1.0,
                label="arc %s" % arc
            )

        plt.axvline(
            0,
            color="0.5",
            linewidth=0.8
        )

        plt.xlabel("LGD [nm/s²]")
        plt.ylabel("Latitude [deg]")
        plt.ylim(*lat_lim)
        plt.grid(True, linewidth=0.4, alpha=0.4)
        plt.legend()

        if save_fig is not None:
            out_dir = pathlib.Path(save_fig).parent
            if str(out_dir) != "." and not out_dir.exists():
                os.makedirs(out_dir, exist_ok=True)

            plt.savefig(
                save_fig,
                bbox_inches="tight"
            )

        plt.show()

    # ============================================================
    # Difference analysis
    # ============================================================

    def compute_pair_difference(
            self,
            region,
            pair,
            lat_grid=None,
            lat_lim=(-65, 5)):
        """
        Compute after-before LGD difference on common latitude grid.

        Parameters
        ----------
        region : dict
            Output from select_region().

        pair : dict
            One element from find_repeat_pairs().

        lat_grid : ndarray or None
            If None, default grid is 0.1 degree.

        lat_lim : tuple
            Latitude range for interpolation.

        Returns
        -------
        lat_grid : ndarray
        delta_lgd : ndarray
            after - before, unit nm/s^2
        after_lgd : ndarray
            interpolated after LGD, unit nm/s^2
        before_lgd : ndarray
            interpolated before LGD, unit nm/s^2
        """

        if lat_grid is None:
            lat_grid = np.arange(
                lat_lim[0],
                lat_lim[1] + 0.001,
                0.1
            )

        after = self._extract_arc_track(
            region=region,
            arcNo=pair["after_arc"],
            lat_lim=lat_lim
        )

        before = self._extract_arc_track(
            region=region,
            arcNo=pair["before_arc"],
            lat_lim=lat_lim
        )

        if after is None or before is None:
            raise ValueError(
                "Cannot extract after or before track for pair: %s" % str(pair)
            )

        after_lgd = self._interp_lgd_to_lat(
            lat=after["lat"],
            lgd=after["lgd"] * 1e9,
            lat_grid=lat_grid
        )

        before_lgd = self._interp_lgd_to_lat(
            lat=before["lat"],
            lgd=before["lgd"] * 1e9,
            lat_grid=lat_grid
        )

        delta_lgd = after_lgd - before_lgd

        return lat_grid, delta_lgd, after_lgd, before_lgd

    def plot_pair_difference(
            self,
            region,
            pair,
            lat_grid=None,
            lat_lim=(-65, 5),
            save_fig=None):
        """
        Plot after-before LGD difference for one repeat-track pair.
        """

        lat_grid, delta, after_lgd, before_lgd = self.compute_pair_difference(
            region=region,
            pair=pair,
            lat_grid=lat_grid,
            lat_lim=lat_lim
        )

        plt.figure(
            figsize=(4.5, 7),
            dpi=200
        )

        plt.plot(
            delta,
            lat_grid,
            "k-",
            linewidth=1.2,
            label="after - before"
        )

        plt.axvline(
            0,
            color="0.5",
            linewidth=0.8
        )

        plt.xlabel("ΔLGD [nm/s²]")
        plt.ylabel("Latitude [deg]")
        plt.ylim(*lat_lim)
        plt.title(
            "%s arc %s - %s arc %s"
            % (
                pair["after_date"],
                pair["after_arc"],
                pair["before_date"],
                pair["before_arc"]
            )
        )
        plt.grid(True, linewidth=0.4, alpha=0.4)
        plt.legend()

        if save_fig is not None:
            out_dir = pathlib.Path(save_fig).parent
            if str(out_dir) != "." and not out_dir.exists():
                os.makedirs(out_dir, exist_ok=True)

            plt.savefig(
                save_fig,
                bbox_inches="tight"
            )

        plt.show()

    def save_pairs_csv(self, pairs, filename):
        """
        Save repeat-track pair information to CSV.
        """

        if len(pairs) == 0:
            print("No pairs to save.")
            return

        out_dir = pathlib.Path(filename).parent
        if str(out_dir) != "." and not out_dir.exists():
            os.makedirs(out_dir, exist_ok=True)

        keys = [
            "after_arc",
            "before_arc",
            "after_day",
            "before_day",
            "after_date",
            "before_date",
            "lon_after",
            "lon_before",
            "dlon",
            "direction"
        ]

        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=keys
            )

            writer.writeheader()

            for p in pairs:
                writer.writerow({
                    key: p.get(key, None)
                    for key in keys
                })

        print("Pairs saved to:", filename)

    # ============================================================
    # Internal utilities
    # ============================================================

    def _check_region(self, region):
        """
        Check region dict.
        """

        required_keys = [
            "time",
            "lon",
            "lat",
            "lgd",
            "arcNo"
        ]

        for key in required_keys:
            if key not in region:
                raise KeyError(
                    "region does not contain required key: %s" % key
                )

        n = len(region["time"])

        for key in required_keys:
            if len(region[key]) != n:
                raise ValueError(
                    "region key length mismatch: %s has length %d, expected %d"
                    % (key, len(region[key]), n)
                )

    def _extract_arc_track(
            self,
            region,
            arcNo,
            lat_lim=None):
        """
        Extract one arc from region.
        """

        idx = region["arcNo"] == arcNo

        if np.sum(idx) == 0:
            return None

        time = region["time"][idx]
        lon = region["lon"][idx]
        lat = region["lat"][idx]
        lgd = region["lgd"][idx]

        valid = (
            np.isfinite(time)
            &
            np.isfinite(lon)
            &
            np.isfinite(lat)
            &
            np.isfinite(lgd)
        )

        if lat_lim is not None:
            valid = (
                valid
                &
                (lat >= lat_lim[0])
                &
                (lat <= lat_lim[1])
            )

        if np.sum(valid) < 2:
            return None

        time = time[valid]
        lon = lon[valid]
        lat = lat[valid]
        lgd = lgd[valid]

        order = np.argsort(time)

        return {
            "time": time[order],
            "lon": lon[order],
            "lat": lat[order],
            "lgd": lgd[order],
        }

    def _interp_lgd_to_lat(
            self,
            lat,
            lgd,
            lat_grid):
        """
        Interpolate LGD to a common latitude grid.
        """

        lat = np.asarray(lat, dtype=np.float64)
        lgd = np.asarray(lgd, dtype=np.float64)

        valid = (
            np.isfinite(lat)
            &
            np.isfinite(lgd)
        )

        lat = lat[valid]
        lgd = lgd[valid]

        if len(lat) < 2:
            return np.full_like(
                lat_grid,
                np.nan,
                dtype=np.float64
            )

        order = np.argsort(lat)

        lat_sorted = lat[order]
        lgd_sorted = lgd[order]

        lat_unique, unique_idx = np.unique(
            lat_sorted,
            return_index=True
        )

        lgd_unique = lgd_sorted[unique_idx]

        if len(lat_unique) < 2:
            return np.full_like(
                lat_grid,
                np.nan,
                dtype=np.float64
            )

        return np.interp(
            lat_grid,
            lat_unique,
            lgd_unique,
            left=np.nan,
            right=np.nan
        )

    def _interp_lon_at_lat(
            self,
            lon,
            lat,
            lat_ref):
        """
        Interpolate longitude at a reference latitude.

        Longitude is unwrapped before interpolation to avoid
        discontinuity around 180/-180 degrees.
        """

        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)

        valid = (
            np.isfinite(lon)
            &
            np.isfinite(lat)
        )

        lon = self._lon_to_180(lon[valid])
        lat = lat[valid]

        if len(lon) < 2:
            return np.nan

        lat_min = np.nanmin(lat)
        lat_max = np.nanmax(lat)

        if not (lat_min <= lat_ref <= lat_max):
            return np.nan

        order = np.argsort(lat)

        lat_sorted = lat[order]
        lon_sorted = lon[order]

        lat_unique, unique_idx = np.unique(
            lat_sorted,
            return_index=True
        )

        lon_unique = lon_sorted[unique_idx]

        if len(lat_unique) < 2:
            return np.nan

        lon_rad = np.radians(lon_unique)
        lon_unwrapped = np.unwrap(lon_rad)

        lon_ref_rad = np.interp(
            lat_ref,
            lat_unique,
            lon_unwrapped
        )

        lon_ref = np.degrees(lon_ref_rad)

        return float(self._lon_to_180(lon_ref))

    def _lon_to_180(self, lon):
        """
        Convert longitude to [-180, 180).
        """

        lon = np.asarray(lon, dtype=np.float64)

        return ((lon + 180.0) % 360.0) - 180.0

    # ============================================================
    # Time conversion
    # ============================================================

    def gps_seconds_to_datetime(self, gps_seconds):
        """
        Convert PyHawk GPS seconds to UTC datetime.

        PyHawk time convention:
            GPS seconds past 2000-01-01 12:00:00

        This follows the same convention as Frame.setTime().
        """

        tai_mjd = (
            (
                float(gps_seconds)
                +
                Pre_Constants.Diff_Tai_GPS
            )
            /
            Pre_Constants.SecForOneDay
            +
            Pre_Constants.J2000_MJD
        )

        _, utc_mjd = sf.taiutc(
            Pre_Constants.DJMJD0,
            tai_mjd
        )

        return self._mjd_to_datetime(utc_mjd)

    def _mjd_to_datetime(self, mjd):
        """
        Convert Modified Julian Date to datetime.
        """

        mjd_epoch = datetime(
            1858,
            11,
            17,
            0,
            0,
            0
        )

        return mjd_epoch + timedelta(
            days=float(mjd)
        )

    def split_region_into_tracks(
            self,
            region,
            gap_factor=5.0,
            min_gap_seconds=60.0,
            min_points=20):
        """
        Split selected regional LGD data into individual ground-track segments.

        Why this is necessary:
            One PyHawk arc can contain multiple orbital passes. If we use arcNo
            directly as one track, discontinuous segments may be connected by
            artificial lines in the plot.

        Parameters
        ----------
        region : dict
            Output from select_region(), containing:
                time, lon, lat, lgd, arcNo

        gap_factor : float
            A time gap larger than gap_factor * nominal_dt will start a new track.

        min_gap_seconds : float
            Minimum time gap threshold in seconds.

        min_points : int
            Minimum number of samples for a valid track.

        Returns
        -------
        new_region : dict
            Same as region, but with an additional key:
                trackID
        """

        self._check_region(region)

        out = {
            "time": [],
            "lon": [],
            "lat": [],
            "lgd": [],
            "arcNo": [],
            "trackID": [],
        }

        track_id = 0

        for arc_no in np.unique(region["arcNo"]):

            idx_arc = region["arcNo"] == arc_no

            time = np.asarray(region["time"][idx_arc], dtype=np.float64)
            lon = np.asarray(region["lon"][idx_arc], dtype=np.float64)
            lat = np.asarray(region["lat"][idx_arc], dtype=np.float64)
            lgd = np.asarray(region["lgd"][idx_arc], dtype=np.float64)

            valid = (
                    np.isfinite(time)
                    &
                    np.isfinite(lon)
                    &
                    np.isfinite(lat)
                    &
                    np.isfinite(lgd)
            )

            time = time[valid]
            lon = lon[valid]
            lat = lat[valid]
            lgd = lgd[valid]

            if len(time) < min_points:
                continue

            order = np.argsort(time)

            time = time[order]
            lon = lon[order]
            lat = lat[order]
            lgd = lgd[order]

            dt = np.diff(time)

            if len(dt) == 0:
                continue

            dt_nominal = np.nanmedian(dt[dt > 0])

            gap_threshold = max(
                min_gap_seconds,
                gap_factor * dt_nominal
            )

            break_points = np.where(dt > gap_threshold)[0] + 1

            starts = np.insert(break_points, 0, 0)
            ends = np.append(break_points, len(time))

            for s, e in zip(starts, ends):

                if e - s < min_points:
                    continue

                out["time"].append(time[s:e])
                out["lon"].append(lon[s:e])
                out["lat"].append(lat[s:e])
                out["lgd"].append(lgd[s:e])
                out["arcNo"].append(
                    np.ones(e - s, dtype=np.int64) * int(arc_no)
                )
                out["trackID"].append(
                    np.ones(e - s, dtype=np.int64) * int(track_id)
                )

                track_id += 1

        if track_id == 0:
            raise RuntimeError(
                "No valid track segments found after splitting region."
            )

        new_region = {}

        for key in out:
            new_region[key] = np.concatenate(out[key])

        index = np.argsort(new_region["time"])

        for key in new_region:
            new_region[key] = new_region[key][index]

        print("Number of regional track segments:", track_id)

        return new_region

    def _extract_track(
            self,
            region,
            trackID,
            lat_lim=None,
            clean_lgd=False,
            lgd_mad_threshold=8.0,
            lgd_abs_limit=None,
            lgd_jump_limit=None):
        """
        Extract one regional track segment by trackID.

        Important:
            The returned samples are sorted by time, not latitude.
            This avoids artificial horizontal connection lines in LGD-lat plots.
        """

        if "trackID" not in region:
            raise KeyError(
                "region has no trackID. Please call split_region_into_tracks() first."
            )

        idx = region["trackID"] == trackID

        if np.sum(idx) == 0:
            return None

        time = np.asarray(region["time"][idx], dtype=np.float64)
        lon = np.asarray(region["lon"][idx], dtype=np.float64)
        lat = np.asarray(region["lat"][idx], dtype=np.float64)
        lgd = np.asarray(region["lgd"][idx], dtype=np.float64)

        valid = (
                np.isfinite(time)
                &
                np.isfinite(lon)
                &
                np.isfinite(lat)
                &
                np.isfinite(lgd)
        )

        if lat_lim is not None:
            valid = (
                    valid
                    &
                    (lat >= lat_lim[0])
                    &
                    (lat <= lat_lim[1])
            )

        time = time[valid]
        lon = lon[valid]
        lat = lat[valid]
        lgd = lgd[valid]

        if len(time) < 2:
            return None

        order = np.argsort(time)

        time = time[order]
        lon = lon[order]
        lat = lat[order]
        lgd = lgd[order]

        if clean_lgd:
            mask = self._lgd_clean_mask(
                lgd=lgd * 1e9,
                mad_threshold=lgd_mad_threshold,
                abs_limit=lgd_abs_limit,
                jump_limit=lgd_jump_limit
            )

            time = time[mask]
            lon = lon[mask]
            lat = lat[mask]
            lgd = lgd[mask]

            if len(time) < 2:
                return None

        return {
            "time": time,
            "lon": lon,
            "lat": lat,
            "lgd": lgd,
        }

    def _lgd_clean_mask(
            self,
            lgd,
            mad_threshold=8.0,
            abs_limit=None,
            jump_limit=None):
        """
        Build a robust mask for LGD plotting.

        Parameters
        ----------
        lgd : ndarray
            LGD in nm/s^2.

        mad_threshold : float
            Robust outlier threshold based on median absolute deviation.

        abs_limit : float or None
            If not None, remove points with |LGD| > abs_limit.

        jump_limit : float or None
            If not None, remove points involved in point-to-point jumps larger
            than jump_limit, unit nm/s^2.

        Notes
        -----
        This function is intended for visualization quality control.
        Do not silently use it for final geophysical estimates without recording it.
        """

        lgd = np.asarray(lgd, dtype=np.float64)

        mask = np.isfinite(lgd)

        if np.sum(mask) < 5:
            return mask

        x = lgd[mask]

        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))

        if mad > 0:
            robust_sigma = 1.4826 * mad
            local_mask = np.abs(x - med) <= mad_threshold * robust_sigma

            full_mask = np.zeros_like(mask, dtype=bool)
            full_mask[np.where(mask)[0]] = local_mask
            mask = mask & full_mask

        if abs_limit is not None:
            mask = mask & (np.abs(lgd) <= abs_limit)

        if jump_limit is not None and len(lgd) >= 3:
            jump_bad = np.zeros_like(mask, dtype=bool)

            diff = np.abs(np.diff(lgd))

            bad_jump = diff > jump_limit

            jump_bad[:-1] = jump_bad[:-1] | bad_jump
            jump_bad[1:] = jump_bad[1:] | bad_jump

            mask = mask & (~jump_bad)

        return mask

    def diagnose_pairs_lgd(
            self,
            region,
            pairs,
            lat_lim=(-65, 5)):
        """
        Print LGD statistics of each paired track.

        Unit:
            nm/s^2
        """

        if "trackID" not in region:
            raise KeyError(
                "region has no trackID. Please call split_region_into_tracks() first."
            )

        print("\nLGD diagnostics for paired tracks:")

        for pair in pairs:

            print(
                "\nPair:",
                pair["after_day"],
                "/",
                pair["before_day"],
                "direction:",
                pair["direction"]
            )

            for name, track_id in [
                ("after", pair["after_track"]),
                ("before", pair["before_track"])
            ]:

                track = self._extract_track(
                    region=region,
                    trackID=track_id,
                    lat_lim=lat_lim,
                    clean_lgd=False
                )

                if track is None:
                    print(name, "track", track_id, "None")
                    continue

                lgd = track["lgd"] * 1e9
                lat = track["lat"]

                print(
                    name,
                    "trackID:",
                    track_id,
                    "arc:",
                    pair[name + "_arc"],
                    "lat:",
                    round(np.nanmin(lat), 2),
                    round(np.nanmax(lat), 2),
                    "LGD min/median/max:",
                    round(np.nanmin(lgd), 3),
                    round(np.nanmedian(lgd), 3),
                    round(np.nanmax(lgd), 3),
                    "n:",
                    len(lgd)
                )


if __name__ == '__main__':

    analyzer = AnalyzeLGD(
        lgd_dir="../temp/LGD",
        date_span=["2021-03-06", "2021-03-23"]
    ).load_all_arcs()


    # 先用宽范围，避免轨迹被裁掉
    region = analyzer.select_region(
        lon_min=135,
        lon_max=165,
        lat_min=-65,
        lat_max=5
    )

    region = analyzer.split_region_into_tracks(
        region=region,
        gap_factor=5.0,
        min_gap_seconds=60.0,
        min_points=20
    )

    summaries = analyzer.summarize_tracks(
        region=region,
        lat_band=(-38, -22),
        lat_ref=-30,
        min_points=20
    )

    analyzer.diagnose_days(
        summaries=summaries,
        required_days=[6, 7, 11, 12, 17, 18, 22, 23]
    )

    pairs = analyzer.find_repeat_pairs(
        summaries=summaries,
        day_pairs=[
            (22, 11),
            (17, 6),
            (23, 12),
            (18, 7)
        ],
        lon_tolerance=1.5,
        direction="descending",
        same_direction=True,
        one_pair_per_day_pair=True,
        verbose=True
    )

    analyzer.print_pairs(pairs)

    analyzer.diagnose_pairs_lgd(
        region=region,
        pairs=pairs,
        lat_lim=(-65, 5)
    )

    analyzer.plot_repeat_tracks(
        region=region,
        pairs=pairs,
        lon_lim=(135, 165),
        lat_lim=(-65, 5),
        lat_band=(-38, -22),
        offset_step=4.5,
        clean_lgd=True,
        lgd_mad_threshold=6.0,
        lgd_abs_limit=8.0,
        lgd_jump_limit=3.0,
        save_fig="../result/img/LGD_repeat_tracks_descending_clean.png"
    )