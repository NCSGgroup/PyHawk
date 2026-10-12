import datetime as dt
import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "plot_kbrr_hawk_csr_postfit_sixpanel.py"
)
SPEC = importlib.util.spec_from_file_location("kbrr_spectrum", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_csr_epoch_conversion_tracks_leap_seconds():
    assert MODULE.csr_time_in_hawk_epoch(43186, dt.date(2006, 1, 2)) == 0
    assert MODULE.csr_time_in_hawk_epoch(43185, dt.date(2010, 1, 1)) == 0


def test_matched_segments_and_band_rms_use_one_grid():
    day = dt.date(2010, 1, 1)
    count = 2300
    hawk_time = np.arange(count, dtype=float) * 5.0
    csr_time = hawk_time + 43200.0 - MODULE.gps_utc_offset(day)
    frequency = 0.004
    hawk_signal = 1.0e-6 * np.sin(2 * np.pi * frequency * hawk_time)
    csr_signal = 0.8e-6 * np.sin(2 * np.pi * frequency * hawk_time)
    coordinates = np.zeros(count)
    products = {
        "hawk": {
            day: np.column_stack(
                (hawk_time, coordinates, coordinates, hawk_signal)
            )
        },
        "csr": {
            day: np.column_stack(
                (csr_time, coordinates, coordinates, csr_signal)
            )
        },
    }

    segments, summary = MODULE.matched_continuous_segments(products, [day])
    assert summary["retained_segment_count"] == 1
    assert summary["matched_sample_count"] == count
    assert np.array_equal(
        segments[0]["hawk_time"], segments[0]["csr_time"]
    )
    assert np.all(np.diff(segments[0]["hawk_time"]) == 5.0)

    grid, hawk_spectrum, csr_spectrum = MODULE.paired_asd(segments)
    hawk_rms = MODULE.band_rms(grid, hawk_spectrum["median_psd"])
    csr_rms = MODULE.band_rms(grid, csr_spectrum["median_psd"])
    assert grid.shape == hawk_spectrum["median_asd"].shape
    assert grid.shape == csr_spectrum["median_asd"].shape
    assert np.isclose(hawk_rms / csr_rms, 1.25, rtol=0.02)
