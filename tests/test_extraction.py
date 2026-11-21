"""Tests for the scaled-extraction plan and runner (M1-T06). No network access."""

from __future__ import annotations

from pathlib import Path

import pytest

from desertification.config import load_config
from desertification.extraction import (
    LOG_COLUMNS,
    append_log,
    build_plan,
    expected_filename,
    extraction_window,
    low_cloud_factor,
    month_windows,
    plan_status,
    run_plan,
    season_windows,
)

AOI = {
    "aoi_id": "x",
    "bbox_utm": [500000.0, 1700000.0, 520000.0, 1720000.0],
    "utm_epsg": 32628,
    "wet_season_months": [12, 1, 2, 3],
    "dry_season_months": [6, 7, 8, 9],
}
PERIOD = {"start": "2019-01-01", "end": "2025-12-31"}


def ext(**over):
    cfg = load_config()["extraction"]
    return {**cfg, **over}


# --- windows ---------------------------------------------------------------------------


def test_extraction_window_centred_and_offset():
    assert extraction_window(AOI["bbox_utm"], 10000, 100) == [
        505000.0,
        1705000.0,
        515000.0,
        1715000.0,
    ]
    assert extraction_window(AOI["bbox_utm"], 10000, 100, (-3000, 2000)) == [
        502000.0,
        1707000.0,
        512000.0,
        1717000.0,
    ]
    with pytest.raises(ValueError):
        extraction_window(AOI["bbox_utm"], 10000, 100, (6000, 0))  # would leave the AOI


def test_season_windows_label_by_start_year():
    wet = season_windows([12, 1, 2, 3], PERIOD["start"], PERIOD["end"])
    assert wet[0] == (2019, "2019-12-01", "2020-03-31") and len(wet) == 6
    dry = season_windows([6, 7, 8, 9], PERIOD["start"], PERIOD["end"])
    assert dry[0] == (2019, "2019-06-01", "2019-09-30") and len(dry) == 7


def test_month_windows():
    months = month_windows(PERIOD["start"], PERIOD["end"])
    assert len(months) == 84
    assert months[1] == (2019, 2, "2019-02-01", "2019-02-28")
    assert (2024, 2, "2024-02-01", "2024-02-29") in months


def test_low_cloud_factor_capped():
    assert low_cloud_factor(0.5, 1.3) == pytest.approx(0.65)
    assert low_cloud_factor(0.9, 1.3) == 1.0


# --- plan ------------------------------------------------------------------------------


def test_build_plan_counts_ids_and_costs():
    plan = build_plan([AOI], ext(), PERIOD, 100, {"x": 120.0}, {"x": 0.5}, calib=1.0, lc_adjust=1.0)
    assert (plan["product"] == "seasonal").sum() == 13 and (
        plan["product"] == "monthly"
    ).sum() == 84
    assert plan["request_id"].is_unique
    seasonal = plan[plan["product"] == "seasonal"].iloc[0]
    assert seasonal["label"] == "s10b_lc40_w10km" and len(seasonal["bands"]) == 10
    assert plan[plan["product"] == "monthly"].iloc[0]["label"] == "m5b_lc40_w10km"
    # 500 x 500 px, 11 input bands, 120/yr x 4/12 = 40 samples, low-cloud factor 0.5
    assert seasonal["projected_pu"] == pytest.approx(500 * 500 / 512**2 * 11 / 3 * 40 * 0.5)
    names = {expected_filename(r, 20) for r in plan.to_dict(orient="records")}
    assert len(names) == len(plan)  # no two requests write the same file


def test_build_plan_without_cloud_filter():
    plan = build_plan([AOI], ext(max_cloud_cover=None), PERIOD, 100, {"x": 120.0}, {"x": 0.5})
    assert plan.iloc[0]["label"] == "s10b_w10km"
    assert plan.iloc[0]["projected_pu"] == pytest.approx(500 * 500 / 512**2 * 11 / 3 * 40)


# --- runner ----------------------------------------------------------------------------


class FakeDownloads:
    """Simulates download_composite: a set of cached ids, optional failures."""

    def __init__(self, cached=(), fail_times=None, pu=1.0):
        self.cached = set(cached)
        self.fail_times = dict(fail_times or {})
        self.api_calls = []
        self.pu = pu

    def __call__(self, row, allow_api):
        rid = row["request_id"]
        if rid in self.cached:
            return Path(f"/cache/{rid}.tif"), False, self.pu
        if not allow_api:
            return None, False, None
        self.api_calls.append(rid)
        if self.fail_times.get(rid, 0) > 0:
            self.fail_times[rid] -= 1
            raise OSError("HTTP 503")
        self.cached.add(rid)
        return Path(f"/cache/{rid}.tif"), True, self.pu


@pytest.fixture
def plan():
    return build_plan([AOI], ext(), PERIOD, 100, {"x": 120.0}, {"x": 0.5})


def test_run_plan_without_permission_makes_no_calls(plan):
    dl = FakeDownloads(cached=["x_dry_2019"])
    report = run_plan(plan, dl, allow_api=False, max_requests=None, pu_budget=None)
    assert dl.api_calls == [] and report.cached == ["x_dry_2019"]
    assert len(report.not_run) == len(plan) - 1


def test_run_plan_request_limit_then_resume(plan):
    dl = FakeDownloads()
    first = run_plan(
        plan, dl, allow_api=True, max_requests=10, pu_budget=None, sleep=lambda s: None
    )
    assert len(first.downloaded) == 10 and "request limit" in first.stopped_reason
    second = run_plan(
        plan, dl, allow_api=True, max_requests=None, pu_budget=None, sleep=lambda s: None
    )
    assert len(second.cached) == 10 and len(second.downloaded) == len(plan) - 10
    third = run_plan(plan, dl, allow_api=True, max_requests=None, pu_budget=None)
    assert third.api_requests == 0 and len(third.cached) == len(plan)
    assert len(dl.api_calls) == len(plan)  # every request downloaded exactly once


def test_run_plan_budget_guard(plan):
    dl = FakeDownloads(pu=None)  # no header: the runner counts projected PU
    budget = plan["projected_pu"].iloc[:3].sum() + 0.01
    report = run_plan(
        plan, dl, allow_api=True, max_requests=None, pu_budget=budget, sleep=lambda s: None
    )
    assert len(report.downloaded) == 3 and "PU budget" in report.stopped_reason


def test_run_plan_retries_with_backoff_then_reports_failure(plan):
    waits = []
    dl = FakeDownloads(fail_times={"x_wet_2019": 1, "x_dry_2019": 5})
    rows = plan[plan["request_id"].isin(["x_wet_2019", "x_dry_2019"])]
    logged = []
    report = run_plan(
        rows,
        dl,
        allow_api=True,
        max_requests=None,
        pu_budget=None,
        attempts=3,
        backoff_s=10,
        sleep=waits.append,
        log=logged.append,
    )
    assert report.downloaded == ["x_wet_2019"]  # succeeded on the 2nd attempt
    assert set(report.failed) == {"x_dry_2019"} and "503" in report.failed["x_dry_2019"]
    assert waits == [10, 10, 20]
    assert {e["status"] for e in logged} == {"downloaded", "failed"}


def test_append_log_and_plan_status(tmp_path, plan):
    path = tmp_path / "download_log.csv"
    entry = dict.fromkeys(LOG_COLUMNS, "")
    entry.update(request_id="a", status="cached")
    append_log(path, entry)
    append_log(path, entry)
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(LOG_COLUMNS) and len(lines) == 3
    status = plan_status(plan, present={"x_wet_2019"}, failed={"x_dry_2019": "HTTP 503"})
    assert status.set_index("request_id").loc["x_dry_2019", "reason"] == "HTTP 503"
    assert (status["status"] == "missing").sum() == len(plan) - 2
