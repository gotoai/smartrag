"""The `kb sync` progress bar (smartrag.cli.progress)."""

from __future__ import annotations

import io

import pytest

from smartrag.cli.progress import ProgressBar, clock, render


def test_render_matches_the_documented_line():
    line = render(1000, 330000, 15 * 60 + 39, 23 * 3600 + 45 * 60 + 41, columns=200)
    assert line.endswith("   0.3%  1000 / 330000  00:15:39 / 23:45:41")
    assert line.startswith("[") and line.count("-") + line.count("#") >= 40


def test_render_fills_the_bar_and_fits_the_terminal():
    full = render(10, 10, 5, 0, columns=80)
    assert "100.0%" in full and "-" not in full.split("]")[0]
    assert len(render(5, 10, 5, None, columns=60)) <= 60
    assert render(0, 10, 0, None, columns=80).endswith("00:00:00 / --:--:--")


def test_clock_hours_grow_past_a_day():
    assert clock(100 * 3600 + 61) == "100:01:01"


def _bar(total: int, samples: list[tuple[float, int]]) -> ProgressBar:
    bar = ProgressBar(stream=io.StringIO())
    bar.total, bar.started = total, 0.0
    bar._samples.extend(samples)
    bar.done = samples[-1][1]
    return bar


def test_no_estimate_before_30_seconds_or_10_files():
    assert _bar(1000, [(0.0, 0), (20.0, 500)]).remaining(20.0) is None
    assert _bar(1000, [(0.0, 0), (40.0, 9)]).remaining(40.0) is None
    assert _bar(1000, [(0.0, 0), (40.0, 10)]).remaining(40.0) == 990 * 4


def test_estimate_uses_the_last_30_seconds():
    # a resumed sync: 50 unchanged files skipped in a second, then a file every 3 s
    bar = _bar(100, [(0.0, 0), (1.0, 50), (40.0, 63), (70.0, 73)])
    assert bar.remaining(70.0) == 27 * 3


def test_slow_files_widen_the_window_to_10_files():
    bar = _bar(100, [(0.0, 0), (30.0, 3), (60.0, 6), (90.0, 9), (120.0, 12)])
    assert bar.remaining(120.0) == 88 * 10  # 12 files in 120 s


def test_estimate_counts_down_between_updates_and_ends_at_zero():
    bar = _bar(1000, [(0.0, 0), (40.0, 10)])
    assert bar.remaining(40.0) == 3960
    bar.done = 500  # a much faster stretch does not move the estimate before 30 s pass
    assert bar.remaining(50.0) == 3950
    bar._samples.append((70.0, 500))
    assert bar.remaining(70.0) == pytest.approx(500 * 30 / 490)  # recomputed at 70 s: 490 files in 30 s
    bar.done = 1000
    assert bar.remaining(71.0) == 0.0


def test_bar_draws_on_plan_and_ends_its_line_on_close():
    out = io.StringIO()
    with ProgressBar(interval=60, stream=out) as bar:
        bar({"event": "plan", "files": 2})
        bar({"event": "document", "status": "added"})
        bar({"event": "document", "status": "unchanged"})
    text = out.getvalue()
    assert text.startswith("\r[") and text.endswith("\n")
    assert "100.0%  2 / 2" in text.splitlines()[-1]


def test_no_plan_no_bar():
    out = io.StringIO()
    with ProgressBar(stream=out):
        pass
    assert out.getvalue() == ""
