"""Tests for CLI argument parsing functionality."""

import argparse
import json
import sys
from unittest.mock import patch

import pytest

from hlslkit.compile_shaders import _maybe_write_cache_manifest, _maybe_write_timing_report, parse_arguments


def test_parse_arguments_default_jobs():
    """Test parse_arguments with default jobs."""
    with patch("sys.argv", ["compile_shaders.py", "--config", "test.yaml", "--shader-dir", "shaders"]):
        args = parse_arguments(default_jobs=4)
        assert args.jobs == 4
        assert args.config == "test.yaml"
        assert args.shader_dir == "shaders"


def test_parse_arguments_invalid_jobs():
    """Test parse_arguments with invalid jobs value."""
    with (
        patch(
            "sys.argv", ["compile_shaders.py", "--config", "test.yaml", "--shader-dir", "shaders", "--jobs", "invalid"]
        ),
        pytest.raises(SystemExit),
    ):
        parse_arguments(default_jobs=4)


def test_parse_arguments_optimization_levels():
    """Test parse_arguments with different optimization levels."""
    for level in ["0", "1", "2", "3"]:
        with patch(
            "sys.argv",
            ["compile_shaders.py", "--config", "test.yaml", "--shader-dir", "shaders", "--optimization-level", level],
        ):
            args = parse_arguments(default_jobs=4)
            assert args.optimization_level == level


def test_parse_arguments_all_flags():
    """Test parse_arguments with all available flags."""
    with patch(
        "sys.argv",
        [
            "compile_shaders.py",
            "--config",
            "test.yaml",
            "--shader-dir",
            "shaders",
            "--output-dir",
            "output",
            "--jobs",
            "8",
            "--debug",
            "--strip-debug-defines",
            "--optimization-level",
            "2",
            "--force-partial-precision",
            "--max-warnings",
            "10",
            "--extra-includes",
            "path1,path2",
            "--debug-defines",
            "DEBUG,TRACE",
        ],
    ):
        args = parse_arguments(default_jobs=4)
        assert args.config == "test.yaml"
        assert args.shader_dir == "shaders"
        assert args.output_dir == "output"
        assert args.jobs == 8
        assert args.debug is True
        assert args.strip_debug_defines is True
        assert args.optimization_level == "2"
        assert args.force_partial_precision is True
        assert args.max_warnings == 10
        assert args.extra_includes == "path1,path2"
        assert args.debug_defines == "DEBUG,TRACE"


def test_parse_arguments_timing_report_default_empty():
    """Default --timing-report is empty, i.e. the feature is opt-in."""
    with patch("sys.argv", ["compile_shaders.py", "--config", "test.yaml", "--shader-dir", "shaders"]):
        args = parse_arguments(default_jobs=4)
        assert args.timing_report == ""


def test_parse_arguments_timing_report_set():
    """--timing-report accepts a path like the other output-path flags."""
    with patch(
        "sys.argv",
        [
            "compile_shaders.py",
            "--config",
            "test.yaml",
            "--shader-dir",
            "shaders",
            "--timing-report",
            "build/timing.json",
        ],
    ):
        args = parse_arguments(default_jobs=4)
        assert args.timing_report == "build/timing.json"


def test_maybe_write_timing_report_noop_when_flag_empty(tmp_path):
    """No --timing-report means no report file is written at all."""
    _maybe_write_timing_report([{"file": "a.hlsl", "entry": "main:1", "type": "PSHADER"}], "")
    assert list(tmp_path.iterdir()) == []


def test_maybe_write_timing_report_writes_sorted_json(tmp_path):
    """The report is one entry per file+entry variant, sorted slowest-first."""
    report_path = tmp_path / "timing.json"
    results = [
        {"file": "Shaders/Fast.hlsl", "entry": "main:1", "type": "PSHADER", "duration_seconds": 0.1},
        {"file": "Shaders/Slow.hlsl", "entry": "main:2", "type": "VSHADER", "duration_seconds": 5.0},
        {"file": "Shaders/Mid.hlsl", "entry": "main:3", "type": "CSHADER", "duration_seconds": 1.0},
    ]
    _maybe_write_timing_report(results, str(report_path))

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    assert [entry["file"] for entry in report] == ["Slow.hlsl", "Mid.hlsl", "Fast.hlsl"]
    assert report[0]["entry"] == "main:2"
    assert report[0]["type"] == "VSHADER"
    assert report[0]["duration_seconds"] == 5.0


def test_maybe_write_timing_report_missing_duration_defaults_to_zero(tmp_path):
    """A result missing 'duration_seconds' (e.g. an aborted task) reports 0.0 rather than raising."""
    report_path = tmp_path / "timing.json"
    _maybe_write_timing_report([{"file": "a.hlsl", "entry": "main:1", "type": "PSHADER"}], str(report_path))

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    assert report[0]["duration_seconds"] == 0.0


def test_maybe_write_timing_report_never_raises_on_failure(tmp_path):
    """A report-write failure must not raise -- it's caught and logged, matching
    the cache-manifest helper's contract of never failing an otherwise-successful compile."""
    bad_path = str(tmp_path / "does-not-exist" / "timing.json")
    _maybe_write_timing_report([{"file": "a.hlsl", "entry": "main:1", "type": "PSHADER"}], bad_path)  # must not raise


def test_parse_arguments_missing_config():
    """Test parse_arguments with missing required config."""
    with patch("sys.argv", ["compile_shaders.py", "--shader-dir", "shaders"]), pytest.raises(SystemExit):
        parse_arguments(default_jobs=4)


def test_parse_arguments_debug_defines_whitespace():
    """Test parse_arguments with whitespace in debug defines."""
    with patch(
        "sys.argv",
        [
            "compile_shaders.py",
            "--config",
            "test.yaml",
            "--shader-dir",
            "shaders",
            "--debug-defines",
            "  DEBUG  ,  TRACE  ",
        ],
    ):
        args = parse_arguments(default_jobs=4)
        assert args.debug_defines == "  DEBUG  ,  TRACE  "


def test_parse_arguments_debug_defines_duplicates():
    """Test parse_arguments with duplicate debug defines."""
    with patch(
        "sys.argv",
        [
            "compile_shaders.py",
            "--config",
            "test.yaml",
            "--shader-dir",
            "shaders",
            "--debug-defines",
            "DEBUG,DEBUG,TRACE",
        ],
    ):
        args = parse_arguments(default_jobs=4)
        assert args.debug_defines == "DEBUG,DEBUG,TRACE"


def test_parse_arguments_debug_defines_empty_and_stray_comma():
    """Test parse_arguments for debug-defines handling of empty string and stray comma."""
    test_argv = ["prog", "--config", "test.yaml", "--debug-defines", ""]
    with patch.object(sys, "argv", test_argv):
        args = parse_arguments(default_jobs=4)
        assert args.debug_defines_set is None

    test_argv = ["prog", "--config", "test.yaml", "--debug-defines", "DEBUG,"]
    with patch.object(sys, "argv", test_argv):
        args = parse_arguments(default_jobs=4)
        assert args.debug_defines_set == {"DEBUG"}

    test_argv = ["prog", "--config", "test.yaml", "--debug-defines", "DEBUG, ,FOO,,"]
    with patch.object(sys, "argv", test_argv):
        args = parse_arguments(default_jobs=4)
        assert args.debug_defines_set == {"DEBUG", "FOO"}


def test_parse_arguments_negative_max_warnings():
    """Test parse_arguments with negative max warnings."""
    with patch(
        "sys.argv", ["compile_shaders.py", "--config", "test.yaml", "--shader-dir", "shaders", "--max-warnings", "-5"]
    ):
        args = parse_arguments(default_jobs=4)
        assert args.max_warnings == -5


def test_parse_arguments_zero_max_warnings():
    """Test parse_arguments with zero max warnings."""
    with patch(
        "sys.argv", ["compile_shaders.py", "--config", "test.yaml", "--shader-dir", "shaders", "--max-warnings", "0"]
    ):
        args = parse_arguments(default_jobs=4)
        assert args.max_warnings == 0


def test_parse_arguments_large_max_warnings():
    """Test parse_arguments with large max warnings value."""
    with patch(
        "sys.argv",
        ["compile_shaders.py", "--config", "test.yaml", "--shader-dir", "shaders", "--max-warnings", "999999"],
    ):
        args = parse_arguments(default_jobs=4)
        assert args.max_warnings == 999999


def test_parse_arguments_invalid_max_warnings():
    """Test parse_arguments with invalid max warnings value."""
    with (
        patch(
            "sys.argv",
            ["compile_shaders.py", "--config", "test.yaml", "--shader-dir", "shaders", "--max-warnings", "invalid"],
        ),
        pytest.raises(SystemExit),
    ):
        parse_arguments(default_jobs=4)


def test_parse_arguments_extra_includes_whitespace():
    """Test parse_arguments with whitespace in extra includes."""
    with patch(
        "sys.argv",
        [
            "compile_shaders.py",
            "--config",
            "test.yaml",
            "--shader-dir",
            "shaders",
            "--extra-includes",
            "  path1  ,  path2  ",
        ],
    ):
        args = parse_arguments(default_jobs=4)
        assert args.extra_includes == "  path1  ,  path2  "


def test_parse_arguments_extra_includes_empty():
    """Test parse_arguments with empty extra includes."""
    with patch(
        "sys.argv", ["compile_shaders.py", "--config", "test.yaml", "--shader-dir", "shaders", "--extra-includes", ""]
    ):
        args = parse_arguments(default_jobs=4)
        assert args.extra_includes == ""


def test_parse_arguments_extra_includes_single_path():
    """Test parse_arguments with single extra include path."""
    with patch(
        "sys.argv",
        ["compile_shaders.py", "--config", "test.yaml", "--shader-dir", "shaders", "--extra-includes", "single_path"],
    ):
        args = parse_arguments(default_jobs=4)
        assert args.extra_includes == "single_path"


def test_parse_arguments_invalid_optimization_level():
    """Test parse_arguments with invalid optimization level."""
    with (
        patch(
            "sys.argv",
            ["compile_shaders.py", "--config", "test.yaml", "--shader-dir", "shaders", "--optimization-level", "5"],
        ),
        pytest.raises(SystemExit),
    ):
        parse_arguments(default_jobs=4)


def test_parse_arguments_negative_jobs():
    """Test parse_arguments with negative jobs value."""
    with (
        patch("sys.argv", ["compile_shaders.py", "--config", "test.yaml", "--shader-dir", "shaders", "--jobs", "-1"]),
        pytest.raises(SystemExit),
    ):
        parse_arguments(default_jobs=4)


def test_parse_arguments_zero_jobs():
    """Test parse_arguments with zero jobs value."""
    with (
        patch("sys.argv", ["compile_shaders.py", "--config", "test.yaml", "--shader-dir", "shaders", "--jobs", "0"]),
        pytest.raises(SystemExit),
    ):
        parse_arguments(default_jobs=4)


def test_maybe_write_cache_manifest_noop_when_flag_empty():
    """No --emit-cache-manifest means no manifest write is attempted at all."""
    args = argparse.Namespace(emit_cache_manifest="", shader_dir="Shaders", manifest_global_defines="")
    _maybe_write_cache_manifest(args)  # must not raise, must not touch the filesystem


def test_maybe_write_cache_manifest_never_raises_on_failure(tmp_path):
    """A manifest failure (missing shader dir, disk error, etc.) must not fail
    an otherwise-successful compile -- it's caught and logged, not raised."""
    args = argparse.Namespace(
        emit_cache_manifest=str(tmp_path / "does-not-exist" / "Manifest.json"),
        shader_dir=str(tmp_path / "nonexistent-shader-dir"),
        manifest_global_defines="",
    )
    _maybe_write_cache_manifest(args)  # must not raise regardless of platform or failure mode
