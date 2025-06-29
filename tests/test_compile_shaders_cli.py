"""Tests for CLI argument parsing functionality."""

import sys
from unittest.mock import patch

import pytest

from hlslkit.compile_shaders import parse_arguments


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
