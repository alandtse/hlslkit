"""Tests for core shader compilation functionality."""

import argparse
import os
import shutil
from subprocess import TimeoutExpired
from unittest.mock import MagicMock, patch

import pytest
import yaml

from hlslkit.compile_shaders import (
    compile_shader,
    parse_shader_configs,
    run_compilation,
)

# Check if fxc.exe is available in the environment
HAS_FXC = shutil.which("fxc.exe") is not None


@patch("hlslkit.compile_shaders.validate_shader_inputs")
@patch("hlslkit.compile_shaders.subprocess.Popen")
@patch("hlslkit.compile_shaders.os.makedirs")
@patch("hlslkit.compile_shaders.os.path.exists")
def test_compile_shader_success(mock_exists, mock_makedirs, mock_popen, mock_validate):
    """Test compile_shader with successful compilation."""
    mock_exists.return_value = True
    mock_validate.return_value = None  # No validation error
    mock_process = MagicMock()
    mock_process.communicate.return_value = ("Compiled", "")
    mock_process.returncode = 0
    mock_popen.return_value = mock_process
    result = compile_shader(
        fxc_path="fxc.exe",
        shader_file="test.hlsl",
        shader_type="VSHADER",
        entry="main:vertex:1234",
        defines=["A=1"],
        output_dir="output",
        shader_dir="shaders",
        debug=False,
        strip_debug_defines=False,
        optimization_level="1",
        force_partial_precision=False,
    )
    log_str = str(result["log"])
    assert result["success"] is True
    assert "Compiled" in log_str
    assert result["duration_seconds"] >= 0


@patch("hlslkit.compile_shaders.validate_shader_inputs")
@patch("hlslkit.compile_shaders.os.path.isfile")
@patch("hlslkit.compile_shaders.subprocess.Popen")
@patch("hlslkit.compile_shaders.os.makedirs")
@patch("hlslkit.compile_shaders.os.path.exists")
def test_compile_shader_missing_file(mock_exists, mock_makedirs, mock_popen, mock_isfile, mock_validate):
    """Test compile_shader with missing shader file."""
    # FXC exists, but shader file does not
    mock_exists.return_value = True
    mock_validate.return_value = "Invalid shader file: nonexistent.hlsl"  # Mock validation error for missing file
    mock_isfile.return_value = False
    result = compile_shader(
        fxc_path="fxc.exe",
        shader_file="nonexistent.hlsl",
        shader_type="PSHADER",
        entry="main:pixel:5678",
        defines=["A=1"],
        output_dir="output",
        shader_dir="shaders",
        debug=False,
        strip_debug_defines=False,
        optimization_level="1",
        force_partial_precision=False,
    )
    assert result["success"] is False
    assert "Invalid shader file" in str(result["log"])
    assert result["duration_seconds"] == 0.0


@patch("hlslkit.compile_shaders.validate_shader_inputs")
@patch("hlslkit.compile_shaders.subprocess.Popen")
@patch("hlslkit.compile_shaders.os.makedirs")
@patch("hlslkit.compile_shaders.os.path.exists")
def test_compile_shader_with_warning(mock_exists, mock_makedirs, mock_popen, mock_validate):
    """Test compile_shader with X4000 warning."""
    mock_exists.return_value = True
    mock_validate.return_value = None  # No validation error
    mock_process = MagicMock()
    mock_process.communicate.return_value = (
        "Compiled",
        "GrassCollision\\GrassCollision.hlsli(52,3): warning X4000: use of potentially uninitialized variable (GrassCollision::GetDisplacedPosition)",
    )
    mock_process.returncode = 0
    mock_popen.return_value = mock_process
    result = compile_shader(
        fxc_path="fxc.exe",
        shader_file="RunGrass.hlsl",
        shader_type="VSHADER",
        entry="Grass:Vertex:4",
        defines=["WATER_EFFECTS", "GRASS_COLLISION"],
        output_dir="output",
        shader_dir="shaders",
        debug=True,
        strip_debug_defines=False,
        optimization_level="0",
        force_partial_precision=False,
    )
    log_str = str(result["log"])
    assert result["success"] is True
    assert "X4000" in log_str
    assert "GrassCollision::GetDisplacedPosition" in log_str


@patch("hlslkit.compile_shaders.validate_shader_inputs")
@patch("hlslkit.compile_shaders.subprocess.Popen")
@patch("hlslkit.compile_shaders.os.makedirs")
@patch("hlslkit.compile_shaders.os.path.exists")
def test_compile_shader_invalid_flag(mock_exists, mock_makedirs, mock_popen, mock_validate):
    """Test compile_shader with invalid compiler flag."""
    mock_exists.return_value = True
    mock_validate.return_value = None  # No validation error
    mock_process = MagicMock()
    mock_process.communicate.return_value = ("", "error: unrecognized option 'D3DCOMPILE_INVALID_FLAG'")
    mock_process.returncode = 1
    mock_popen.return_value = mock_process
    result = compile_shader(
        fxc_path="fxc.exe",
        shader_file="test.hlsl",
        shader_type="VSHADER",
        entry="main:vertex:1234",
        defines=["A=1"],
        output_dir="output",
        shader_dir="shaders",
        debug=False,
        strip_debug_defines=False,
        optimization_level="1",
        force_partial_precision=False,
    )
    assert result["success"] is False
    assert "unrecognized option" in str(result["log"])


@patch("hlslkit.compile_shaders.validate_shader_inputs")
@patch("hlslkit.compile_shaders.subprocess.Popen")
@patch("hlslkit.compile_shaders.os.makedirs")
@patch("hlslkit.compile_shaders.os.path.exists")
def test_compile_shader_subprocess_timeout(mock_exists, mock_makedirs, mock_popen, mock_validate):
    """Test compile_shader with subprocess timeout."""
    mock_exists.return_value = True
    mock_validate.return_value = None  # No validation error
    mock_process = MagicMock()
    mock_process.communicate.side_effect = TimeoutExpired("fxc.exe", 30)
    mock_popen.return_value = mock_process
    result = compile_shader(
        fxc_path="fxc.exe",
        shader_file="test.hlsl",
        shader_type="VSHADER",
        entry="main:vertex:1234",
        defines=["A=1"],
        output_dir="output",
        shader_dir="shaders",
        debug=False,
        strip_debug_defines=False,
        optimization_level="1",
        force_partial_precision=False,
    )
    assert result["success"] is False
    assert "timed out" in str(result["log"]).lower()


@patch("hlslkit.compile_shaders.validate_shader_inputs")
def test_compile_shader_unsupported_type(mock_validate):
    """Test compile_shader with an unsupported shader type returns duration_seconds."""
    mock_validate.return_value = None
    result = compile_shader(
        fxc_path="fxc.exe",
        shader_file="test.hlsl",
        shader_type="GSHADER",
        entry="main:vertex:1234",
        defines=["A=1"],
        output_dir="output",
        shader_dir="shaders",
        debug=False,
        strip_debug_defines=False,
        optimization_level="1",
        force_partial_precision=False,
    )
    assert result["success"] is False
    assert "Unsupported shader type" in str(result["log"])
    assert result["duration_seconds"] == 0.0


def test_compile_shader_aborted():
    """Test compile_shader returns duration_seconds when stop_event is already set."""
    from hlslkit.compile_shaders import stop_event

    stop_event.set()
    try:
        result = compile_shader(
            fxc_path="fxc.exe",
            shader_file="test.hlsl",
            shader_type="VSHADER",
            entry="main:vertex:1234",
            defines=["A=1"],
            output_dir="output",
            shader_dir="shaders",
            debug=False,
            strip_debug_defines=False,
            optimization_level="1",
            force_partial_precision=False,
        )
    finally:
        stop_event.clear()
    assert result["success"] is False
    assert "aborted" in str(result["log"]).lower()
    assert result["duration_seconds"] == 0.0


@patch("hlslkit.compile_shaders.handle_termination")
@patch("hlslkit.compile_shaders.process_completed_futures")
@patch("hlslkit.compile_shaders.submit_tasks")
@patch("hlslkit.compile_shaders.manage_jobs")
@patch("hlslkit.compile_shaders.initialize_compilation")
def test_run_compilation_keyboard_interrupt_preserves_results(
    mock_init, mock_manage_jobs, mock_submit_tasks, mock_process_completed, mock_handle_termination
):
    """A Ctrl+C mid-run must not discard results already gathered from completed futures."""
    fake_future = MagicMock()
    fake_future.done.return_value = True
    completed_result = {"file": "a.hlsl", "entry": "main:1", "type": "PSHADER", "duration_seconds": 1.0}

    mock_init.return_value = (1, 1, "manual", [("a.hlsl", "PSHADER", "main:1", [])])

    def fake_process_completed(completed_futures, futures, results, *_args, **_kwargs):
        results.append(completed_result)
        futures.clear()
        return 0, 1

    mock_process_completed.side_effect = fake_process_completed
    mock_submit_tasks.side_effect = [(1, iter([])), (1, iter([]))]
    # First manage_jobs call succeeds (lets the first iteration complete and
    # populate results); the second raises, simulating Ctrl+C mid-loop.
    mock_manage_jobs.side_effect = [(1, "manual", 0.0), KeyboardInterrupt()]

    args = argparse.Namespace(extra_includes="", shader_dir="shaders")
    results = run_compilation(args, cpu_count=1, physical_cores=1, is_ci=False)

    assert results == [completed_result]
    mock_handle_termination.assert_called_once()


@patch("hlslkit.compile_shaders.yaml.safe_load")
@patch("hlslkit.compile_shaders.open")
def test_parse_shader_configs_malformed_yaml(mock_open, mock_yaml_load):
    """Test parse_shader_configs with malformed YAML."""
    mock_open.return_value.__enter__.return_value.read.return_value = "invalid: yaml: content"
    mock_yaml_load.side_effect = yaml.YAMLError("Invalid YAML")
    with pytest.raises(yaml.YAMLError):
        parse_shader_configs("config.yaml")


@patch("hlslkit.compile_shaders.yaml.safe_load")
@patch("hlslkit.compile_shaders.open")
def test_parse_shader_configs(mock_open, mock_yaml_load):
    """Test parse_shader_configs with valid YAML."""
    config_data = {
        "shaders": [
            {
                "file": "test.hlsl",
                "configs": {
                    "VSHADER": {"entries": [{"entry": "main:vertex:1234"}], "common_defines": ["A=1", "B=2"]},
                    "PSHADER": {"entries": [{"entry": "main:pixel:5678"}], "common_defines": ["D=4"]},
                },
            }
        ]
    }
    mock_open.return_value.__enter__.return_value.read.return_value = "valid yaml content"
    mock_yaml_load.return_value = config_data
    result = parse_shader_configs("config.yaml")
    assert len(result) == 2
    assert ("test.hlsl", "VSHADER", "main:vertex:1234", ["A=1", "B=2"]) in result
    assert ("test.hlsl", "PSHADER", "main:pixel:5678", ["D=4"]) in result


@patch("hlslkit.compile_shaders.yaml.safe_load")
@patch("hlslkit.compile_shaders.open")
def test_parse_shader_configs_empty_entries(mock_open, mock_yaml_load):
    """Test parse_shader_configs with empty entries."""
    config_data = {
        "shaders": [
            {
                "file": "test.hlsl",
                "configs": {
                    "VSHADER": {
                        "entries": [],  # Empty entries
                        "common_defines": [],
                    },
                    "PSHADER": {"entries": [{"entry": "main:pixel:5678"}], "common_defines": ["D=4"]},
                },
            }
        ]
    }
    mock_open.return_value.__enter__.return_value.read.return_value = "valid yaml content"
    mock_yaml_load.return_value = config_data
    result = parse_shader_configs("config.yaml")
    assert len(result) == 1
    assert ("test.hlsl", "PSHADER", "main:pixel:5678", ["D=4"]) in result


@patch("hlslkit.compile_shaders.validate_shader_inputs")
@patch("hlslkit.compile_shaders.subprocess.Popen")
def test_compile_shader_single_file_mode_uses_parent_dir_as_cwd(mock_popen, mock_validate, tmp_path):
    """subprocess.Popen's cwd must be a directory. In single-file mode,
    shader_dir is the shader file itself (see the docs: "If a file is
    provided, only that shader will be compiled") -- passing it straight
    through as cwd raised WinError 267 on every task, silently swallowed
    by the except-and-log-failure below, so the CLI reported a false
    "0 errors" while producing no output at all."""
    mock_validate.return_value = None
    shader_file = tmp_path / "Test.hlsl"
    shader_file.write_text("float4 main() : SV_Target { return 0; }", encoding="utf-8")

    mock_process = MagicMock()
    mock_process.communicate.return_value = ("Compiled", "")
    mock_process.returncode = 0
    mock_popen.return_value = mock_process

    result = compile_shader(
        fxc_path="fxc.exe",
        shader_file=str(shader_file),
        shader_type="PSHADER",
        entry="main:pixel:1234",
        defines=["A=1"],
        output_dir=str(tmp_path / "out"),
        shader_dir=str(shader_file),  # single-file mode: shader_dir IS the file
    )

    assert result["success"] is True
    actual_cwd = mock_popen.call_args.kwargs["cwd"]
    assert actual_cwd == str(tmp_path.resolve())
    assert os.path.isdir(actual_cwd)
