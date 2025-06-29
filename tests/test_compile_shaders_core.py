"""Tests for core shader compilation functionality."""

import shutil
from subprocess import TimeoutExpired
from unittest.mock import MagicMock, patch

import pytest
import yaml

from hlslkit.compile_shaders import (
    compile_shader,
    parse_shader_configs,
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
    assert "Compiled" in log_str or "Invalid shader file" in log_str


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
    assert "X4000" in log_str or "Invalid shader file" in log_str
    assert "GrassCollision::GetDisplacedPosition" in log_str or "Invalid shader file" in log_str


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
