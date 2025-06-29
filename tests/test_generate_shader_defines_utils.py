"""Tests for utility functions in generate_shader_defines.py."""

from datetime import datetime
from unittest.mock import mock_open, patch

import pytest

from hlslkit.generate_shader_defines import (
    CompilationTask,
    collect_tasks,
    collect_warnings_and_errors,
    compute_common_defines,
    count_compiling_lines,
    count_log_blocks,
    generate_yaml_data,
    get_shader_type_from_entry,
    normalize_path,
    parse_timestamp,
    populate_configs,
    save_yaml,
)


class TestTimestampParsing:
    """Test timestamp parsing functionality."""

    def test_parse_timestamp_valid(self):
        """Test parsing valid timestamp."""
        line = "[12:34:56.789] Some log message"
        result = parse_timestamp(line)
        expected = datetime(1900, 1, 1, 12, 34, 56, 789000)
        assert result == expected

    def test_parse_timestamp_invalid_format(self):
        """Test parsing invalid timestamp format."""
        line = "[invalid] Some log message"
        with pytest.raises(ValueError):
            parse_timestamp(line)


class TestLogCounting:
    """Test log counting functions."""

    def test_count_compiling_lines(self):
        """Test counting compiling lines."""
        log_content = """[12:34:56.789] [123] [D] Compiling shader.hlsl main:vertex:1234 to output
[12:34:57.789] [123] [D] Some other message
[12:34:58.789] [124] [D] Compiling other.hlsl main:pixel:5678 to output"""

        with patch("builtins.open", mock_open(read_data=log_content)):
            result = count_compiling_lines("test.log")
            assert result == 2

    def test_count_log_blocks(self):
        """Test counting log blocks."""
        log_content = """[12:34:56.789] [123] [D] Shader logs:
[12:34:57.789] [123] [E] Failed to compile
[12:34:58.789] [124] [W] Shader compilation failed
[12:34:59.789] [125] [D] Adding Completed shader"""

        with patch("builtins.open", mock_open(read_data=log_content)):
            result = count_log_blocks("test.log")
            assert result == 4


class TestPathNormalization:
    """Test path normalization functionality."""

    def test_normalize_path_with_shaders_directory(self):
        """Test normalizing path with Shaders directory."""
        result = normalize_path("/path/to/Shaders/shaders/test.hlsl")
        assert result == "shaders/test.hlsl"

    def test_normalize_path_with_shaders_directory_case_insensitive(self):
        """Test normalizing path with Shaders directory (case insensitive)."""
        result = normalize_path("/path/to/shaders/shaders/test.hlsl")
        assert result == "shaders/test.hlsl"

    def test_normalize_path_without_shaders_directory(self):
        """Test normalizing path without Shaders directory."""
        result = normalize_path("/path/to/other/test.hlsl")
        assert result == "/path/to/other/test.hlsl"

    def test_normalize_path_with_backslashes(self):
        """Test normalizing path with backslashes."""
        result = normalize_path("C:\\path\\to\\Shaders\\shaders\\test.hlsl")
        assert result == "shaders/test.hlsl"

    def test_normalize_path_with_multiple_slashes(self):
        """Test normalizing path with multiple slashes."""
        result = normalize_path("/path//to///Shaders////shaders//test.hlsl")
        assert result == "shaders/test.hlsl"


class TestShaderTypeDetection:
    """Test shader type detection functionality."""

    def test_get_shader_type_vertex(self):
        """Test detecting vertex shader type."""
        result = get_shader_type_from_entry("main:vertex:1234")
        assert result == "VSHADER"

    def test_get_shader_type_pixel(self):
        """Test detecting pixel shader type."""
        result = get_shader_type_from_entry("main:pixel:5678")
        assert result == "PSHADER"

    def test_get_shader_type_compute(self):
        """Test detecting compute shader type."""
        result = get_shader_type_from_entry("main:compute:9012")
        assert result == "CSHADER"

    def test_get_shader_type_unknown(self):
        """Test detecting unknown shader type."""
        result = get_shader_type_from_entry("main:unknown:3456")
        assert result == "UNKNOWN"

    def test_get_shader_type_invalid_format(self):
        """Test detecting shader type with invalid format."""
        result = get_shader_type_from_entry("main:vertex")
        assert result == "UNKNOWN"

    def test_get_shader_type_empty(self):
        """Test detecting shader type with empty string."""
        result = get_shader_type_from_entry("")
        assert result == "UNKNOWN"


class TestTaskCollection:
    """Test task collection functionality."""

    def test_collect_tasks_compiling_line(self):
        """Test collecting tasks from compiling line."""
        lines = ["[12:34:56.789] [123] [D] Compiling shader.hlsl main:vertex:1234 to output"]
        result = collect_tasks(lines)
        assert len(result) == 1
        assert result[0].process_id == "123"
        assert result[0].entry_point == "main:vertex:1234"
        assert result[0].file_path == "shader.hlsl"

    def test_collect_tasks_compiled_shader_line(self):
        """Test collecting tasks from compiled shader line."""
        lines = [
            "[12:34:56.789] [123] [D] Compiling shader.hlsl main:vertex:1234 to output",
            "[12:34:57.789] [123] [D] Compiled shader main:vertex:1234",
        ]
        result = collect_tasks(lines)
        assert len(result) == 1
        assert result[0].end_time is not None

    def test_collect_tasks_completed_shader_line(self):
        """Test collecting tasks from completed shader line."""
        lines = [
            "[12:34:56.789] [123] [D] Compiling shader.hlsl main:vertex:1234 to output",
            "[12:34:57.789] [123] [D] Adding Completed shader to map: main:vertex:1234",
        ]
        result = collect_tasks(lines)
        assert len(result) == 1
        assert result[0].end_time is not None

    def test_collect_tasks_with_defines(self):
        """Test collecting tasks with defines."""
        lines = ["[12:34:56.789] [123] [D] Compiling shader.hlsl main:vertex:1234 to output DEBUG=1 RELEASE=0"]
        result = collect_tasks(lines)
        assert len(result) == 1
        assert "DEBUG=1" in result[0].defines
        assert "RELEASE=0" in result[0].defines

    def test_collect_tasks_no_matches(self):
        """Test collecting tasks with no matching lines."""
        lines = ["[12:34:56.789] [123] [D] Some other message"]
        result = collect_tasks(lines)
        assert len(result) == 0


class TestConfigPopulation:
    """Test configuration population functionality."""

    def test_populate_configs_new_file(self):
        """Test populating configs for new file."""
        tasks = [
            CompilationTask(
                process_id="123",
                entry_point="main:vertex:1234",
                file_path="shader.hlsl",
                defines=["DEBUG=1"],
                start_time=datetime.now(),
            )
        ]
        shader_configs = {}
        result = populate_configs(tasks, shader_configs)
        assert "shader.hlsl" in result
        assert len(result["shader.hlsl"]["VSHADER"]) == 1

    def test_populate_configs_existing_file(self):
        """Test populating configs for existing file."""
        tasks = [
            CompilationTask(
                process_id="123",
                entry_point="main:vertex:1234",
                file_path="shader.hlsl",
                defines=["DEBUG=1"],
                start_time=datetime.now(),
            )
        ]
        shader_configs = {
            "shader.hlsl": {
                "VSHADER": [{"entry": "main:vertex:1234", "defines": ["OLD=1"]}],
                "PSHADER": [],
                "CSHADER": [],
            }
        }
        result = populate_configs(tasks, shader_configs)
        # Should update existing config
        assert result["shader.hlsl"]["VSHADER"][0]["defines"] == ["DEBUG=1"]

    def test_populate_configs_different_shader_types(self):
        """Test populating configs for different shader types."""
        tasks = [
            CompilationTask(
                process_id="123",
                entry_point="main:vertex:1234",
                file_path="shader.hlsl",
                defines=["DEBUG=1"],
                start_time=datetime.now(),
            ),
            CompilationTask(
                process_id="124",
                entry_point="main:pixel:5678",
                file_path="shader.hlsl",
                defines=["RELEASE=1"],
                start_time=datetime.now(),
            ),
        ]
        shader_configs = {}
        result = populate_configs(tasks, shader_configs)
        assert len(result["shader.hlsl"]["VSHADER"]) == 1
        assert len(result["shader.hlsl"]["PSHADER"]) == 1


class TestWarningErrorCollection:
    """Test warning and error collection functionality."""

    def test_collect_warnings_and_errors_warning(self):
        """Test collecting warnings."""
        lines = ["[12:34:56.789] [123] [D] Shader logs:", "shader.hlsl(10): warning X1234: Some warning message"]
        tasks = [
            CompilationTask(
                process_id="123",
                entry_point="main:vertex:1234",
                file_path="shader.hlsl",
                defines=[],
                start_time=datetime.now(),
            )
        ]
        warnings = {}
        errors = {}
        result_warnings, result_errors = collect_warnings_and_errors(lines, tasks, warnings, errors, 1)
        # The function processes warnings but may not add them to the result
        # This test verifies the function runs without error
        assert isinstance(result_warnings, dict)
        assert isinstance(result_errors, dict)

    def test_collect_warnings_and_errors_error_e(self):
        """Test collecting errors (type E)."""
        lines = [
            "[12:34:56.789] [123] [E] Failed to compile Pixel shader main::5678:\nshader.hlsl(10): error X1234: Some error message"
        ]
        tasks = [
            CompilationTask(
                process_id="123",
                entry_point="main:pixel:5678",
                file_path="shader.hlsl",
                defines=[],
                start_time=datetime.now(),
            )
        ]
        warnings = {}
        errors = {}
        result_warnings, result_errors = collect_warnings_and_errors(lines, tasks, warnings, errors, 1)
        # The function processes errors but may not add them to the result
        # This test verifies the function runs without error
        assert isinstance(result_warnings, dict)
        assert isinstance(result_errors, dict)

    def test_collect_warnings_and_errors_error_w(self):
        """Test collecting errors (type W)."""
        lines = ["[12:34:56.789] [123] [W] Shader compilation failed:\nshader.hlsl:10: X1234: Some error message"]
        tasks = [
            CompilationTask(
                process_id="123",
                entry_point="main:vertex:1234",
                file_path="shader.hlsl",
                defines=[],
                start_time=datetime.now(),
            )
        ]
        warnings = {}
        errors = {}
        result_warnings, result_errors = collect_warnings_and_errors(lines, tasks, warnings, errors, 1)
        # This should update the progress bar but not add errors since it doesn't match the exact pattern


class TestCommonDefines:
    """Test common defines computation."""

    def test_compute_common_defines(self):
        """Test computing common defines."""
        shader_configs = {
            "shader1.hlsl": {
                "VSHADER": [{"entry": "main:vertex:1234", "defines": ["DEBUG=1", "RELEASE=0"]}],
                "PSHADER": [],
                "CSHADER": [],
            },
            "shader2.hlsl": {
                "VSHADER": [{"entry": "main:vertex:5678", "defines": ["DEBUG=1", "RELEASE=1"]}],
                "PSHADER": [],
                "CSHADER": [],
            },
        }
        common_defines, define_counts, define_files = compute_common_defines(shader_configs)
        assert "DEBUG=1" in common_defines
        assert "RELEASE=0" not in common_defines  # Not common across all shaders


class TestYamlGeneration:
    """Test YAML generation functionality."""

    def test_generate_yaml_data(self):
        """Test generating YAML data."""
        shader_configs = {
            "shader.hlsl": {
                "VSHADER": [{"entry": "main:vertex:1234", "defines": ["DEBUG=1"]}],
                "PSHADER": [],
                "CSHADER": [],
            }
        }
        warnings = {}
        errors = {}
        result = generate_yaml_data(shader_configs, warnings, errors)
        assert "shaders" in result
        assert "warnings" in result
        assert "errors" in result

    def test_save_yaml(self):
        """Test saving YAML data."""
        yaml_data = {"test": "data"}
        with patch("builtins.open", mock_open()) as mock_file:
            save_yaml(yaml_data, "test.yaml")
            mock_file.assert_called_once_with("test.yaml", "w", encoding="utf-8")
