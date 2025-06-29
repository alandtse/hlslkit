import pytest

from hlslkit import generate_shader_defines


def test_parse_log_file_not_found():
    with pytest.raises(FileNotFoundError):
        generate_shader_defines.parse_log("nonexistent.log")


def test_parse_log_invalid(monkeypatch, tmp_path):
    # Create a file with invalid log content
    log_file = tmp_path / "bad.log"
    log_file.write_text("not a real log line\n")
    # Should not raise, but return empty configs/warnings/errors
    configs, warnings, errors = generate_shader_defines.parse_log(str(log_file))
    assert isinstance(configs, dict)
    assert isinstance(warnings, dict)
    assert isinstance(errors, dict)


def test_generate_yaml_data_empty():
    # Should not raise for empty input
    result = generate_shader_defines.generate_yaml_data({}, {}, {})
    assert isinstance(result, dict)


def test_generate_yaml_data_minimal():
    # Should not raise for minimal valid input
    configs = {"foo.hlsl": {"PSHADER": [{"entry": "main:vertex:1234", "defines": ["FOO=1"]}]}}
    result = generate_shader_defines.generate_yaml_data(configs, {}, {})
    assert isinstance(result, dict)
