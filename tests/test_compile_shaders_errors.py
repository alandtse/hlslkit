import sys
from unittest.mock import patch

import pytest

from hlslkit import compile_shaders


def test_main_no_args(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["compile_shaders.py"])
    with pytest.raises(SystemExit):
        compile_shaders.main()
    # No assertion on output, just ensure it exits


def test_main_invalid_file(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["compile_shaders.py", "nonexistent.hlsl"])
    with patch("builtins.open", side_effect=FileNotFoundError), pytest.raises(SystemExit):
        compile_shaders.main()
    # No assertion on output, just ensure it exits


def test_validate_shader_inputs_invalid():
    # Should return error message for missing fxc_path
    result = compile_shaders.validate_shader_inputs("", "", "", [], "")
    assert isinstance(result, str)


def test_validate_shader_inputs_valid(tmp_path, monkeypatch):
    # Should return None for valid input (simulate fxc.exe and shader file)
    fxc = tmp_path / "fxc.exe"
    fxc.write_text("")
    shader = tmp_path / "shader.hlsl"
    shader.write_text("// shader code")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    monkeypatch.setattr("shutil.which", lambda path: str(fxc) if path == str(fxc) or path == "fxc.exe" else None)
    result = compile_shaders.validate_shader_inputs(str(fxc), str(shader), str(output_dir), [], str(tmp_path))
    assert result is None
