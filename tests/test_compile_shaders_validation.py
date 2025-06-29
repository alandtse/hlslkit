"""Tests for shader input validation functionality."""

from unittest.mock import patch

from hlslkit.compile_shaders import validate_shader_inputs


def test_validate_shader_inputs_single_file_mode():
    """Test validate_shader_inputs when shader_dir is a file path."""
    # Test when shader_dir is a file (single-file mode)
    with (
        patch("hlslkit.compile_shaders.shutil.which") as mock_which,
        patch("hlslkit.compile_shaders.os.path.isfile") as mock_isfile,
        patch("hlslkit.compile_shaders.os.path.abspath") as mock_abspath,
        patch("hlslkit.compile_shaders.os.path.isdir") as mock_isdir,
    ):
        mock_which.return_value = "fxc.exe"
        mock_isfile.return_value = True  # shader_dir is a file
        mock_abspath.return_value = "/absolute/path/to/shader.hlsl"
        mock_isdir.return_value = True  # output directory exists

        result = validate_shader_inputs("fxc.exe", "shader.hlsl", "output", [], "/path/to/shader.hlsl")
        assert result is None  # Should pass validation


def test_validate_shader_inputs_single_file_mode_invalid_file():
    """Test validate_shader_inputs when shader_dir is a file but shader_file doesn't exist."""
    with (
        patch("hlslkit.compile_shaders.shutil.which") as mock_which,
        patch("hlslkit.compile_shaders.os.path.isfile") as mock_isfile,
        patch("hlslkit.compile_shaders.os.path.abspath") as mock_abspath,
        patch("hlslkit.compile_shaders.os.path.join") as mock_join,
    ):
        mock_which.return_value = "fxc.exe"
        mock_isfile.return_value = True  # shader_dir is a file
        mock_abspath.return_value = "/absolute/path/to/shader.hlsl"
        mock_join.return_value = "/path/to/shader.hlsl"

        # Mock that the shader file doesn't exist or has wrong extension
        with patch("hlslkit.compile_shaders.os.path.isfile") as mock_file_exists:
            mock_file_exists.return_value = False

            result = validate_shader_inputs("fxc.exe", "nonexistent.hlsl", "output", [], "/path/to/shader.hlsl")
            assert result is not None
            assert "Invalid shader file" in result


def test_validate_shader_inputs_single_file_mode_wrong_extension():
    """Test validate_shader_inputs when shader_dir is a file but shader_file has wrong extension."""
    with (
        patch("hlslkit.compile_shaders.shutil.which") as mock_which,
        patch("hlslkit.compile_shaders.os.path.isfile") as mock_isfile,
        patch("hlslkit.compile_shaders.os.path.abspath") as mock_abspath,
        patch("hlslkit.compile_shaders.os.path.join") as mock_join,
    ):
        mock_which.return_value = "fxc.exe"
        mock_isfile.return_value = True  # shader_dir is a file
        mock_abspath.return_value = "/absolute/path/to/shader.txt"
        mock_join.return_value = "/path/to/shader.txt"

        # Mock that the file exists but has wrong extension
        with patch("hlslkit.compile_shaders.os.path.isfile") as mock_file_exists:
            mock_file_exists.return_value = True

            result = validate_shader_inputs("fxc.exe", "shader.txt", "output", [], "/path/to/shader.txt")
            assert result is not None
            assert "Invalid shader file" in result


def test_validate_shader_inputs_fxc_not_found():
    """Test validate_shader_inputs when fxc.exe is not found."""
    with patch("hlslkit.compile_shaders.shutil.which") as mock_which:
        mock_which.return_value = None  # fxc.exe not found
        result = validate_shader_inputs(
            fxc_path="nonexistent_fxc.exe",
            shader_file="test.hlsl",
            output_dir="output",
            defines=["A=1"],
            shader_dir="shaders",
        )
        assert result is not None
        assert "fxc.exe not found" in result


def test_validate_shader_inputs_invalid_output_dir():
    """Test validate_shader_inputs with invalid output directory."""
    with patch("hlslkit.compile_shaders.shutil.which") as mock_which:
        mock_which.return_value = "/path/to/fxc.exe"
        with patch("hlslkit.compile_shaders.os.path.isfile") as mock_isfile:
            mock_isfile.return_value = False  # shader_dir is a directory
            with patch("hlslkit.compile_shaders.os.path.join") as mock_join:
                mock_join.return_value = "/shaders/test.hlsl"
                with patch("hlslkit.compile_shaders.os.path.isfile") as mock_isfile2:
                    mock_isfile2.return_value = True
                    with patch("hlslkit.compile_shaders.os.path.abspath") as mock_abspath:
                        mock_abspath.return_value = "/absolute/output"
                        with patch("hlslkit.compile_shaders.os.path.isdir") as mock_isdir:
                            mock_isdir.return_value = False  # Output dir doesn't exist
                            result = validate_shader_inputs(
                                fxc_path="fxc.exe",
                                shader_file="test.hlsl",
                                output_dir="nonexistent_output",
                                defines=["A=1"],
                                shader_dir="shaders",
                            )
                            assert result is not None
                            assert "Invalid output directory" in result


def test_validate_shader_inputs_invalid_defines():
    """Test validate_shader_inputs with invalid defines."""
    with patch("hlslkit.compile_shaders.shutil.which") as mock_which:
        mock_which.return_value = "/path/to/fxc.exe"
        with patch("hlslkit.compile_shaders.os.path.isfile") as mock_isfile:
            mock_isfile.return_value = False  # shader_dir is a directory
            with patch("hlslkit.compile_shaders.os.path.join") as mock_join:
                mock_join.return_value = "/shaders/test.hlsl"
                with patch("hlslkit.compile_shaders.os.path.isfile") as mock_isfile2:
                    mock_isfile2.return_value = True
                    with patch("hlslkit.compile_shaders.os.path.abspath") as mock_abspath:
                        mock_abspath.return_value = "/absolute/output"
                        with patch("hlslkit.compile_shaders.os.path.isdir") as mock_isdir:
                            mock_isdir.return_value = True
                            result = validate_shader_inputs(
                                fxc_path="fxc.exe",
                                shader_file="test.hlsl",
                                output_dir="output",
                                defines=["invalid-define", "valid_define=1"],
                                shader_dir="shaders",
                            )
                            assert result is not None
                            assert "Invalid defines" in result
                            assert "invalid-define" in result


def test_validate_shader_inputs_valid_defines():
    """Test validate_shader_inputs with valid defines."""
    with patch("hlslkit.compile_shaders.shutil.which") as mock_which:
        mock_which.return_value = "/path/to/fxc.exe"
        with patch("hlslkit.compile_shaders.os.path.isfile") as mock_isfile:
            mock_isfile.return_value = False  # shader_dir is a directory
            with patch("hlslkit.compile_shaders.os.path.join") as mock_join:
                mock_join.return_value = "/shaders/test.hlsl"
                with patch("hlslkit.compile_shaders.os.path.isfile") as mock_isfile2:
                    mock_isfile2.return_value = True
                    with patch("hlslkit.compile_shaders.os.path.abspath") as mock_abspath:
                        mock_abspath.return_value = "/absolute/output"
                        with patch("hlslkit.compile_shaders.os.path.isdir") as mock_isdir:
                            mock_isdir.return_value = True
                            result = validate_shader_inputs(
                                fxc_path="fxc.exe",
                                shader_file="test.hlsl",
                                output_dir="output",
                                defines=["VALID_DEFINE=1", "ANOTHER_VALID_DEFINE"],
                                shader_dir="shaders",
                            )
                            assert result is None  # Should pass validation


def test_validate_shader_inputs_non_hlsl_file():
    """Test validate_shader_inputs with non-HLSL file extension."""
    with patch("hlslkit.compile_shaders.shutil.which") as mock_which:
        mock_which.return_value = "/path/to/fxc.exe"
        with patch("hlslkit.compile_shaders.os.path.isfile") as mock_isfile:
            mock_isfile.return_value = False  # shader_dir is a directory
            with patch("hlslkit.compile_shaders.os.path.join") as mock_join:
                mock_join.return_value = "/shaders/test.txt"
                with patch("hlslkit.compile_shaders.os.path.isfile") as mock_isfile2:
                    mock_isfile2.return_value = True
                    with patch("hlslkit.compile_shaders.os.path.abspath") as mock_abspath:
                        mock_abspath.return_value = "/absolute/output"
                        with patch("hlslkit.compile_shaders.os.path.isdir") as mock_isdir:
                            mock_isdir.return_value = True
                            result = validate_shader_inputs(
                                fxc_path="fxc.exe",
                                shader_file="test.txt",  # Not .hlsl or .hlsli
                                output_dir="output",
                                defines=["A=1"],
                                shader_dir="shaders",
                            )
                            assert result is not None
                            assert "Invalid shader file" in result
