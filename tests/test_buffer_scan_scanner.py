"""Tests for FileScanner class and file scanning functionality."""

import re
from unittest.mock import mock_open, patch

from hlslkit.buffer_scan import FileScanner


class TestFileScanner:
    """Test FileScanner class functionality."""

    def test_init(self):
        """Test FileScanner initialization."""
        scanner = FileScanner("/test/path")
        assert scanner.cwd == "/test/path"
        assert isinstance(scanner.excluded_dirs, set)

    def test_get_short_path_with_skyrim_community_shaders(self):
        """Test _get_short_path with skyrim-community-shaders in path."""
        scanner = FileScanner("/test/path")
        full_path = "/some/path/skyrim-community-shaders/shaders/test.hlsl"
        result = scanner._get_short_path(full_path)
        assert result == "shaders/test.hlsl"

    def test_get_short_path_without_skyrim_community_shaders(self):
        """Test _get_short_path without skyrim-community-shaders in path."""
        scanner = FileScanner("/test/path")
        full_path = "/some/other/path/test.hlsl"
        with patch("os.path.relpath", return_value="relative/path/test.hlsl"):
            result = scanner._get_short_path(full_path)
            assert result == "relative/path/test.hlsl"

    def test_get_short_path_case_insensitive(self):
        """Test _get_short_path with case insensitive matching."""
        scanner = FileScanner("/test/path")
        full_path = "/some/path/SKYRIM-COMMUNITY-SHADERS/shaders/test.hlsl"
        result = scanner._get_short_path(full_path)
        assert result == "shaders/test.hlsl"

    @patch("os.walk")
    @patch("hlslkit.buffer_scan.process_file")
    def test_scan_for_buffers(self, mock_process_file, mock_walk):
        """Test scan_for_buffers method."""
        scanner = FileScanner("/test/path")

        # Mock os.walk to return a simple directory structure
        mock_walk.return_value = [
            ("/test/path", ["subdir"], ["test.hlsl", "other.txt"]),
        ]

        # Mock feature pattern
        feature_pattern = re.compile(r"(?P<feature>test)")

        # Mock shader pattern
        shader_pattern = re.compile(r"cbuffer")

        # Mock HLSL types
        hlsl_types = {"b": "bool", "f": "float"}

        # Mock defines list
        defines_list = [{"TEST": "1"}]

        # Mock pattern (not used in this test but required)
        pattern = re.compile(r".*")

        result, compilation_units = scanner.scan_for_buffers(
            pattern, feature_pattern, shader_pattern, hlsl_types, defines_list
        )

        assert isinstance(result, list)
        assert isinstance(compilation_units, dict)
        # process_file should be called for the .hlsl file
        mock_process_file.assert_called()

    @patch("os.walk")
    @patch("builtins.open", new_callable=mock_open, read_data="struct TestStruct { int a; };")
    @patch("hlslkit.buffer_scan.extract_structs")
    @patch("hlslkit.buffer_scan.is_shader_io_struct")
    def test_scan_for_structs_hlsl(self, mock_is_shader_io, mock_extract, mock_open, mock_walk):
        """Test scan_for_structs with HLSL files."""
        scanner = FileScanner("/test/path")

        # Mock os.walk to return HLSL files
        mock_walk.return_value = [
            ("/test/path", ["subdir"], ["test.hlsl"]),
        ]

        # Mock extract_structs to return a struct
        mock_extract.return_value = {"TestStruct": {"fields": [], "line": 1}}

        # Mock is_shader_io_struct to return False
        mock_is_shader_io.return_value = False

        hlsl_structs, cpp_structs = scanner.scan_for_structs()

        assert isinstance(hlsl_structs, dict)
        assert isinstance(cpp_structs, dict)
        assert "TestStruct" in hlsl_structs

    @patch("os.walk")
    @patch("builtins.open", new_callable=mock_open, read_data="struct TestStruct { int a; };")
    @patch("hlslkit.buffer_scan.extract_structs")
    @patch("hlslkit.buffer_scan.is_shader_io_struct")
    def test_scan_for_structs_cpp(self, mock_is_shader_io, mock_extract, mock_open, mock_walk):
        """Test scan_for_structs with C++ files."""
        scanner = FileScanner("/test/path")

        # Mock os.walk to return C++ files
        mock_walk.return_value = [
            ("/test/path", ["subdir"], ["test.cpp"]),
        ]

        # Mock extract_structs to return a struct
        mock_extract.return_value = {"TestStruct": {"fields": [], "line": 1}}

        # Mock is_shader_io_struct to return False
        mock_is_shader_io.return_value = False

        hlsl_structs, cpp_structs = scanner.scan_for_structs()

        assert isinstance(hlsl_structs, dict)
        assert isinstance(cpp_structs, dict)
        assert "TestStruct" in cpp_structs

    @patch("os.walk")
    @patch("builtins.open", new_callable=mock_open, read_data="struct TestStruct { int a; };")
    @patch("hlslkit.buffer_scan.extract_structs")
    @patch("hlslkit.buffer_scan.is_shader_io_struct")
    def test_scan_for_structs_shader_io(self, mock_is_shader_io, mock_extract, mock_open, mock_walk):
        """Test scan_for_structs with shader IO structs."""
        scanner = FileScanner("/test/path")

        # Mock os.walk to return HLSL files
        mock_walk.return_value = [
            ("/test/path", ["subdir"], ["test.hlsl"]),
        ]

        # Mock extract_structs to return a shader IO struct
        mock_extract.return_value = {"VSInput": {"fields": [], "line": 1}}

        # Mock is_shader_io_struct to return True for VSInput
        def mock_is_shader_io_side_effect(name):
            return name == "VSInput"

        mock_is_shader_io.side_effect = mock_is_shader_io_side_effect

        hlsl_structs, cpp_structs = scanner.scan_for_structs()

        # VSInput should be skipped
        assert "VSInput" not in hlsl_structs

    @patch("os.walk")
    @patch("builtins.open", new_callable=mock_open, read_data="struct TestStruct { int a; };")
    @patch("hlslkit.buffer_scan.extract_structs")
    @patch("hlslkit.buffer_scan.is_shader_io_struct")
    def test_scan_for_structs_invalid_data(self, mock_is_shader_io, mock_extract, mock_open, mock_walk):
        """Test scan_for_structs with invalid struct data."""
        scanner = FileScanner("/test/path")

        # Mock os.walk to return HLSL files
        mock_walk.return_value = [
            ("/test/path", ["subdir"], ["test.hlsl"]),
        ]

        # Mock extract_structs to return invalid data
        mock_extract.return_value = {"TestStruct": "invalid_data"}

        # Mock is_shader_io_struct to return False
        mock_is_shader_io.return_value = False

        hlsl_structs, cpp_structs = scanner.scan_for_structs()

        # Invalid data should be skipped
        assert "TestStruct" not in hlsl_structs

    @patch("os.walk")
    @patch("builtins.open", new_callable=mock_open, read_data="struct TestStruct { int a; };")
    @patch("hlslkit.buffer_scan.extract_structs")
    @patch("hlslkit.buffer_scan.is_shader_io_struct")
    def test_scan_for_structs_template_handling(self, mock_is_shader_io, mock_extract, mock_open, mock_walk):
        """Test scan_for_structs with template handling."""
        scanner = FileScanner("/test/path")

        # Mock os.walk to return HLSL files
        mock_walk.return_value = [
            ("/test/path", ["subdir"], ["test.hlsl"]),
        ]

        # Mock extract_structs to return template and real struct
        mock_extract.return_value = {
            "TestStruct": {"fields": [], "line": 1, "is_template": True},
            "RealStruct": {"fields": [{"name": "a", "type": "int"}], "line": 2, "is_template": False},
        }

        # Mock is_shader_io_struct to return False
        mock_is_shader_io.return_value = False

        hlsl_structs, cpp_structs = scanner.scan_for_structs()

        # Both should be included
        assert "TestStruct" in hlsl_structs
        assert "RealStruct" in hlsl_structs

    @patch("os.walk")
    @patch("builtins.open", side_effect=OSError("File not found"))
    def test_scan_for_structs_file_error(self, mock_open, mock_walk):
        """Test scan_for_structs with file reading error."""
        scanner = FileScanner("/test/path")

        # Mock os.walk to return files
        mock_walk.return_value = [
            ("/test/path", ["subdir"], ["test.hlsl"]),
        ]

        hlsl_structs, cpp_structs = scanner.scan_for_structs()

        # Should handle error gracefully
        assert isinstance(hlsl_structs, dict)
        assert isinstance(cpp_structs, dict)
        assert len(hlsl_structs) == 0
        assert len(cpp_structs) == 0

    @patch("os.walk")
    def test_scan_for_structs_unsupported_files(self, mock_walk):
        """Test scan_for_structs with unsupported file types."""
        scanner = FileScanner("/test/path")

        # Mock os.walk to return unsupported files
        mock_walk.return_value = [
            ("/test/path", ["subdir"], ["test.txt", "test.pdf"]),
        ]

        hlsl_structs, cpp_structs = scanner.scan_for_structs()

        # No structs should be found
        assert len(hlsl_structs) == 0
        assert len(cpp_structs) == 0
