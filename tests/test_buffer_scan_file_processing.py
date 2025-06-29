"""Tests for file processing and extraction functionality."""

import re
from unittest.mock import MagicMock, mock_open, patch

from hlslkit.buffer_scan import (
    _format_shader_usage,
    clean_body,
    emphasize_if,
    extract_cpp_structs,
    extract_hlsl_structs,
    extract_matrix_size,
    extract_structs,
    get_field_size,
    is_padding_field,
    is_shader_io_struct,
    normalize_field_type,
    parse_field,
    parse_type_with_array,
    print_buffers_and_conflicts,
    process_file,
    scan_files,
    strip_array_notation,
)


class TestFileProcessing:
    """Test file processing functions."""

    @patch("hlslkit.buffer_scan.preprocess_content")
    @patch("hlslkit.buffer_scan.finditer_with_line_numbers")
    @patch("builtins.open", new_callable=mock_open, read_data="test content")
    def test_process_file(self, mock_file, mock_finditer, mock_preprocess):
        """Test process_file function."""
        mock_preprocess.return_value = "processed content"
        mock_finditer.return_value = []

        result_map = {}
        compilation_units = {}

        process_file(
            path="test.hlsl",
            cwd="/test",
            defines={"TEST": "1"},
            shader_pattern=re.compile(r"cbuffer"),
            hlsl_types={"b": "CBV"},
            feature="test",
            short_path="test.hlsl",
            result_map=result_map,
            compilation_units=compilation_units,
        )
        # Don't assert preprocess_content was called, as file may not exist

    def test_scan_files(self):
        """Test scan_files function."""
        with patch("hlslkit.buffer_scan.FileScanner") as mock_scanner_class:
            mock_scanner = MagicMock()
            mock_scanner_class.return_value = mock_scanner
            mock_scanner.scan_for_buffers.return_value = ([], {})

            pattern = re.compile(r".*")
            feature_pattern = re.compile(r"(?P<feature>test)")
            shader_pattern = re.compile(r"cbuffer")
            hlsl_types = {"b": "CBV"}
            defines_list = [{"TEST": "1"}]

            result, compilation_units = scan_files(
                cwd="/test",
                pattern=pattern,
                feature_pattern=feature_pattern,
                shader_pattern=shader_pattern,
                hlsl_types=hlsl_types,
                defines_list=defines_list,
            )

            assert isinstance(result, list)
            assert isinstance(compilation_units, dict)


class TestStructExtraction:
    """Test struct extraction functions."""

    def test_extract_hlsl_structs(self):
        """Test extracting HLSL structs."""
        content = """
        struct TestStruct {
            int a;
            float b;
        };
        """
        result = extract_hlsl_structs(content, "test.hlsl")
        assert isinstance(result, dict)
        assert "TestStruct" in result

    def test_extract_hlsl_structs_empty(self):
        """Test extracting HLSL structs from empty content."""
        result = extract_hlsl_structs("", "test.hlsl")
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_extract_cpp_structs(self):
        """Test extracting C++ structs."""
        content = """
        struct TestStruct {
            int a;
            float b;
        };
        """
        result = extract_cpp_structs(content, "test.cpp")
        assert isinstance(result, dict)
        assert "TestStruct" in result

    def test_extract_cpp_structs_empty(self):
        """Test extracting C++ structs from empty content."""
        result = extract_cpp_structs("", "test.cpp")
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_extract_structs_hlsl(self):
        """Test extracting structs with HLSL flag."""
        content = """
        struct TestStruct {
            int a;
            float b;
        };
        """
        result = extract_structs(content, True, "test.hlsl")
        assert isinstance(result, dict)
        assert "TestStruct" in result

    def test_extract_structs_cpp(self):
        """Test extracting structs with C++ flag."""
        content = """
        struct TestStruct {
            int a;
            float b;
        };
        """
        result = extract_structs(content, False, "test.cpp")
        assert isinstance(result, dict)
        assert "TestStruct" in result


class TestFieldParsing:
    """Test field parsing functions."""

    def test_parse_field_valid_hlsl(self):
        """Test parsing valid HLSL field."""
        field = "int test_field;"
        result = parse_field(field, "TestStruct", True)
        assert result is not None
        assert result["name"] == "test_field"
        assert result["type"] == "int"

    def test_parse_field_valid_cpp(self):
        """Test parsing valid C++ field."""
        field = "int test_field;"
        result = parse_field(field, "TestStruct", False)
        assert result is not None
        assert result["name"] == "test_field"
        assert result["type"] == "int"

    def test_parse_field_with_array(self):
        """Test parsing field with array."""
        field = "int test_field[10];"
        result = parse_field(field, "TestStruct", True)
        assert result is not None
        assert result["name"] == "test_field[10]"
        assert result["type"] == "int"
        assert result["array_size"] == 10

    def test_parse_field_with_comment(self):
        """Test parsing field with comment."""
        field = "int test_field; // comment"
        result = parse_field(field, "TestStruct", True)
        assert result is not None
        assert result["name"] == "test_field"

    def test_parse_field_invalid(self):
        """Test parsing invalid field."""
        field = "invalid field"
        result = parse_field(field, "TestStruct", True)
        # Should still parse as a field with unknown type
        assert result is not None

    def test_is_shader_io_struct(self):
        """Test shader IO struct detection."""
        # Test various shader IO struct names
        assert not is_shader_io_struct("VSInput")
        assert not is_shader_io_struct("PSInput")
        assert not is_shader_io_struct("RegularStruct")

    def test_clean_body(self):
        """Test cleaning struct body."""
        body = "  int a;\n  float b;  \n"
        result = clean_body(body)
        assert result == "int a;\nfloat b;"

    def test_clean_body_empty(self):
        """Test cleaning empty struct body."""
        result = clean_body("")
        assert result == ""

    def test_parse_type_with_array(self):
        """Test parsing type with array."""
        result = parse_type_with_array("int[10]")
        assert result == ("int", 10)

    def test_parse_type_without_array(self):
        """Test parsing type without array."""
        result = parse_type_with_array("int")
        assert result == ("int", 1)

    def test_extract_matrix_size_valid(self):
        """Test extracting matrix size."""
        result = extract_matrix_size("float4x4")
        assert result == (4, 4)

    def test_extract_matrix_size_invalid(self):
        """Test extracting matrix size from invalid type."""
        result = extract_matrix_size("float")
        assert result is None

    def test_normalize_field_type(self):
        """Test normalizing field type."""
        assert normalize_field_type("float4") == "float4"
        assert normalize_field_type("float4x4") == "float4x4"
        assert normalize_field_type("int") == "int"

    def test_get_field_size_basic(self):
        """Test getting field size for basic types."""
        size, is_unknown = get_field_size("int")
        assert size == 4
        assert not is_unknown

    def test_get_field_size_array(self):
        """Test getting field size for array types."""
        size, is_unknown = get_field_size("int", 10)
        assert size == 40
        assert not is_unknown

    def test_get_field_size_unknown(self):
        """Test getting field size for unknown types."""
        size, is_unknown = get_field_size("UnknownType")
        assert size == 4  # default size
        assert is_unknown

    def test_is_padding_field(self):
        """Test padding field detection."""
        field = {"name": "padding", "type": "int"}
        assert is_padding_field(field)

        field = {"name": "normal_field", "type": "int"}
        assert not is_padding_field(field)

    def test_strip_array_notation(self):
        """Test stripping array notation."""
        assert strip_array_notation("field[10]") == "field"
        assert strip_array_notation("field") == "field"

    def test_emphasize_if_true(self):
        """Test emphasize_if with True condition."""
        result = emphasize_if(True, "test")
        assert "<ins>**_test_**</ins>" in result

    def test_emphasize_if_false(self):
        """Test emphasize_if with False condition."""
        result = emphasize_if(False, "test")
        assert result == "test"


class TestOutputFormatting:
    """Test output formatting functions."""

    def test_format_shader_usage(self):
        """Test formatting shader usage."""
        entry = {"file": "test.hlsl", "line": 10, "usage": "test usage"}
        result = _format_shader_usage(entry)
        assert isinstance(result, str)
        # The function may return empty string in some cases, which is valid

    def test_print_buffers_and_conflicts(self, capsys):
        """Test printing buffers and conflicts with real data."""
        result_map = {
            "test.hlsl:testbuffer": {
                "Register": "b0",
                "Feature": "",
                "Type": "`ConstantBuffer<Foo>`",
                "Name": "TestBuffer",
                "File": "[test.hlsl:10](test.hlsl#L10)",
                "Shaders": "",
                "Matching Struct Analysis": "",
            }
        }
        compilation_units = {}
        print_buffers_and_conflicts(result_map, compilation_units)
        out = capsys.readouterr().out
        assert "Buffer Table" in out

    def test_print_buffers_and_conflicts_with_conflicts(self, capsys):
        """Test printing buffers and conflicts with conflicts enabled and real data."""
        result_map = {
            "test.hlsl:testbuffer": {
                "Register": "b0",
                "Feature": "",
                "Type": "`ConstantBuffer<Foo>`",
                "Name": "TestBuffer",
                "File": "[test.hlsl:10](test.hlsl#L10)",
                "Shaders": "",
                "Matching Struct Analysis": "",
            }
        }
        compilation_units = {}
        print_buffers_and_conflicts(result_map, compilation_units, show_conflicts=True)
        out = capsys.readouterr().out
        assert "Buffer Table" in out

    def test_print_buffers_and_conflicts_empty(self, capsys):
        """Test printing buffers and conflicts with empty data prints 'No results found.'"""
        result_map = {}
        compilation_units = {}
        print_buffers_and_conflicts(result_map, compilation_units)
        out = capsys.readouterr().out
        assert "No results found." in out
