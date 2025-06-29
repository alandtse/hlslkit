"""Tests for utility functions in buffer_scan.py."""

from unittest.mock import patch

from hlslkit.buffer_scan import (
    add_debug_info,
    capture_pattern,
    clean_body,
    clear_debug_info,
    create_link,
    create_struct_analysis_link,
    create_struct_section_id,
    emphasize_if,
    extract_matrix_size,
    finditer_with_line_numbers,
    get_defines_list,
    get_excluded_dirs,
    get_field_size,
    get_hlsl_types,
    is_padding_field,
    is_shader_io_struct,
    normalize_field_type,
    parse_field,
    parse_type_with_array,
    preprocess_content,
    strip_array_notation,
)


class TestDebugInfo:
    """Test debug info functionality."""

    def test_add_debug_info(self):
        """Test adding debug information."""
        from hlslkit.buffer_scan import DEBUG_INFO

        clear_debug_info()
        add_debug_info("test message")
        assert "test message" in DEBUG_INFO

    def test_clear_debug_info(self):
        """Test clearing debug information."""
        from hlslkit.buffer_scan import DEBUG_INFO

        add_debug_info("test message")
        clear_debug_info()
        assert len(DEBUG_INFO) == 0


class TestLinkGeneration:
    """Test link generation functions."""

    def test_create_link_without_line(self):
        """Test creating a link without line number."""
        result = create_link("test/file.hlsl")
        expected = "https://github.com/doodlum/skyrim-community-shaders/blob/dev/test/file.hlsl"
        assert result == expected

    def test_create_link_with_line(self):
        """Test creating a link with line number."""
        result = create_link("test/file.hlsl", 42)
        expected = "https://github.com/doodlum/skyrim-community-shaders/blob/dev/test/file.hlsl#L42"
        assert result == expected

    def test_create_link_with_special_characters(self):
        """Test creating a link with special characters."""
        result = create_link("test/file with spaces.hlsl")
        expected = "https://github.com/doodlum/skyrim-community-shaders/blob/dev/test/file%20with%20spaces.hlsl"
        assert result == expected

    def test_create_struct_section_id(self):
        """Test creating struct section ID."""
        result = create_struct_section_id("TestStruct", "test_file.hlsl")
        assert result == "hlsl-teststruct-test_filehlsl"

    def test_create_struct_section_id_with_parentheses(self):
        """Test creating struct section ID with parentheses."""
        result = create_struct_section_id("Test(Struct)", "test(file).hlsl")
        assert result == "hlsl-teststruct-testfilehlsl"

    def test_create_struct_analysis_link_unmatched(self):
        """Test creating analysis link for unmatched struct."""
        result = create_struct_analysis_link("TestStruct", "test.hlsl", "Unmatched")
        assert result == "[Unmatched](#hlsl-teststruct-testhlsl)"

    def test_create_struct_analysis_link_mismatched(self):
        """Test creating analysis link for mismatched struct."""
        result = create_struct_analysis_link("TestStruct", "test.hlsl", "Mismatched (OtherStruct)")
        assert result == "[Mismatched (`OtherStruct`)](#hlsl-teststruct-testhlsl)"

    def test_create_struct_analysis_link_mismatched_no_name(self):
        """Test creating analysis link for mismatched struct without name."""
        result = create_struct_analysis_link("TestStruct", "test.hlsl", "Mismatched")
        assert result == "[Mismatched](#hlsl-teststruct-testhlsl)"

    def test_create_struct_analysis_link_matched(self):
        """Test creating analysis link for matched struct."""
        result = create_struct_analysis_link("TestStruct", "test.hlsl", "Matched")
        assert result == "[`TestStruct`](#hlsl-teststruct-testhlsl)"

    def test_create_struct_analysis_link_other_status(self):
        """Test creating analysis link for other status."""
        result = create_struct_analysis_link("TestStruct", "test.hlsl", "CustomStatus")
        assert result == "[`TestStruct`](#hlsl-teststruct-testhlsl)"


class TestPatternMatching:
    """Test pattern matching functions."""

    def test_finditer_with_line_numbers_no_matches(self):
        """Test finditer_with_line_numbers with no matches."""
        result = finditer_with_line_numbers(r"nonexistent", "test string")
        assert result == []

    def test_finditer_with_line_numbers_with_matches(self):
        """Test finditer_with_line_numbers with matches."""
        result = finditer_with_line_numbers(r"test", "test string\ntest again")
        assert len(result) == 2
        assert result[0][0] == 1  # line number
        assert result[1][0] == 2  # line number

    def test_finditer_with_line_numbers_with_line_directives(self):
        """Test finditer_with_line_numbers with #line directives."""
        text = '#line 100 "test.hlsl"\ntest string'
        result = finditer_with_line_numbers(r"test", text)
        assert len(result) == 2  # Both "test" in filename and "test" in content
        assert result[0][0] == 100  # adjusted line number for content

    def test_finditer_with_line_numbers_with_line_map(self):
        """Test finditer_with_line_numbers with line map."""
        line_map = {1: 10, 2: 20}
        result = finditer_with_line_numbers(r"test", "test string\ntest again", line_map=line_map)
        assert len(result) == 2
        assert result[0][0] == 10  # mapped line number
        assert result[1][0] == 20  # mapped line number

    def test_capture_pattern(self):
        """Test capture_pattern function."""
        text = '#line 50 "test.hlsl"\ntest string\nanother test'
        result = capture_pattern(text, r"test")
        assert len(result) == 1  # Only matches at the beginning of lines
        assert result[0][0] == 50  # adjusted line number


class TestConfiguration:
    """Test configuration functions."""

    def test_get_hlsl_types(self):
        """Test getting HLSL types."""
        types = get_hlsl_types()
        assert isinstance(types, dict)
        assert "b" in types  # bool type
        assert "s" in types  # sampler type

    def test_get_defines_list(self):
        """Test getting defines list."""
        defines = get_defines_list()
        assert isinstance(defines, list)
        assert all(isinstance(d, dict) for d in defines)

    def test_get_excluded_dirs(self):
        """Test getting excluded directories."""
        with patch("os.path.exists", return_value=True):
            excluded = get_excluded_dirs("/test/path")
            assert isinstance(excluded, set)
            # Should include common excluded directories
            assert "tools" in excluded
            assert "build" in excluded


class TestPreprocessing:
    """Test preprocessing functions."""

    def test_preprocess_content(self):
        """Test preprocessing content with defines."""
        content = "#define TEST 1\nint value = TEST;"
        defines = {"TEST": "42"}
        result = preprocess_content(content, defines)
        # The function doesn't actually replace defines, it just returns the content
        assert result == content

    def test_preprocess_content_no_defines(self):
        """Test preprocessing content without defines."""
        content = "int value = 1;"
        defines = {}
        result = preprocess_content(content, defines)
        assert result == content


class TestStructUtilities:
    """Test struct utility functions."""

    def test_is_shader_io_struct(self):
        """Test shader IO struct detection."""
        assert not is_shader_io_struct("VSInput")
        assert not is_shader_io_struct("PSInput")
        assert not is_shader_io_struct("RegularStruct")

    def test_clean_body(self):
        """Test cleaning struct body."""
        body = "  int a;\n  float b;  \n"
        result = clean_body(body)
        assert result == "int a;\nfloat b;"

    def test_parse_field_valid(self):
        """Test parsing valid field."""
        field = "int test_field;"
        result = parse_field(field, "TestStruct", False)
        assert result is not None
        assert result["name"] == "test_field"
        assert result["type"] == "int"

    def test_parse_field_invalid(self):
        """Test parsing invalid field."""
        field = "invalid field"
        result = parse_field(field, "TestStruct", False)
        # The function actually parses this as a field named "field" with unknown type
        assert result is not None
        assert result["name"] == "field"

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
