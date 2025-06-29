import pytest

from hlslkit.buffer_scan import (
    InvalidStructDictType,
    calculate_hlsl_struct_size,
    calculate_struct_size,
    clean_body,
    extract_matrix_size,
    generate_comparison_table,
    get_field_size,
    parse_field,
)


def test_clean_body_removes_multiline_comments():
    body = """
    /* comment */
    int a;
    float b; // trailing
    /* another */
    """
    result = clean_body(body)
    assert "/*" not in result
    assert "int a;" in result
    assert "float b;" in result


def test_parse_field_invalid_hlsl():
    # Should return a dict with is_unknown_type True for unparseable field
    result = parse_field("not a field", "Struct", True)
    assert isinstance(result, dict)
    assert result.get("is_unknown_type") is True


def test_parse_field_invalid_cpp():
    result = parse_field("not a field", "Struct", False)
    assert isinstance(result, dict)
    assert result.get("is_unknown_type") is True


def test_parse_field_empty_string():
    result = parse_field("", "Struct", True)
    assert result is None


def test_parse_field_whitespace():
    result = parse_field("   ", "Struct", False)
    assert result is None


def test_extract_matrix_size_invalid():
    # Should return None for non-matrix
    assert extract_matrix_size("float") is None
    assert extract_matrix_size("") is None


def test_get_field_size_unknown_type():
    # Should return default size and True for unknown
    size, is_unknown = get_field_size("unknown_type")
    assert size == 4
    assert is_unknown


def test_calculate_struct_size_empty_fields():
    assert calculate_struct_size([]) == 0


def test_calculate_hlsl_struct_size_empty_fields():
    assert calculate_hlsl_struct_size([]) == 0


def test_invalid_struct_dict_type_raises():
    with pytest.raises(InvalidStructDictType):
        raise InvalidStructDictType("invalid")


def test_generate_comparison_table_handles_empty():
    # Should not raise, even if all data is empty
    result = generate_comparison_table(
        hlsl_name="",
        cpp_name="",
        hlsl_data={},
        cpp_data={},
        align_matches=[],
        report={},
        candidates=[],
    )
    assert isinstance(result, str)
