"""Tests for utility functions."""

from hlslkit.compile_shaders import flatten_defines, normalize_path


def test_normalize_path_with_shaders():
    """Test path normalization with Shaders directory."""
    assert normalize_path("C:/Projects/Shaders/src/test.hlsl") == "src/test.hlsl"
    assert normalize_path("D:\\Games\\Skyrim\\Shaders\\water.hlsl") == "water.hlsl"
    assert normalize_path("/home/user/skyrim-community-shaders/build/all/aio/Shaders/water.hlsl") == "water.hlsl"


def test_normalize_path_no_shaders():
    """Test normalize_path without Shaders in path."""
    path = "C:/Projects/src/test.hlsl"
    expected = "C:/Projects/src/test.hlsl"
    assert normalize_path(path) == expected


def test_normalize_path_with_backslashes():
    """Test normalize_path with backslashes in path."""
    path = "C:\\Projects\\Shaders\\src\\test.hlsl"
    expected = "src/test.hlsl"
    assert normalize_path(path) == expected


def test_normalize_path_with_mixed_slashes():
    """Test normalize_path with mixed slashes in path."""
    path = "C:/Projects\\Shaders/src\\test.hlsl"
    expected = "src/test.hlsl"
    assert normalize_path(path) == expected


def test_normalize_path_yaml_style():
    """Test path normalization with YAML-style paths."""
    assert normalize_path("common/color.hlsli") == "common/color.hlsli"
    assert normalize_path("shaders/common/color.hlsli") == "common/color.hlsli"
    assert normalize_path("Shaders/common/color.hlsli") == "common/color.hlsli"
    assert normalize_path("C:/Projects/Shaders/common/color.hlsli") == "common/color.hlsli"


def test_normalize_path_ending_in_shaders():
    """Test normalize_path with a path ending in 'Shaders' (no trailing slash)."""
    assert normalize_path("C:/Game/Content/Shaders") == ""
    assert normalize_path("C:/Game/Content/Shaders/") == ""
    assert normalize_path("Shaders") == ""
    assert normalize_path("Shaders/") == ""


def test_flatten_defines():
    """Test flatten_defines function."""
    defines = [["A=1", "B"], ["B", "C=2"], ["D"]]
    result = flatten_defines(defines)
    assert result == ["A=1", "B", "C=2", "D"]


def test_flatten_defines_with_duplicates():
    """Test flatten_defines with duplicate defines."""
    defines = [["A=1", "B"], ["B", "A=2"], ["C"]]
    result = flatten_defines(defines)
    assert result == ["A=1", "B", "A=2", "C"]  # Duplicate "B" removed, but "A=1" and "A=2" are different values


def test_flatten_defines_empty():
    """Test flatten_defines with empty input."""
    defines = []
    result = flatten_defines(defines)
    assert result == []


def test_flatten_defines_invalid():
    """Test flatten_defines with None in input."""
    defines = [["A=1"], None, ["B"]]
    result = flatten_defines(defines)
    assert result == ["A=1", None, "B"]  # Matches actual behavior


def test_normalize_path_empty_string():
    """Test normalize_path with empty string."""
    assert normalize_path("") == ""


def test_normalize_path_none_input():
    """Test normalize_path with None input."""
    assert normalize_path(None) == ""


def test_normalize_path_unicode_characters():
    """Test normalize_path with unicode characters."""
    path = "C:/Projects/Shaders/测试/文件.hlsl"
    expected = "测试/文件.hlsl"
    assert normalize_path(path) == expected


def test_normalize_path_very_long_path():
    """Test normalize_path with very long path."""
    long_path = "C:/" + "a/" * 100 + "Shaders/file.hlsl"
    expected = "file.hlsl"
    assert normalize_path(long_path) == expected


def test_normalize_path_special_characters():
    """Test normalize_path with special characters."""
    path = "C:/Projects/Shaders/path with spaces/file.hlsl"
    expected = "path with spaces/file.hlsl"
    assert normalize_path(path) == expected


def test_normalize_path_multiple_shaders_occurrences():
    """Test normalize_path with multiple 'Shaders' occurrences."""
    path = "C:/Shaders/Projects/Shaders/src/test.hlsl"
    expected = "src/test.hlsl"  # Should use the last occurrence
    assert normalize_path(path) == expected


def test_normalize_path_case_variations():
    """Test normalize_path with different case variations."""
    assert normalize_path("C:/Projects/SHADERS/src/test.hlsl") == "src/test.hlsl"
    assert normalize_path("C:/Projects/shaders/src/test.hlsl") == "src/test.hlsl"
    assert normalize_path("C:/Projects/ShAdErS/src/test.hlsl") == "src/test.hlsl"


def test_normalize_path_only_filename():
    """Test normalize_path with only filename."""
    assert normalize_path("test.hlsl") == "test.hlsl"
    assert normalize_path("Shaders/test.hlsl") == "test.hlsl"


def test_normalize_path_relative_paths():
    """Test normalize_path with relative paths."""
    assert normalize_path("./Shaders/src/test.hlsl") == "src/test.hlsl"
    assert normalize_path("../Shaders/src/test.hlsl") == "src/test.hlsl"


def test_flatten_defines_nested_lists():
    """Test flatten_defines with deeply nested lists."""
    defines = [["A=1", ["B=2", ["C=3"]]], ["D=4"]]
    result = flatten_defines(defines)
    assert result == ["A=1", "B=2", "C=3", "D=4"]


def test_flatten_defines_mixed_types():
    """Test flatten_defines with mixed types."""
    defines = [["A=1"], "B=2", ["C=3"], None, ["D=4"]]
    result = flatten_defines(defines)
    assert result == ["A=1", "B=2", "C=3", None, "D=4"]


def test_flatten_defines_very_large_input():
    """Test flatten_defines with very large input."""
    defines = [["A" + str(i)] for i in range(1000)]
    result = flatten_defines(defines)
    assert len(result) == 1000
    assert result[0] == "A0"
    assert result[999] == "A999"


def test_flatten_defines_empty_nested():
    """Test flatten_defines with empty nested lists."""
    defines = [["A=1"], [], ["B=2"], [["C=3"], []], ["D=4"]]
    result = flatten_defines(defines)
    assert result == ["A=1", "B=2", "C=3", "D=4"]


def test_flatten_defines_single_element():
    """Test flatten_defines with single element."""
    defines = ["A=1"]
    result = flatten_defines(defines)
    assert result == ["A=1"]


def test_flatten_defines_complex_structure():
    """Test flatten_defines with complex nested structure."""
    defines = [[["A=1", "B=2"], ["C=3"]], [["D=4"], [["E=5", "F=6"], ["G=7"]]], ["H=8"]]
    result = flatten_defines(defines)
    expected = ["A=1", "B=2", "C=3", "D=4", "E=5", "F=6", "G=7", "H=8"]
    assert result == expected
