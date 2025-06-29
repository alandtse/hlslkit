"""Tests for struct analysis and comparison functionality."""

from hlslkit.buffer_scan import (
    AnalysisLink,
    InvalidStructDictType,
    StructAnalyzer,
    StructCandidate,
    StructMatch,
    _compute_alignment_report,
    align_structs,
    are_fields_equivalent,
    calculate_hlsl_struct_size,
    calculate_struct_size,
    compute_match_score,
    compute_name_similarity,
    compute_struct_alignment,
    count_field_differences,
    fuzzy_lcs,
    generate_comparison_table,
    get_field_similarity,
    get_struct_signature,
    normalize_array_types,
)


class TestStructSizeCalculation:
    """Test struct size calculation functions."""

    def test_calculate_struct_size_basic(self):
        """Test calculating struct size with basic fields."""
        fields = [
            {"name": "a", "type": "int", "size": 4},
            {"name": "b", "type": "float", "size": 4},
        ]
        result = calculate_struct_size(fields)
        assert result == 8

    def test_calculate_struct_size_with_alignment(self):
        """Test calculating struct size with 16-byte alignment."""
        fields = [
            {"name": "a", "type": "int", "size": 4},
            {"name": "b", "type": "float", "size": 4},
        ]
        result = calculate_struct_size(fields, align_to_16=True)
        assert result == 32  # 16-byte alignment for each field

    def test_calculate_struct_size_empty(self):
        """Test calculating struct size with empty fields."""
        result = calculate_struct_size([])
        assert result == 0

    def test_calculate_hlsl_struct_size(self):
        """Test calculating HLSL struct size."""
        fields = [
            {"name": "a", "type": "int", "size": 4},
            {"name": "b", "type": "float", "size": 4},
        ]
        result = calculate_hlsl_struct_size(fields)
        assert result == 16  # HLSL uses 16-byte alignment


class TestFieldComparison:
    """Test field comparison functions."""

    def test_are_fields_equivalent_same(self):
        """Test field equivalence with identical fields."""
        cpp_field = {"name": "test", "type": "int", "size": 4}
        hlsl_field = {"name": "test", "type": "int", "size": 4}
        result = are_fields_equivalent(cpp_field, hlsl_field)
        assert result is True

    def test_are_fields_equivalent_different(self):
        """Test field equivalence with different fields."""
        cpp_field = {"name": "test", "type": "int", "size": 4}
        hlsl_field = {"name": "other", "type": "float", "size": 4}
        result = are_fields_equivalent(cpp_field, hlsl_field)
        assert result is False

    def test_normalize_array_types(self):
        """Test array type normalization."""
        hlsl_type = "int[10]"
        cpp_type = "int[10]"
        result_hlsl, result_cpp = normalize_array_types(hlsl_type, cpp_type)
        assert result_hlsl == "int[10]"  # Array notation is preserved
        assert result_cpp == "int[10]"

    def test_get_field_similarity(self):
        """Test field similarity calculation."""
        cpp_field = {"name": "test_field", "type": "int", "size": 4}
        hlsl_field = {"name": "test_field", "type": "int", "size": 4}
        name_sim, type_sim, is_equivalent = get_field_similarity(cpp_field, hlsl_field)
        assert 0 <= name_sim <= 1
        assert 0 <= type_sim <= 1
        assert isinstance(is_equivalent, bool)


class TestStructAlignment:
    """Test struct alignment functions."""

    def test_compute_struct_alignment(self):
        """Test computing struct alignment."""
        cpp_data = {"fields": [{"name": "a", "type": "int", "size": 4}], "size": 4, "file": "test.cpp", "line": 1}
        hlsl_data = {"fields": [{"name": "a", "type": "int", "size": 4}], "size": 4, "file": "test.hlsl", "line": 1}
        score, align_matches, report = compute_struct_alignment(cpp_data, hlsl_data)
        assert isinstance(score, float)
        assert isinstance(align_matches, list)
        assert isinstance(report, dict)

    def test_compute_match_score(self):
        """Test computing match score."""
        hlsl_fields = [{"name": "a", "type": "int", "size": 4}]
        cpp_fields = [{"name": "a", "type": "int", "size": 4}]
        score = compute_match_score("TestStruct", "TestStruct", hlsl_fields, cpp_fields)
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_compute_match_score_with_lcs(self):
        """Test computing match score with LCS pairs."""
        hlsl_fields = [{"name": "a", "type": "int", "size": 4}]
        cpp_fields = [{"name": "a", "type": "int", "size": 4}]
        lcs_pairs = [(0, 0)]
        score = compute_match_score("TestStruct", "TestStruct", hlsl_fields, cpp_fields, lcs_pairs)
        assert isinstance(score, float)

    def test_compute_alignment_report(self):
        """Test computing alignment report."""
        cpp_fields = [{"name": "a", "type": "int", "size": 4}]
        hlsl_fields = [{"name": "a", "type": "int", "size": 4}]
        cpp_data = {"size": 4}
        hlsl_data = {"size": 4}
        align_matches, report = _compute_alignment_report(cpp_fields, hlsl_fields, cpp_data, hlsl_data)
        assert isinstance(align_matches, list)
        assert isinstance(report, dict)

    def test_align_structs(self):
        """Test aligning structs."""
        cpp_data = {"fields": [{"name": "a", "type": "int", "size": 4}], "size": 4}
        hlsl_data = {"fields": [{"name": "a", "type": "int", "size": 4}], "size": 4}
        score, align_matches, report = align_structs(cpp_data, hlsl_data)
        assert isinstance(score, float)
        assert isinstance(align_matches, list)
        assert isinstance(report, dict)


class TestFuzzyLCS:
    """Test fuzzy LCS functionality."""

    def test_fuzzy_lcs_empty(self):
        """Test fuzzy LCS with empty fields."""
        result = fuzzy_lcs([], [])
        assert result == []

    def test_fuzzy_lcs_similar_names(self):
        """Test fuzzy LCS with similar field names."""
        hlsl_fields = [{"name": "test_field", "type": "int"}]
        cpp_fields = [{"name": "test_field", "type": "int"}]
        result = fuzzy_lcs(hlsl_fields, cpp_fields)
        assert isinstance(result, list)

    def test_fuzzy_lcs_different_names(self):
        """Test fuzzy LCS with different field names."""
        hlsl_fields = [{"name": "field1", "type": "int"}]
        cpp_fields = [{"name": "field2", "type": "int"}]
        result = fuzzy_lcs(hlsl_fields, cpp_fields)
        assert isinstance(result, list)


class TestNameSimilarity:
    """Test name similarity functions."""

    def test_compute_name_similarity_identical(self):
        """Test name similarity with identical names."""
        result = compute_name_similarity("test", "test")
        assert result == 1.0

    def test_compute_name_similarity_similar(self):
        """Test name similarity with similar names."""
        result = compute_name_similarity("test_field", "testField")
        assert 0 < result < 1.0

    def test_compute_name_similarity_different(self):
        """Test name similarity with different names."""
        result = compute_name_similarity("field1", "field2")
        assert 0 <= result < 1.0


class TestFieldDifferences:
    """Test field difference counting."""

    def test_count_field_differences(self):
        """Test counting field differences."""
        align_matches = [
            ({"name": "a", "type": "int"}, {"name": "a", "type": "int"}),
            ({"name": "b", "type": "int"}, None),
            (None, {"name": "c", "type": "int"}),
        ]
        missing, extra, mismatched = count_field_differences(align_matches)
        assert isinstance(missing, int)
        assert isinstance(extra, int)
        assert isinstance(mismatched, int)


class TestStructSignature:
    """Test struct signature generation."""

    def test_get_struct_signature(self):
        """Test getting struct signature."""
        fields = [
            {"name": "a", "type": "int", "size": 4},
            {"name": "b", "type": "float", "size": 4},
        ]
        result = get_struct_signature(fields)
        assert isinstance(result, str)
        assert len(result) > 0


class TestStructAnalyzer:
    """Test StructAnalyzer class."""

    def test_init(self):
        """Test StructAnalyzer initialization."""
        hlsl_structs = {"TestStruct": [{"fields": [], "file": "test.hlsl", "line": 1}]}
        cpp_structs = {"TestStruct": [{"fields": [], "file": "test.cpp", "line": 1}]}
        analyzer = StructAnalyzer(hlsl_structs, cpp_structs)
        assert analyzer.hlsl_structs == hlsl_structs
        assert analyzer.cpp_structs == cpp_structs

    def test_add_buffer_location(self):
        """Test adding buffer location."""
        analyzer = StructAnalyzer({}, {})
        analyzer.add_buffer_location("test.hlsl", "TestBuffer", 10)
        key = "test.hlsl:testbuffer"
        assert key in analyzer.buffer_locations

    def test_get_buffer_location(self):
        """Test getting buffer location."""
        analyzer = StructAnalyzer({}, {})
        analyzer.add_buffer_location("test.hlsl", "TestBuffer", 10)
        result = analyzer.get_buffer_location("test.hlsl", "TestBuffer", 10)
        assert result == ("test.hlsl", "TestBuffer")

    def test_get_buffer_location_not_found(self):
        """Test getting buffer location when not found."""
        analyzer = StructAnalyzer({}, {})
        result = analyzer.get_buffer_location("test.hlsl", "TestBuffer", 10)
        assert result is None

    def test_is_composite_buffer_true(self):
        """Test composite buffer detection when true."""
        analyzer = StructAnalyzer({}, {})
        struct_data = {"is_cbuffer": True, "fields": [{"name": "field", "type": "UserStruct", "size": 4}]}
        # Mock that UserStruct exists in hlsl_structs
        analyzer.hlsl_structs = {"UserStruct": []}
        result = analyzer._is_composite_buffer(struct_data)
        assert result is True

    def test_is_composite_buffer_false(self):
        """Test composite buffer detection when false."""
        analyzer = StructAnalyzer({}, {})
        struct_data = {"is_cbuffer": False, "fields": [{"name": "field", "type": "int", "size": 4}]}
        result = analyzer._is_composite_buffer(struct_data)
        assert result is False

    def test_get_nested_fields(self):
        """Test getting nested fields."""
        analyzer = StructAnalyzer({}, {})
        struct_data = {"fields": [{"name": "field", "type": "int", "size": 4}]}
        result = analyzer.get_nested_fields(struct_data)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_get_field_name(self):
        """Test getting field name."""
        analyzer = StructAnalyzer({}, {})
        field = {"name": "test_field", "type": "int"}
        result = analyzer.get_field_name(field)
        assert result == "test_field"

    def test_find_struct_candidates(self):
        """Test finding struct candidates."""
        analyzer = StructAnalyzer({}, {})
        hlsl_data = {"fields": [], "file": "test.hlsl", "line": 1}
        hlsl_fields = []
        matched_cpp_structs = set()
        result = analyzer.find_struct_candidates("TestStruct", hlsl_data, hlsl_fields, matched_cpp_structs)
        assert isinstance(result, list)

    def test_find_best_match_from_candidates(self):
        """Test finding best match from candidates."""
        analyzer = StructAnalyzer({}, {})
        hlsl_data = {"fields": [], "file": "test.hlsl", "line": 1}
        hlsl_fields = []
        candidates = []
        result = analyzer._find_best_match_from_candidates("TestStruct", hlsl_data, hlsl_fields, candidates)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_is_match_good_enough(self):
        """Test match quality assessment."""
        analyzer = StructAnalyzer({}, {})
        score = 0.8
        report = {"field_matches": 5, "total_fields": 5}
        candidates = []
        result = analyzer._is_match_good_enough(score, report, candidates)
        assert isinstance(result, bool)

    def test_compare_all_structs(self):
        """Test comparing all structs."""
        analyzer = StructAnalyzer({}, {})
        result_map = {}
        result = analyzer.compare_all_structs(result_map)
        assert isinstance(result, dict)

    def test_update_result_map(self):
        """Test updating result map."""
        analyzer = StructAnalyzer({}, {})
        result_map = {}
        analyzer.update_result_map(result_map)
        # Should not raise an exception


class TestComparisonTable:
    """Test comparison table generation."""

    def test_generate_comparison_table(self):
        """Test generating comparison table."""
        hlsl_data = {"fields": [{"name": "a", "type": "int", "size": 4}], "file": "test.hlsl", "line": 1}
        cpp_data = {"fields": [{"name": "a", "type": "int", "size": 4}], "file": "test.cpp", "line": 1}
        align_matches = [({"name": "a", "type": "int"}, {"name": "a", "type": "int"})]
        report = {"field_matches": 1, "total_fields": 1}
        candidates = []
        result = generate_comparison_table(
            "TestStruct", "TestStruct", hlsl_data, cpp_data, align_matches, report, candidates
        )
        assert isinstance(result, str)
        assert len(result) > 0


class TestInvalidStructDictType:
    """Test InvalidStructDictType exception."""

    def test_invalid_struct_dict_type(self):
        """Test InvalidStructDictType exception."""
        obj = "invalid"
        exception = InvalidStructDictType(obj)
        assert "Expected StructDict" in str(exception)


class TestStructMatch:
    """Test StructMatch dataclass."""

    def test_struct_match_creation(self):
        """Test creating StructMatch."""
        match = StructMatch(
            hlsl_name="TestStruct",
            hlsl_file="test.hlsl",
            hlsl_line=1,
            cpp_name="TestStruct",
            cpp_file="test.cpp",
            cpp_line=1,
            score=0.8,
            align_matches=[],
            report={},
            candidates=[],
        )
        assert match.hlsl_name == "TestStruct"
        assert match.cpp_name == "TestStruct"
        assert match.score == 0.8

    def test_struct_match_is_matched_true(self):
        """Test is_matched property when true."""
        match = StructMatch(
            hlsl_name="TestStruct",
            hlsl_file="test.hlsl",
            hlsl_line=1,
            cpp_name="TestStruct",
            cpp_file="test.cpp",
            cpp_line=1,
            score=0.8,
            align_matches=[],
            report={},
            candidates=[],
        )
        assert match.is_matched is True

    def test_struct_match_is_matched_false(self):
        """Test is_matched property when false."""
        match = StructMatch(
            hlsl_name="TestStruct",
            hlsl_file="test.hlsl",
            hlsl_line=1,
            cpp_name="",
            cpp_file="test.cpp",
            cpp_line=1,
            score=0.0,
            align_matches=[],
            report={},
            candidates=[],
        )
        assert match.is_matched is False


class TestStructCandidate:
    """Test StructCandidate dataclass."""

    def test_struct_candidate_creation(self):
        """Test creating StructCandidate."""
        candidate = StructCandidate(name="TestStruct", data={"fields": []}, score=0.8, align_matches=[], report={})
        assert candidate.name == "TestStruct"
        assert candidate.score == 0.8


class TestAnalysisLink:
    """Test AnalysisLink dataclass."""

    def test_analysis_link_creation(self):
        """Test creating AnalysisLink."""
        link = AnalysisLink(
            link="test_link",
            is_match=True,
            cpp_name="TestStruct",
            cpp_file="test.cpp",
            cpp_line=1,
            score=0.8,
            status="Matched",
        )
        assert link.link == "test_link"
        assert link.is_match is True
        assert link.cpp_name == "TestStruct"
