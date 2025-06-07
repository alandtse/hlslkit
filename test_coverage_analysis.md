# hlslkit buffer_scan Module - Test Coverage Analysis

## Current Coverage Summary

-   **Total Coverage**: 42% (515 out of 1131 statements covered)
-   **Tests Passing**: 58/58
-   **Branch Coverage**: 58 partial branches covered out of 578 total
-   **Missing Lines**: 616 statements not covered

## Test Coverage Assessment

### ✅ Well-Covered Areas (Good Test Coverage)

1. **Basic Utility Functions**

    - `create_link()` - GitHub URL generation
    - `finditer_with_line_numbers()` - Pattern matching with line numbers
    - `capture_pattern()` - Pattern capture functionality
    - `get_hlsl_types()` - HLSL type mapping
    - `get_defines_list()` - Shader defines list

2. **Field Parsing Functions**

    - `parse_hlsl_field()` - HLSL field parsing (basic, arrays, packoffset)
    - `parse_cpp_field()` - C++ field parsing (basic, arrays)
    - Edge cases for empty/malformed inputs

3. **Struct Extraction Functions**

    - `extract_hlsl_structs()` - HLSL struct extraction (cbuffers, regular structs)
    - `extract_cpp_structs()` - C++ struct extraction
    - `extract_structs()` - Wrapper function
    - Edge cases for empty inputs, malformed syntax, comments

4. **Basic StructAnalyzer Operations**

    - Initialization and basic methods
    - `get_field_name()` and `get_nested_fields()`
    - Basic struct comparison functionality

5. **FileScanner Class**

    - Initialization and basic scanning functionality
    - Basic file system operations

6. **Struct Alignment Functions**
    - `align_structs()` - Basic perfect match and no match scenarios
    - Edge cases with empty structs

## ❌ Major Coverage Gaps (Need Additional Tests)

### 1. **File Preprocessing with PCPP** (Lines 261-420) - HIGH PRIORITY

**Current Coverage**: 0%
**Missing Functionality**:

-   PCPP preprocessor integration
-   Include path resolution (`package/Shaders`, `Common`, `features/*/Shaders`)
-   Preprocessor define handling
-   Line mapping between preprocessed and original content
-   File validation and error handling

**Recommended Tests**:

```python
def test_process_file_with_defines_basic()
def test_process_file_with_defines_include_paths()
def test_process_file_with_defines_line_mapping()
def test_process_file_with_defines_invalid_file()
def test_process_file_with_defines_preprocessor_errors()
```

### 2. **`.gitignore` Parsing** (Lines 179-194) - MEDIUM PRIORITY

**Current Coverage**: 0%
**Missing Functionality**:

-   `.gitignore` file parsing with pathspec
-   Directory exclusion pattern extraction
-   Error handling for malformed `.gitignore` files
-   Default exclusion fallback

**Recommended Tests**:

```python
def test_get_excluded_dirs_from_gitignore_valid_file()
def test_get_excluded_dirs_from_gitignore_missing_file()
def test_get_excluded_dirs_from_gitignore_malformed_file()
def test_get_excluded_dirs_from_gitignore_no_pathspec()
```

### 3. **Complex Struct Analysis** (Lines 1408-1608) - HIGH PRIORITY

**Current Coverage**: 0%
**Missing Functionality**:

-   Advanced struct comparison algorithms
-   Struct candidate finding (`find_struct_candidates()`)
-   Buffer location detection (`get_buffer_location()`)
-   Complex alignment scoring
-   Multi-struct matching scenarios

**Recommended Tests**:

```python
def test_find_struct_candidates_multiple_matches()
def test_find_struct_candidates_no_matches()
def test_get_buffer_location_various_file_types()
def test_complex_struct_alignment_scoring()
def test_multi_struct_comparison_scenarios()
```

### 4. **Composite Buffer Processing** (Lines 1876-1920) - MEDIUM PRIORITY

**Current Coverage**: Partial
**Missing Functionality**:

-   `_process_composite_buffer()` complete logic
-   Composite buffer detection edge cases
-   Nested struct resolution
-   Composite buffer vs regular buffer differentiation

**Recommended Tests**:

```python
def test_process_composite_buffer_complex_nesting()
def test_process_composite_buffer_missing_dependencies()
def test_composite_buffer_detection_edge_cases()
def test_composite_vs_regular_buffer_classification()
```

### 5. **Report Generation & Table Formatting** (Lines 2012-2150) - LOW PRIORITY

**Current Coverage**: 0%
**Missing Functionality**:

-   Comparison table generation
-   Report formatting and output
-   Table styling and alignment
-   Output file writing

**Recommended Tests**:

```python
def test_generate_comparison_tables()
def test_format_comparison_report()
def test_table_output_formatting()
def test_report_file_output()
```

### 6. **CLI Interface** (Lines 2271-2475) - LOW PRIORITY

**Current Coverage**: 0%
**Missing Functionality**:

-   Command line argument parsing
-   Main function execution
-   CLI error handling
-   Input validation

**Recommended Tests**:

```python
def test_main_function_with_valid_args()
def test_main_function_invalid_directory()
def test_cli_argument_parsing()
def test_cli_error_handling()
```

### 7. **Error Handling & Edge Cases** (Scattered) - MEDIUM PRIORITY

**Missing Functionality**:

-   File system error handling
-   Invalid input validation
-   Exception propagation
-   Logging integration testing

## 🎯 Recommended Test Priorities

### Phase 1: Core Functionality (Target: +15% coverage)

1. **File preprocessing tests** - Most critical missing functionality
2. **Complex struct comparison tests** - Core algorithm testing
3. **Error handling tests** - Robustness improvements

### Phase 2: Advanced Features (Target: +10% coverage)

1. **Composite buffer processing tests**
2. **`.gitignore` parsing tests**
3. **Buffer location detection tests**

### Phase 3: Output & Interface (Target: +5% coverage)

1. **Report generation tests**
2. **CLI interface tests**
3. **Table formatting tests**

## 🔍 Specific Test Scenarios to Add

### Integration Test Scenarios

1. **End-to-end struct matching** with real HLSL/C++ file pairs
2. **Large file processing** with multiple includes and complex preprocessing
3. **Multi-feature analysis** with different shader define combinations
4. **Cross-platform path handling** (Windows/Unix path differences)

### Edge Case Scenarios

1. **Malformed shader files** with syntax errors
2. **Circular include dependencies**
3. **Very large struct definitions** (performance testing)
4. **Unicode and special characters** in file paths and content
5. **Memory pressure scenarios** with large file sets

### Error Recovery Scenarios

1. **Partial file processing failures**
2. **Missing dependency recovery**
3. **Invalid preprocessor define handling**
4. **Graceful degradation** when external tools fail

## 📊 Expected Coverage Improvement

With the recommended additional tests:

-   **Current**: 42% coverage
-   **Phase 1 Target**: 57% coverage (+15%)
-   **Phase 2 Target**: 67% coverage (+10%)
-   **Phase 3 Target**: 72% coverage (+5%)
-   **Total Potential**: ~72% coverage

This would provide comprehensive coverage of the core functionality while maintaining maintainable test code.
