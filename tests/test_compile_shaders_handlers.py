from hlslkit.compile_shaders import ErrorHandler, IssueHandler, WarningHandler


def test_issue_handler_base_class():
    """Test the base IssueHandler class functionality."""
    result = {"file": "/path/to/test.hlsl", "entry": "main", "type": "PSHADER"}

    handler = IssueHandler(result)

    # Test location normalization
    location = handler.normalize_location("/path/to/test.hlsl", "10")
    assert location == "/path/to/test.hlsl:10"

    # Test issue data creation
    issue_data = handler.create_issue_data("E1234", "Test error", location)
    assert issue_data == {
        "code": "E1234",
        "message": "Test error",
        "location": location,
        "context": {"shader_type": "PSHADER", "entry_point": "main"},
    }

    # Test instance tracking
    instances = {}
    handler.add_to_instances(instances, location, issue_data)
    assert location in instances
    assert len(instances[location]) == 1
    assert instances[location][0] == issue_data

    # Test duplicate prevention
    handler.add_to_instances(instances, location, issue_data)
    assert len(instances[location]) == 1  # Should not add duplicate


def test_warning_handler():
    """Test the WarningHandler class functionality."""
    result = {"file": "/path/to/test.hlsl", "entry": "main", "type": "PSHADER"}

    handler = WarningHandler(result)
    baseline_warnings = {}
    suppress_warnings = []
    all_warnings = {}
    new_warnings_dict = {}
    suppressed_count = 0

    # Test warning processing
    warning_line = "test.hlsl(10): warning X1234: Test warning"
    all_warnings, new_warnings_dict, suppressed_count = handler.process(
        warning_line, baseline_warnings, suppress_warnings, all_warnings, new_warnings_dict, suppressed_count
    )

    assert "x1234:test warning" in all_warnings
    assert len(new_warnings_dict) == 1
    assert suppressed_count == 0

    # Test warning suppression
    suppress_warnings = ["X1234"]
    suppress_warnings = [code.lower() for code in suppress_warnings]
    all_warnings = {}
    new_warnings_dict = {}
    suppressed_count = 0

    all_warnings, new_warnings_dict, suppressed_count = handler.process(
        warning_line, baseline_warnings, suppress_warnings, all_warnings, new_warnings_dict, suppressed_count
    )

    assert "x1234:test warning" not in all_warnings
    assert len(new_warnings_dict) == 0
    assert suppressed_count == 1


def test_error_handler():
    """Test the ErrorHandler class functionality."""
    result = {"file": "/path/to/test.hlsl", "entry": "main", "type": "PSHADER"}

    handler = ErrorHandler(result)
    errors = {}

    # Test error processing
    error_line = "test.hlsl(10): error E1234: Test error"
    errors = handler.process(error_line, errors)

    assert "test.hlsl:main" in errors
    assert len(errors["test.hlsl:main"]["instances"]) == 1
    assert "test.hlsl:10" in errors["test.hlsl:main"]["instances"]
    assert len(errors["test.hlsl:main"]["instances"]["test.hlsl:10"]) == 1

    # Test duplicate prevention
    errors = handler.process(error_line, errors)
    assert len(errors["test.hlsl:main"]["instances"]["test.hlsl:10"]) == 1


def test_warning_handler_with_list_format_baseline():
    """Test WarningHandler.process with a baseline warning whose 'instances' is a list (legacy format)."""

    result = {"file": "test.hlsl", "entry": "main", "type": "PSHADER"}
    handler = WarningHandler(result)
    # Simulate a warning line
    warning_line = "test.hlsl(10): warning X1234: Test warning"
    # Baseline warning with 'instances' as a list
    baseline_warnings = {
        "x1234:test warning": {
            "code": "X1234",
            "message": "Test warning",
            "instances": ["test.hlsl:10", "test.hlsl:20"],
        }
    }
    suppress_warnings = []
    all_warnings = {}
    new_warnings_dict = {}
    suppressed_count = 0
    # Should not raise AttributeError and should process correctly
    all_warnings, new_warnings_dict, suppressed_count = handler.process(
        warning_line, baseline_warnings, suppress_warnings, all_warnings, new_warnings_dict, suppressed_count
    )
    # The warning should be present in all_warnings
    assert "x1234:test warning" in all_warnings
    # Verify the warning is properly structured in all_warnings
    warning_data = all_warnings["x1234:test warning"]
    assert warning_data["code"] == "X1234"
    assert warning_data["message"] == "Test warning"
    assert "instances" in warning_data
    assert "test.hlsl:10" in warning_data["instances"]
    assert "entries" in warning_data["instances"]["test.hlsl:10"]
    assert "main" in warning_data["instances"]["test.hlsl:10"]["entries"]
    # The new_warnings_dict should be empty because the warning is not new (already in baseline)
    assert len(new_warnings_dict) == 0
    # The suppressed count should be 0
    assert suppressed_count == 0

    # Test with a new warning not in baseline
    warning_line = "test.hlsl(30): warning X1234: Test warning"
    all_warnings, new_warnings_dict, suppressed_count = handler.process(
        warning_line, baseline_warnings, suppress_warnings, all_warnings, new_warnings_dict, suppressed_count
    )
    # The new warning should be in new_warnings_dict
    assert len(new_warnings_dict) > 0
    # Find the new warning by its location
    new_warning_key = next((k for k in new_warnings_dict if "test.hlsl:30" in k), None)
    assert new_warning_key is not None
    assert new_warning_key in new_warnings_dict
    # Verify the new warning is properly structured
    new_warning = new_warnings_dict[new_warning_key]
    assert new_warning["code"] == "X1234"
    assert new_warning["message"] == "Test warning"
    assert "test.hlsl:30" in new_warning["instances"]
    assert "main" in new_warning["instances"]["test.hlsl:30"]["entries"]


def test_issue_handler_unicode_location():
    """Test IssueHandler with Unicode characters in location."""
    result = {"file": "/path/to/test.hlsl", "entry": "main", "type": "PSHADER"}
    handler = IssueHandler(result)

    # Test with Unicode characters in file path
    location = handler.normalize_location("/path/to/测试.hlsl", "10")
    assert location == "/path/to/测试.hlsl:10"

    issue_data = handler.create_issue_data("E1234", "Test error", location)
    assert issue_data["location"] == location


def test_issue_handler_very_long_message():
    """Test IssueHandler with very long error messages."""
    result = {"file": "/path/to/test.hlsl", "entry": "main", "type": "PSHADER"}
    handler = IssueHandler(result)

    long_message = "A" * 10000  # Very long message
    location = handler.normalize_location("/path/to/test.hlsl", "10")
    issue_data = handler.create_issue_data("E1234", long_message, location)
    assert issue_data["message"] == long_message


def test_issue_handler_special_characters_in_message():
    """Test IssueHandler with special characters in error messages."""
    result = {"file": "/path/to/test.hlsl", "entry": "main", "type": "PSHADER"}
    handler = IssueHandler(result)

    special_message = "Error with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
    location = handler.normalize_location("/path/to/test.hlsl", "10")
    issue_data = handler.create_issue_data("E1234", special_message, location)
    assert issue_data["message"] == special_message


def test_issue_handler_empty_message():
    """Test IssueHandler with empty error messages."""
    result = {"file": "/path/to/test.hlsl", "entry": "main", "type": "PSHADER"}
    handler = IssueHandler(result)

    location = handler.normalize_location("/path/to/test.hlsl", "10")
    issue_data = handler.create_issue_data("E1234", "", location)
    assert issue_data["message"] == ""


def test_warning_handler_malformed_line():
    """Test WarningHandler with malformed warning lines."""
    result = {"file": "/path/to/test.hlsl", "entry": "main", "type": "PSHADER"}
    handler = WarningHandler(result)
    baseline_warnings = {}
    suppress_warnings = []
    all_warnings = {}
    new_warnings_dict = {}
    suppressed_count = 0

    # Test with malformed line (no colon after warning code)
    malformed_line = "test.hlsl(10): warning X1234 Test warning"
    all_warnings, new_warnings_dict, suppressed_count = handler.process(
        malformed_line, baseline_warnings, suppress_warnings, all_warnings, new_warnings_dict, suppressed_count
    )
    # The handler expects proper format with colon, so this should not be processed
    assert "x1234:test warning" not in all_warnings


def test_warning_handler_case_insensitive_suppression():
    """Test WarningHandler with case-insensitive warning suppression."""
    result = {"file": "/path/to/test.hlsl", "entry": "main", "type": "PSHADER"}
    handler = WarningHandler(result)
    baseline_warnings = {}
    suppress_warnings = ["x1234"]  # Lowercase
    all_warnings = {}
    new_warnings_dict = {}
    suppressed_count = 0

    warning_line = "test.hlsl(10): warning X1234: Test warning"  # Uppercase
    all_warnings, new_warnings_dict, suppressed_count = handler.process(
        warning_line, baseline_warnings, suppress_warnings, all_warnings, new_warnings_dict, suppressed_count
    )
    assert "x1234:test warning" not in all_warnings
    assert suppressed_count == 1


def test_error_handler_malformed_line():
    """Test ErrorHandler with malformed error lines."""
    result = {"file": "/path/to/test.hlsl", "entry": "main", "type": "PSHADER"}
    handler = ErrorHandler(result)
    errors = {}

    # Test with malformed line (no colon after error code)
    malformed_line = "test.hlsl(10): error E1234 Test error"
    errors = handler.process(malformed_line, errors)
    # The handler expects proper format with colon, so this should not be processed
    assert "test.hlsl:main" not in errors


def test_error_handler_multiple_errors_same_location():
    """Test ErrorHandler with multiple errors at the same location."""
    result = {"file": "/path/to/test.hlsl", "entry": "main", "type": "PSHADER"}
    handler = ErrorHandler(result)
    errors = {}

    # Add first error
    error_line1 = "test.hlsl(10): error E1234: First error"
    errors = handler.process(error_line1, errors)
    assert len(errors["test.hlsl:main"]["instances"]["test.hlsl:10"]) == 1

    # Add second error at same location
    error_line2 = "test.hlsl(10): error E5678: Second error"
    errors = handler.process(error_line2, errors)
    assert len(errors["test.hlsl:main"]["instances"]["test.hlsl:10"]) == 2


def test_error_handler_different_entry_points():
    """Test ErrorHandler with errors from different entry points."""
    result1 = {"file": "/path/to/test.hlsl", "entry": "main", "type": "PSHADER"}
    result2 = {"file": "/path/to/test.hlsl", "entry": "GetDisplacedPosition", "type": "PSHADER"}

    handler1 = ErrorHandler(result1)
    handler2 = ErrorHandler(result2)
    errors = {}

    # Add error from first entry point
    error_line1 = "test.hlsl(10): error E1234: First error"
    errors = handler1.process(error_line1, errors)
    assert "test.hlsl:main" in errors

    # Add error from second entry point
    error_line2 = "test.hlsl(20): error E5678: Second error"
    errors = handler2.process(error_line2, errors)
    # The handler converts entry points to lowercase
    assert "test.hlsl:getdisplacedposition" in errors
    assert "test.hlsl:main" in errors  # First entry point should still be there
