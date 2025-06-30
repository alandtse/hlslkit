import contextlib
import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import yaml

from hlslkit.generate_shader_defines import (
    CompilationTask,
    collect_tasks,
    count_compiling_lines,
    count_log_blocks,
    generate_yaml_data,
    get_shader_type_from_entry,
    normalize_path,
    optimize_anchor_deduplication,
    parse_log,
    parse_timestamp,
    populate_configs,
    save_yaml,
)


def test_normalize_path_with_shaders():
    """Test normalize_path with Shaders in path."""
    path = "C:/Projects/Shaders/src/test.hlsl"
    expected = "src/test.hlsl"
    assert normalize_path(path) == expected


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


def test_get_shader_type_from_entry():
    """Test get_shader_type_from_entry function."""
    assert get_shader_type_from_entry("main:vertex:1234") == "VSHADER"
    assert get_shader_type_from_entry("main:pixel:5678") == "PSHADER"
    assert get_shader_type_from_entry("main:compute:9012") == "CSHADER"
    assert get_shader_type_from_entry("main:unknown:3456") == "UNKNOWN"


@patch("hlslkit.generate_shader_defines.open")
def test_parse_log(mock_open):
    """Test parse_log function with a sample log."""
    mock_file = MagicMock()
    mock_file.__enter__.return_value.readlines.return_value = [
        "[00:45:10.539] [35768] [D] Compiling Data/Shaders/Sky.hlsl Sky:Vertex:0 to VSHADER D3DCOMPILE_SKIP_OPTIMIZATION D3DCOMPILE_DEBUG OCCLUSION SCREEN_SPACE_SHADOWS WETNESS_EFFECTS LIGHT_LIMIT_FIX DYNAMIC_CUBEMAPS CLOUD_SHADOWS WATER_EFFECTS SSS TERRAIN_SHADOWS SKYLIGHTING TERRAIN_BLENDING LOD_BLENDING ISL IBL",
        "[00:45:10.540] [35768] [D] Shader logs:",
        "Data/Shaders/Sky.hlsl(10): warning X3206: implicit truncation",
        "[00:45:10.541] [35768] [D] Compiled shader Sky:Vertex:0",
    ]
    mock_open.return_value = mock_file
    shader_configs, warnings, errors = parse_log("log.txt")
    assert "Sky.hlsl" in shader_configs
    assert shader_configs["Sky.hlsl"]["VSHADER"] == [
        {
            "entry": "Sky:Vertex:0",
            "defines": [
                "CLOUD_SHADOWS",
                "D3DCOMPILE_DEBUG",
                "D3DCOMPILE_SKIP_OPTIMIZATION",
                "DYNAMIC_CUBEMAPS",
                "IBL",
                "ISL",
                "LIGHT_LIMIT_FIX",
                "LOD_BLENDING",
                "OCCLUSION",
                "SCREEN_SPACE_SHADOWS",
                "SKYLIGHTING",
                "SSS",
                "TERRAIN_BLENDING",
                "TERRAIN_SHADOWS",
                "VSHADER",
                "WATER_EFFECTS",
                "WETNESS_EFFECTS",
            ],
        }
    ]
    assert "x3206:implicit truncation" in warnings
    assert warnings["x3206:implicit truncation"]["instances"]["sky.hlsl:10"]["entries"] == ["Sky:Vertex:0"]
    assert errors == {}


@patch("hlslkit.generate_shader_defines.open")
def test_parse_log_with_x4000_warning(mock_open):
    """Test parse_log with X4000 warning."""
    mock_file = MagicMock()
    mock_file.__enter__.return_value.readlines.return_value = [
        "[00:45:10.544] [37824] [D] Compiling Data/Shaders/RunGrass.hlsl Grass:Vertex:4 to VSHADER D3DCOMPILE_DEBUG WATER_EFFECTS GRASS_COLLISION",
        "[00:45:10.544] [37824] [D] Shader logs:",
        "GrassCollision\\GrassCollision.hlsli(52,3): warning X4000: use of potentially uninitialized variable (GrassCollision::GetDisplacedPosition)",
        "[00:45:10.544] [37824] [D] Compiled shader Grass:Vertex:4",
    ]
    mock_open.return_value = mock_file
    shader_configs, warnings, errors = parse_log("log.txt")
    assert "RunGrass.hlsl" in shader_configs
    assert shader_configs["RunGrass.hlsl"]["VSHADER"] == [
        {"entry": "Grass:Vertex:4", "defines": ["D3DCOMPILE_DEBUG", "GRASS_COLLISION", "VSHADER", "WATER_EFFECTS"]}
    ]
    assert "x4000:use of potentially uninitialized variable (grasscollision::getdisplacedposition)" in warnings
    assert warnings["x4000:use of potentially uninitialized variable (grasscollision::getdisplacedposition)"][
        "instances"
    ]["grasscollision/grasscollision.hlsli:52,3"]["entries"] == ["Grass:Vertex:4"]
    assert errors == {}


@patch("hlslkit.generate_shader_defines.open")
def test_parse_log_with_forward_slashes(mock_open):
    """Test parse_log with forward slashes in warning path."""
    mock_file = MagicMock()
    mock_file.__enter__.return_value.readlines.return_value = [
        "[00:45:10.544] [37824] [D] Compiling Data/Shaders/RunGrass.hlsl Grass:Vertex:4 to VSHADER D3DCOMPILE_DEBUG WATER_EFFECTS GRASS_COLLISION",
        "[00:45:10.544] [37824] [D] Shader logs:",
        "GrassCollision/GrassCollision.hlsli(52): warning X4000: use of potentially uninitialized variable (GrassCollision::GetDisplacedPosition)",
        "[00:45:10.544] [37824] [D] Compiled shader Grass:Vertex:4",
    ]
    mock_open.return_value = mock_file
    shader_configs, warnings, errors = parse_log("log.txt")
    assert "RunGrass.hlsl" in shader_configs
    assert shader_configs["RunGrass.hlsl"]["VSHADER"] == [
        {"entry": "Grass:Vertex:4", "defines": ["D3DCOMPILE_DEBUG", "GRASS_COLLISION", "VSHADER", "WATER_EFFECTS"]}
    ]
    assert "x4000:use of potentially uninitialized variable (grasscollision::getdisplacedposition)" in warnings
    assert warnings["x4000:use of potentially uninitialized variable (grasscollision::getdisplacedposition)"][
        "instances"
    ]["grasscollision/grasscollision.hlsli:52"]["entries"] == ["Grass:Vertex:4"]
    assert errors == {}


@patch("hlslkit.generate_shader_defines.open")
def test_parse_log_with_conflicting_defines(mock_open):
    """Test parse_log with conflicting defines."""
    mock_file = MagicMock()
    mock_file.__enter__.return_value.readlines.return_value = [
        "[00:45:10.555] [1268] [D] Compiling Data/Shaders/RunGrass.hlsl Grass:Vertex:10007 to VSHADER D3DCOMPILE_DEBUG WATER_EFFECTS GRASS_COLLISION WATER_EFFECTS",
        "[00:45:10.555] [1268] [D] Compiled shader Grass:Vertex:10007",
    ]
    mock_open.return_value = mock_file
    shader_configs, warnings, errors = parse_log("log.txt")
    assert shader_configs["RunGrass.hlsl"]["VSHADER"] == [
        {
            "entry": "Grass:Vertex:10007",
            "defines": ["D3DCOMPILE_DEBUG", "GRASS_COLLISION", "VSHADER", "WATER_EFFECTS", "WATER_EFFECTS"],
        }
    ]


@patch("hlslkit.generate_shader_defines.open")
def test_parse_log_empty(mock_open):
    """Test parse_log with empty log file."""
    mock_file = MagicMock()
    mock_file.__enter__.return_value.readlines.return_value = []
    mock_open.return_value = mock_file
    shader_configs, warnings, errors = parse_log("log.txt")
    assert shader_configs == {}
    assert warnings == {}
    assert errors == {}


@patch("hlslkit.generate_shader_defines.open")
def test_parse_log_malformed(mock_open):
    """Test parse_log with malformed log line."""
    mock_file = MagicMock()
    mock_file.__enter__.return_value.readlines.return_value = ["[invalid log line]"]
    mock_open.return_value = mock_file
    shader_configs, warnings, errors = parse_log("log.txt")
    assert shader_configs == {}
    assert warnings == {}
    assert errors == {}


@patch("hlslkit.generate_shader_defines.open")
def test_parse_log_with_error(mock_open):
    """Test parse_log with compilation error."""
    mock_file = MagicMock()
    mock_file.__enter__.return_value.readlines.return_value = [
        "[00:45:10.544] [37824] [D] Compiling Data/Shaders/RunGrass.hlsl Grass:Vertex:4 to VSHADER D3DCOMPILE_DEBUG",
        "[00:45:10.544] [37824] [D] Shader logs:",
        "RunGrass.hlsl(10): error X1000: syntax error",
        "[00:45:10.544] [37824] [D] Compilation failed",
    ]
    mock_open.return_value = mock_file
    shader_configs, warnings, errors = parse_log("log.txt")
    assert shader_configs["RunGrass.hlsl"]["VSHADER"] == [
        {"entry": "Grass:Vertex:4", "defines": ["D3DCOMPILE_DEBUG", "VSHADER"]}
    ]
    assert errors == {}  # No errors parsed, based on failure


# Doctest examples converted to unit tests
def test_parse_timestamp_doctest():
    """Test parse_timestamp function from doctest example."""
    result = parse_timestamp("[12:34:56.789] [123] [D] Compiling...")
    expected = datetime(1900, 1, 1, 12, 34, 56, 789000)
    assert result == expected


def test_collect_tasks_doctest():
    """Test collect_tasks function from doctest example."""
    lines = ["[12:34:56.789] [123] [D] Compiling src/test.hlsl main:vertex:1234 to A=1"]
    tasks = collect_tasks(lines)
    assert len(tasks) == 1
    assert tasks[0].entry_point == "main:vertex:1234"
    assert tasks[0].file_path == "src/test.hlsl"
    assert tasks[0].defines == ["A=1"]


def test_populate_configs_doctest():
    """Test populate_configs function from doctest example."""
    task = CompilationTask("123", "main:vertex:1234", "src/test.hlsl", ["A=1"], datetime.now())
    tasks = [task]
    configs = populate_configs(tasks, {})
    expected_config = [{"entry": "main:vertex:1234", "defines": ["A=1"]}]
    assert configs["src/test.hlsl"]["VSHADER"] == expected_config


@patch("hlslkit.generate_shader_defines.open")
def test_parse_log_doctest(mock_open):
    """Test parse_log function from doctest example."""
    mock_file = MagicMock()
    mock_file.__enter__.return_value.readlines.return_value = [
        "[12:34:56.789] [123] [D] Compiling src/test.hlsl main:vertex:1234 to A=1",
        "[12:34:56.790] [123] [D] Compiled shader main:vertex:1234",
    ]
    mock_open.return_value = mock_file

    configs, warnings, errors = parse_log("CommunityShaders.log")
    expected_config = [{"entry": "main:vertex:1234", "defines": ["A=1"]}]
    assert configs["src/test.hlsl"]["VSHADER"] == expected_config


@patch("hlslkit.generate_shader_defines.open")
def test_count_compiling_lines_doctest(mock_open):
    """Test count_compiling_lines function from doctest example."""
    mock_file = MagicMock()
    mock_file.__enter__.return_value.__iter__.return_value = [
        "[12:34:56.789] [123] [D] Compiling src/test1.hlsl main:vertex:1234 to A=1",
        "[12:34:56.790] [123] [D] Some other log entry",
        "[12:34:56.791] [123] [D] Compiling src/test2.hlsl main:pixel:5678 to B=2",
        "[12:34:56.792] [123] [D] Another log entry",
        "[12:34:56.793] [123] [D] Compiling src/test3.hlsl main:compute:9012 to C=3",
    ]
    mock_open.return_value = mock_file

    result = count_compiling_lines("CommunityShaders.log")
    assert result == 3  # Should count 3 "[D] Compiling" lines


@patch("hlslkit.generate_shader_defines.open")
def test_count_log_blocks_doctest(mock_open):
    """Test count_log_blocks function."""
    mock_file = MagicMock()
    mock_file.__iter__.return_value = [
        "[00:45:10.539] [35768] [D] Shader logs:",
        "[00:45:10.540] [35768] [E] Failed to compile",
        "[00:45:10.541] [35768] [W] Shader compilation failed",
        "[00:45:10.542] [35768] [D] Adding Completed shader",
    ]
    mock_open.return_value.__enter__.return_value = mock_file
    assert count_log_blocks("log.txt") == 4


def test_generate_yaml_data_structure():
    """Test that generate_yaml_data produces the expected structure."""
    shader_configs = {
        "test.hlsl": {
            "PSHADER": [
                {"entry": "main:1234", "defines": ["A=1", "B=2"]},
                {"entry": "main:5678", "defines": ["A=1", "C=3"]},
            ],
            "VSHADER": [
                {"entry": "main:9012", "defines": ["A=1", "D=4"]},
            ],
        }
    }
    warnings = {
        "x3206:implicit truncation": {
            "code": "X3206",
            "message": "implicit truncation",
            "instances": {
                "test.hlsl:10": {"entries": ["main:1234"]},
            },
        }
    }
    errors = {}

    yaml_data = generate_yaml_data(shader_configs, warnings, errors)

    # Verify structure
    assert "common_defines" in yaml_data
    assert "common_pshader_defines" in yaml_data
    assert "common_vshader_defines" in yaml_data
    assert "common_cshader_defines" in yaml_data
    assert "file_common_defines" in yaml_data
    assert "warnings" in yaml_data
    assert "errors" in yaml_data
    assert "shaders" in yaml_data

    # Verify shader structure
    assert len(yaml_data["shaders"]) == 1
    shader = yaml_data["shaders"][0]
    assert shader["file"] == "test.hlsl"
    assert "configs" in shader
    assert "PSHADER" in shader["configs"]
    assert "VSHADER" in shader["configs"]

    # Verify PSHADER config
    pshader_config = shader["configs"]["PSHADER"]
    assert "common_defines" in pshader_config
    assert "entries" in pshader_config
    assert len(pshader_config["entries"]) == 2

    # Verify entries structure
    entry1 = pshader_config["entries"][0]
    assert "entry" in entry1
    assert "defines" in entry1
    assert entry1["entry"] == "main:1234"


def test_generate_yaml_data_with_anchors():
    """Test that generate_yaml_data can produce YAML with anchors."""
    shader_configs = {
        "test.hlsl": {
            "PSHADER": [
                {"entry": "main:1234", "defines": ["A=1", "B=2"]},
                {"entry": "main:5678", "defines": ["A=1", "B=2"]},  # Same defines
            ],
            "VSHADER": [
                {"entry": "main:9012", "defines": ["A=1", "B=2"]},  # Same defines
            ],
        }
    }
    warnings = {}
    errors = {}

    yaml_data = generate_yaml_data(shader_configs, warnings, errors)

    # Verify that common defines are extracted
    assert "A=1" in yaml_data["common_defines"]
    assert "B=2" in yaml_data["common_defines"]

    # Verify that individual entries don't repeat common defines
    pshader_config = yaml_data["shaders"][0]["configs"]["PSHADER"]
    for entry in pshader_config["entries"]:
        assert "A=1" not in entry["defines"]
        assert "B=2" not in entry["defines"]


def test_save_yaml_with_anchors():
    """Test that save_yaml can handle YAML data with anchors."""
    yaml_data = {
        "common_defines": ["A=1", "B=2"],
        "shaders": [
            {
                "file": "test.hlsl",
                "configs": {
                    "PSHADER": {
                        "common_defines": ["A=1", "B=2"],
                        "entries": [
                            {"entry": "main:1234", "defines": []},
                            {"entry": "main:5678", "defines": []},
                        ],
                    }
                },
            }
        ],
    }

    # This should not raise an exception
    with patch("builtins.open", create=True) as mock_open:
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        save_yaml(yaml_data, "test.yaml")
        mock_file.write.assert_called()


def test_yaml_output_has_anchors_and_is_loadable():
    """Test that save_yaml emits anchors for repeated lists and that the YAML is loadable."""

    from hlslkit.generate_shader_defines import save_yaml

    common = ["A=1", "B=2"]
    yaml_data = {
        "common_defines": common,
        "shaders": [
            {
                "file": "test.hlsl",
                "configs": {
                    "PSHADER": {
                        "common_defines": common,
                        "entries": [
                            {"entry": "main:1234", "defines": []},
                            {"entry": "main:5678", "defines": []},
                        ],
                    },
                    "VSHADER": {
                        "common_defines": common,
                        "entries": [
                            {"entry": "main:9012", "defines": []},
                        ],
                    },
                },
            }
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_yaml(yaml_data, tmp_path)
        with open(tmp_path, encoding="utf-8") as f:
            output = f.read()
        # Check for YAML anchor (&) and alias (*)
        assert "&" in output and "*" in output
        # Check that loading the YAML gives the same structure
        loaded = yaml.safe_load(output)
        assert loaded["common_defines"] == ["A=1", "B=2"]
        assert loaded["shaders"][0]["configs"]["PSHADER"]["common_defines"] == ["A=1", "B=2"]
        assert loaded["shaders"][0]["configs"]["VSHADER"]["common_defines"] == ["A=1", "B=2"]
    finally:
        # Clean up
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def test_yaml_deduplication_and_nested_anchors():
    """Test that save_yaml deduplicates equal lists and emits anchors for flat and nested cases."""
    import tempfile

    from hlslkit.generate_shader_defines import save_yaml

    # Separate but equal lists
    a = ["A=1", "B=2"]
    b = ["A=1", "B=2"]  # different object, same value
    c = ["A=1", "B=2"]
    # Nested repeated list
    nested1 = [["X", "Y"], ["X", "Y"]]
    nested2 = [["X", "Y"], ["X", "Y"]]
    yaml_data = {
        "common_defines": a,
        "shaders": [
            {
                "file": "test.hlsl",
                "configs": {
                    "PSHADER": {
                        "common_defines": b,
                        "entries": [
                            {"entry": "main:1234", "defines": []},
                            {"entry": "main:5678", "defines": []},
                        ],
                    },
                    "VSHADER": {
                        "common_defines": c,
                        "entries": [
                            {"entry": "main:9012", "defines": []},
                        ],
                    },
                },
            }
        ],
        "nested": nested1,
        "nested2": nested2,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_yaml(yaml_data, tmp_path)
        with open(tmp_path, encoding="utf-8") as f:
            output = f.read()
        # Check for YAML anchor (&) and alias (*)
        assert output.count("&") >= 2  # at least two anchors (flat and nested)
        assert output.count("*") >= 2  # at least two aliases
        # Check that loading the YAML gives the same structure (by value)
        loaded = yaml.safe_load(output)
        assert loaded["common_defines"] == ["A=1", "B=2"]
        assert loaded["shaders"][0]["configs"]["PSHADER"]["common_defines"] == ["A=1", "B=2"]
        assert loaded["shaders"][0]["configs"]["VSHADER"]["common_defines"] == ["A=1", "B=2"]
        # Nested lists
        assert loaded["nested"] == [["X", "Y"], ["X", "Y"]]
        assert loaded["nested2"] == [["X", "Y"], ["X", "Y"]]
    finally:
        # Clean up
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def test_optimize_anchor_deduplication_simple():
    """Test that optimize_anchor_deduplication works with simple lists."""

    # Create separate list objects with identical content
    a = ["A=1", "B=2"]
    b = ["A=1", "B=2"]  # different object, same value
    c = ["A=1", "B=2"]  # different object, same value

    yaml_data = {
        "list1": a,
        "list2": b,
        "list3": c,
    }

    optimized, _ = optimize_anchor_deduplication(yaml_data)

    # All three lists should now be the same object
    assert optimized["list1"] is optimized["list2"]
    assert optimized["list2"] is optimized["list3"]
    assert optimized["list1"] == ["A=1", "B=2"]


def test_optimize_anchor_deduplication_nested():
    """Test that optimize_anchor_deduplication works with nested lists."""

    # Create nested lists
    nested1 = [["X", "Y"], ["X", "Y"]]
    nested2 = [["X", "Y"], ["X", "Y"]]

    yaml_data = {
        "nested1": nested1,
        "nested2": nested2,
    }

    optimized, _ = optimize_anchor_deduplication(yaml_data)

    # The nested lists should now be the same object
    assert optimized["nested1"] is optimized["nested2"]
    assert optimized["nested1"] == [["X", "Y"], ["X", "Y"]]


def test_optimize_anchor_deduplication_complex():
    """Test that optimize_anchor_deduplication works with complex nested structures."""

    # Create separate list objects with identical content
    a = ["A=1", "B=2"]
    b = ["A=1", "B=2"]  # different object, same value
    c = ["A=1", "B=2"]  # different object, same value

    # Complex nested structure similar to the failing test
    yaml_data = {
        "common_defines": a,
        "shaders": [
            {
                "file": "test.hlsl",
                "configs": {
                    "PSHADER": {
                        "common_defines": b,
                        "entries": [
                            {"entry": "main:1234", "defines": []},
                            {"entry": "main:5678", "defines": []},
                        ],
                    },
                    "VSHADER": {
                        "common_defines": c,
                        "entries": [
                            {"entry": "main:9012", "defines": []},
                        ],
                    },
                },
            }
        ],
    }

    optimized, _ = optimize_anchor_deduplication(yaml_data)

    # All three lists should now be the same object
    assert optimized["common_defines"] is optimized["shaders"][0]["configs"]["PSHADER"]["common_defines"]
    assert (
        optimized["shaders"][0]["configs"]["PSHADER"]["common_defines"]
        is optimized["shaders"][0]["configs"]["VSHADER"]["common_defines"]
    )
    assert optimized["common_defines"] == ["A=1", "B=2"]
