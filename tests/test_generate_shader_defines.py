from datetime import datetime
from unittest.mock import MagicMock, patch

from hlslkit.generate_shader_defines import (
    CompilationTask,
    collect_tasks,
    count_compiling_lines,
    count_log_blocks,
    get_shader_type_from_entry,
    normalize_path,
    parse_log,
    parse_timestamp,
    populate_configs,
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
                "VSHADER",
                "D3DCOMPILE_SKIP_OPTIMIZATION",
                "D3DCOMPILE_DEBUG",
                "OCCLUSION",
                "SCREEN_SPACE_SHADOWS",
                "WETNESS_EFFECTS",
                "LIGHT_LIMIT_FIX",
                "DYNAMIC_CUBEMAPS",
                "CLOUD_SHADOWS",
                "WATER_EFFECTS",
                "SSS",
                "TERRAIN_SHADOWS",
                "SKYLIGHTING",
                "TERRAIN_BLENDING",
                "LOD_BLENDING",
                "ISL",
                "IBL",
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
        {"entry": "Grass:Vertex:4", "defines": ["VSHADER", "D3DCOMPILE_DEBUG", "WATER_EFFECTS", "GRASS_COLLISION"]}
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
        {"entry": "Grass:Vertex:4", "defines": ["VSHADER", "D3DCOMPILE_DEBUG", "WATER_EFFECTS", "GRASS_COLLISION"]}
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
            "defines": ["VSHADER", "D3DCOMPILE_DEBUG", "WATER_EFFECTS", "GRASS_COLLISION", "WATER_EFFECTS"],
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
        {"entry": "Grass:Vertex:4", "defines": ["VSHADER", "D3DCOMPILE_DEBUG"]}
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
    """Test count_log_blocks function from doctest example."""
    mock_file = MagicMock()
    mock_file.__enter__.return_value.__iter__.return_value = [
        "[12:34:56.789] [123] [D] Compiling src/test1.hlsl main:vertex:1234 to A=1",
        "[12:34:56.790] [123] [D] Shader logs:",
        "[12:34:56.791] [123] [E] Failed to compile Pixel shader",
        "[12:34:56.792] [123] [W] Shader compilation failed",
        "[12:34:56.793] [123] [D] Adding Completed shader to map",
        "[12:34:56.794] [123] [D] Some other log entry",
    ]
    mock_open.return_value = mock_file

    result = count_log_blocks("CommunityShaders.log")
    assert (
        result == 4
    )  # Should count 4 log blocks (Shader logs, Failed to compile, compilation failed, Adding Completed)
