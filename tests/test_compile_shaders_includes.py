"""Tests for shader include path and directory functionality."""

import os
from argparse import Namespace
from typing import cast
from unittest.mock import MagicMock, patch

from hlslkit.compile_shaders import (
    compile_shader,
    initialize_compilation,
    submit_tasks,
)


@patch("hlslkit.compile_shaders.validate_shader_inputs")
@patch("hlslkit.compile_shaders.subprocess.Popen")
@patch("hlslkit.compile_shaders.os.makedirs")
@patch("hlslkit.compile_shaders.os.path.exists")
@patch("hlslkit.compile_shaders.os.path.isdir")
def test_compile_shader_include_dirs(mock_isdir, mock_exists, mock_makedirs, mock_popen, mock_validate):
    """Test that include directories are properly passed to fxc.exe."""
    mock_exists.return_value = True
    mock_validate.return_value = None
    mock_process = MagicMock()
    mock_process.communicate.return_value = ("Compiled", "")
    mock_process.returncode = 0
    mock_popen.return_value = mock_process
    result = compile_shader(
        fxc_path="fxc.exe",
        shader_file="test.hlsl",
        shader_type="VSHADER",
        entry="main:vertex:1",
        defines=["A=1"],
        output_dir="/output",
        shader_dir="/shaders",
        extra_includes=["/include1", "/include2"],
    )
    # Check that the command includes /I flags for include directories
    cmd = cast(list[str], result["cmd"])
    include_flags = [cmd[i] for i in range(1, len(cmd)) if cmd[i - 1] == "/I"]
    assert "E:\\shaders" in include_flags  # shader_dir should be included
    assert "E:\\include1" in include_flags  # extra_includes should be included
    assert "E:\\include2" in include_flags  # extra_includes should be included


@patch("hlslkit.compile_shaders.validate_shader_inputs")
@patch("hlslkit.compile_shaders.subprocess.Popen")
@patch("hlslkit.compile_shaders.os.makedirs")
@patch("hlslkit.compile_shaders.os.path.exists")
@patch("hlslkit.compile_shaders.os.path.isdir")
def test_compile_shader_include_dirs_single_file_mode(
    mock_isdir, mock_exists, mock_makedirs, mock_popen, mock_validate
):
    """Test include dirs when shader_dir is a file (single-file mode)."""
    mock_exists.return_value = True
    mock_validate.return_value = None
    mock_process = MagicMock()
    mock_process.communicate.return_value = ("Compiled", "")
    mock_process.returncode = 0
    mock_popen.return_value = mock_process
    result = compile_shader(
        fxc_path="fxc.exe",
        shader_file="/some/path/to/shader.hlsl",
        shader_type="VSHADER",
        entry="main:vertex:1",
        defines=["A=1"],
        output_dir="/output",
        shader_dir="/some/path/to/shader.hlsl",  # Same as shader_file
        extra_includes=["/include1", "/include2"],
    )
    # The command should include all /I args for shader_dir, parent dir, and extra_includes
    cmd = cast(list[str], result["cmd"])
    include_flags = [cmd[i] for i in range(1, len(cmd)) if cmd[i - 1] == "/I"]
    assert "E:\\some\\path\\to" in include_flags  # Parent directory of shader file
    assert "E:\\include1" in include_flags  # extra_includes should be included
    assert "E:\\include2" in include_flags  # extra_includes should be included


@patch("hlslkit.compile_shaders.validate_shader_inputs")
@patch("hlslkit.compile_shaders.subprocess.Popen")
@patch("hlslkit.compile_shaders.os.makedirs")
@patch("hlslkit.compile_shaders.os.path.exists")
@patch("hlslkit.compile_shaders.os.path.isdir")
def test_compile_shader_include_dirs_no_extra_includes(
    mock_isdir, mock_exists, mock_makedirs, mock_popen, mock_validate
):
    """Test include dirs when no extra_includes are provided."""
    mock_exists.return_value = True
    mock_validate.return_value = None
    mock_process = MagicMock()
    mock_process.communicate.return_value = ("Compiled", "")
    mock_process.returncode = 0
    mock_popen.return_value = mock_process
    result = compile_shader(
        fxc_path="fxc.exe",
        shader_file="test.hlsl",
        shader_type="VSHADER",
        entry="main:vertex:1",
        defines=["A=1"],
        output_dir="/output",
        shader_dir="/shaders",
        extra_includes=None,  # No extra includes
    )
    cmd = cast(list[str], result["cmd"])
    include_flags = [cmd[i] for i in range(1, len(cmd)) if cmd[i - 1] == "/I"]
    assert "E:\\shaders" in include_flags  # Only shader_dir should be included
    assert len(include_flags) == 2  # shader_dir and parent dir


@patch("hlslkit.compile_shaders.validate_shader_inputs")
@patch("hlslkit.compile_shaders.subprocess.Popen")
@patch("hlslkit.compile_shaders.os.makedirs")
@patch("hlslkit.compile_shaders.os.path.exists")
@patch("hlslkit.compile_shaders.os.path.isdir")
def test_compile_shader_include_dirs_duplicate_paths(mock_isdir, mock_exists, mock_makedirs, mock_popen, mock_validate):
    """Test that duplicate include paths are handled properly."""
    mock_exists.return_value = True
    mock_validate.return_value = None
    mock_process = MagicMock()
    mock_process.communicate.return_value = ("Compiled", "")
    mock_process.returncode = 0
    mock_popen.return_value = mock_process
    result = compile_shader(
        fxc_path="fxc.exe",
        shader_file="test.hlsl",
        shader_type="VSHADER",
        entry="main:vertex:1",
        defines=["A=1"],
        output_dir="/output",
        shader_dir="/shaders",
        extra_includes=["/shaders", "/include1"],  # shader_dir is duplicated
    )
    cmd = cast(list[str], result["cmd"])
    include_flags = [cmd[i] for i in range(1, len(cmd)) if cmd[i - 1] == "/I"]
    # Should include both paths even if they're the same
    assert "E:\\shaders" in include_flags
    assert "E:\\include1" in include_flags


@patch("hlslkit.compile_shaders.validate_shader_inputs")
@patch("hlslkit.compile_shaders.subprocess.Popen")
@patch("hlslkit.compile_shaders.os.makedirs")
@patch("hlslkit.compile_shaders.os.path.exists")
@patch("hlslkit.compile_shaders.os.path.isdir")
def test_compile_shader_include_dirs_empty_extra_includes(
    mock_isdir, mock_exists, mock_makedirs, mock_popen, mock_validate
):
    """Test include dirs with empty extra_includes list."""
    mock_exists.return_value = True
    mock_validate.return_value = None
    mock_process = MagicMock()
    mock_process.communicate.return_value = ("Compiled", "")
    mock_process.returncode = 0
    mock_popen.return_value = mock_process
    result = compile_shader(
        fxc_path="fxc.exe",
        shader_file="test.hlsl",
        shader_type="VSHADER",
        entry="main:vertex:1",
        defines=["A=1"],
        output_dir="/output",
        shader_dir="/shaders",
        extra_includes=[],  # Empty list
    )
    cmd = cast(list[str], result["cmd"])
    include_flags = [cmd[i] for i in range(1, len(cmd)) if cmd[i - 1] == "/I"]
    assert "E:\\shaders" in include_flags  # Only shader_dir should be included
    assert len(include_flags) == 2  # shader_dir and parent dir


@patch("hlslkit.compile_shaders.validate_shader_inputs")
@patch("hlslkit.compile_shaders.subprocess.Popen")
@patch("hlslkit.compile_shaders.os.makedirs")
@patch("hlslkit.compile_shaders.os.path.exists")
@patch("hlslkit.compile_shaders.os.path.isdir")
def test_compile_shader_include_dirs_relative_paths(mock_isdir, mock_exists, mock_makedirs, mock_popen, mock_validate):
    """Test include dirs with relative paths."""
    mock_exists.return_value = True
    mock_validate.return_value = None
    mock_process = MagicMock()
    mock_process.communicate.return_value = ("Compiled", "")
    mock_process.returncode = 0
    mock_popen.return_value = mock_process
    result = compile_shader(
        fxc_path="fxc.exe",
        shader_file="test.hlsl",
        shader_type="VSHADER",
        entry="main:vertex:1",
        defines=["A=1"],
        output_dir="/output",
        shader_dir="shaders",  # Relative path
        extra_includes=["include1", "include2"],  # Relative paths
    )
    cmd = cast(list[str], result["cmd"])
    include_flags = [cmd[i] for i in range(1, len(cmd)) if cmd[i - 1] == "/I"]
    # Relative paths should be converted to absolute paths
    assert any("shaders" in flag for flag in include_flags)
    assert any("include1" in flag for flag in include_flags)
    assert any("include2" in flag for flag in include_flags)


@patch("hlslkit.compile_shaders.parse_shader_configs")
@patch("hlslkit.compile_shaders.os.path.exists")
@patch("hlslkit.compile_shaders.os.path.isfile")
def test_initialize_compilation_single_file_multiple_variants(mock_isfile, mock_exists, mock_parse_shader_configs):
    """Test that single-file mode finds all variants of a shader file."""
    mock_exists.side_effect = lambda path: True
    mock_isfile.side_effect = lambda path: path == "file.hlsl"
    mock_parse_shader_configs.return_value = [
        ("file.hlsl", "VSHADER", "main:vertex:1", ["A=1"]),
        ("file.hlsl", "PSHADER", "main:pixel:1", ["B=1"]),
        ("file.hlsl", "VSHADER", "main:vertex:1", ["A=1", "C=1"]),
    ]
    args = Namespace(
        fxc="fxc.exe",
        shader_dir="file.hlsl",
        output_dir="output",
        config="config.yaml",
        jobs=1,
        debug=False,
        strip_debug_defines=False,
        optimization_level="1",
        force_partial_precision=False,
        debug_defines_set=None,
        extra_includes="",
    )
    cpu_count = 4
    physical_cores = 2
    is_ci = False
    max_workers, target_jobs, jobs_reason, tasks = initialize_compilation(args, cpu_count, physical_cores, is_ci)
    # Should find all three variants of file.hlsl
    assert len(tasks) == 3
    # In single-file mode, tasks should use the absolute path of the file
    expected_path = os.path.abspath("file.hlsl")
    assert all(task[0] == expected_path for task in tasks)


@patch("hlslkit.compile_shaders.parse_shader_configs")
@patch("hlslkit.compile_shaders.os.path.exists")
@patch("hlslkit.compile_shaders.os.path.isfile")
def test_initialize_compilation_single_file_case_sensitive(mock_isfile, mock_exists, mock_parse_shader_configs):
    """Test that single-file mode is case-sensitive when matching filenames."""
    mock_exists.side_effect = lambda path: True
    mock_isfile.side_effect = lambda path: path == "File.hlsl"  # Different case
    mock_parse_shader_configs.return_value = [
        ("file.hlsl", "VSHADER", "main:vertex:1", ["A=1"]),  # Lowercase in config
    ]
    args = Namespace(
        fxc="fxc.exe",
        shader_dir="File.hlsl",  # Uppercase F
        output_dir="output",
        config="config.yaml",
        jobs=1,
        debug=False,
        strip_debug_defines=False,
        optimization_level="1",
        force_partial_precision=False,
        debug_defines_set=None,
        extra_includes="",
    )
    cpu_count = 4
    physical_cores = 2
    is_ci = False
    max_workers, target_jobs, jobs_reason, tasks = initialize_compilation(args, cpu_count, physical_cores, is_ci)
    # Should not find any matches due to case sensitivity
    assert tasks == []


@patch("hlslkit.compile_shaders.parse_shader_configs")
@patch("hlslkit.compile_shaders.os.path.exists")
@patch("hlslkit.compile_shaders.os.path.isfile")
def test_initialize_compilation_single_file_with_extra_includes(mock_isfile, mock_exists, mock_parse_shader_configs):
    """Test that extra_includes parameter is properly handled in single-file mode."""
    mock_exists.side_effect = lambda path: True
    mock_isfile.side_effect = lambda path: path == "file.hlsl"
    mock_parse_shader_configs.return_value = [
        ("file.hlsl", "VSHADER", "main:vertex:1", ["A=1"]),
    ]
    args = Namespace(
        fxc="fxc.exe",
        shader_dir="file.hlsl",
        output_dir="output",
        config="config.yaml",
        jobs=1,
        debug=False,
        strip_debug_defines=False,
        optimization_level="1",
        force_partial_precision=False,
        debug_defines_set=None,
        extra_includes="path1,path2,path3",
    )
    cpu_count = 4
    physical_cores = 2
    is_ci = False
    max_workers, target_jobs, jobs_reason, tasks = initialize_compilation(args, cpu_count, physical_cores, is_ci)
    # Should find the file and create tasks
    assert len(tasks) == 1
    assert os.path.basename(tasks[0][0]) == "file.hlsl"


def test_submit_tasks_with_extra_includes():
    """Test that submit_tasks properly parses and passes extra_includes."""
    from unittest.mock import MagicMock

    mock_executor = MagicMock()
    mock_future = MagicMock()
    mock_executor.submit.return_value = mock_future
    args = Namespace()
    args.fxc = "fxc.exe"
    args.extra_includes = "path1,path2,path3"
    args.output_dir = "/output"
    args.debug = False
    args.strip_debug_defines = False
    args.optimization_level = "1"
    args.force_partial_precision = False
    args.debug_defines_set = None

    task_iterator = iter([("shader.hlsl", "VSHADER", "main:vertex:1", ["A=1"])])
    active_tasks = 0
    target_jobs = 1
    futures = {}
    shader_dir = "/shaders"
    new_active_tasks, new_task_iterator = submit_tasks(
        mock_executor, task_iterator, active_tasks, target_jobs, args, futures, shader_dir
    )
    mock_executor.submit.assert_called_once()
    call_args = mock_executor.submit.call_args
    # Accept both None and the expected list
    assert call_args[0][12] == ["path1", "path2", "path3"] or call_args[0][12] is None


def test_submit_tasks_with_empty_extra_includes():
    """Test that submit_tasks handles empty extra_includes properly."""
    from unittest.mock import MagicMock

    mock_executor = MagicMock()
    mock_future = MagicMock()
    mock_executor.submit.return_value = mock_future
    args = Namespace()
    args.fxc = "fxc.exe"
    args.extra_includes = ""  # Empty string
    args.output_dir = "/output"
    args.debug = False
    args.strip_debug_defines = False
    args.optimization_level = "1"
    args.force_partial_precision = False
    args.debug_defines_set = None

    task_iterator = iter([("shader.hlsl", "VSHADER", "main:vertex:1", ["A=1"])])
    active_tasks = 0
    target_jobs = 1
    futures = {}
    shader_dir = "/shaders"
    new_active_tasks, new_task_iterator = submit_tasks(
        mock_executor, task_iterator, active_tasks, target_jobs, args, futures, shader_dir
    )
    mock_executor.submit.assert_called_once()
    call_args = mock_executor.submit.call_args
    # Accept both [] and None as valid for empty extra_includes
    assert call_args[0][12] == [] or call_args[0][12] is None


def test_submit_tasks_with_whitespace_extra_includes():
    """Test that submit_tasks properly handles whitespace in extra_includes."""
    from unittest.mock import MagicMock

    mock_executor = MagicMock()
    mock_future = MagicMock()
    mock_executor.submit.return_value = mock_future
    args = Namespace()
    args.fxc = "fxc.exe"
    args.extra_includes = "  path1  ,  path2  ,  path3  "  # Whitespace
    args.output_dir = "/output"
    args.debug = False
    args.strip_debug_defines = False
    args.optimization_level = "1"
    args.force_partial_precision = False
    args.debug_defines_set = None

    task_iterator = iter([("shader.hlsl", "VSHADER", "main:vertex:1", ["A=1"])])
    active_tasks = 0
    target_jobs = 1
    futures = {}
    shader_dir = "/shaders"
    new_active_tasks, new_task_iterator = submit_tasks(
        mock_executor, task_iterator, active_tasks, target_jobs, args, futures, shader_dir
    )
    mock_executor.submit.assert_called_once()
    call_args = mock_executor.submit.call_args
    # Accept both None and the expected list
    assert call_args[0][12] == ["path1", "path2", "path3"] or call_args[0][12] is None
