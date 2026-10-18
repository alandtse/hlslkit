"""Tests for the failure and edge paths of the compile content dedupe."""

import argparse
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock

from hlslkit import compile_shaders, dedupe
from hlslkit.dedupe import expand_results, group_variants, select_canary_members, verify_deduped_members

TASK_A = ("a.hlsl", "PSHADER", "main:1", ["A"])
TASK_B = ("b.hlsl", "PSHADER", "main:2", ["B"])


def _result(task, log="", success=True):
    return {
        "file": task[0],
        "entry": task[2],
        "type": task[1],
        "log": log,
        "success": success,
        "cmd": ["fxc.exe"],
        "duration_seconds": 1.0,
    }


def _write_blob(output_dir, task, data):
    path = compile_shaders.shader_output_path(output_dir, task[0], task[1], task[2])
    assert path is not None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Path(path).write_bytes(data)
    return path


def _args(tmp_path):
    return argparse.Namespace(
        fxc="fxc.exe",
        shader_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        extra_includes="",
        debug=False,
        strip_debug_defines=False,
        optimization_level="1",
        force_partial_precision=False,
        debug_defines_set=None,
    )


def test_preprocess_variant_returns_none_for_a_missing_shader(tmp_path):
    text = dedupe.preprocess_variant("fxc.exe", "missing.hlsl", "main:1", [], str(tmp_path))
    assert text is None


def test_preprocess_variant_returns_none_when_fxc_cannot_start(tmp_path, monkeypatch):
    (tmp_path / "a.hlsl").write_text("float x;\n")

    def boom(*_args, **_kwargs):
        raise OSError

    monkeypatch.setattr(dedupe.subprocess, "Popen", boom)
    assert dedupe.preprocess_variant("fxc.exe", "a.hlsl", "main:1", [], str(tmp_path)) is None


def test_group_variants_treats_a_raising_preprocess_as_a_failure():
    def preprocess(task):
        if task is TASK_A:
            raise RuntimeError("boom")
        return "text"

    groups, failures = group_variants([TASK_A, TASK_B], preprocess, "1", False, 1)
    assert failures == 1
    assert sorted(groups) == [[0], [1]]


def test_expand_results_fails_a_member_whose_blob_path_is_unsupported(tmp_path):
    tasks = [("a.hlsl", "XSHADER", "main:1", []), ("b.hlsl", "XSHADER", "main:2", [])]
    expanded = expand_results(tasks, [[0, 1]], [_result(tasks[0])], str(tmp_path))
    assert [r["success"] for r in expanded] == [True, False]


def test_expand_results_fails_a_member_when_the_blob_copy_fails(tmp_path, monkeypatch):
    tasks = [TASK_A, TASK_B]
    _write_blob(str(tmp_path), TASK_A, b"blob")

    def fail_copy(*_args, **_kwargs):
        raise OSError

    monkeypatch.setattr(shutil, "copyfile", fail_copy)
    expanded = expand_results(tasks, [[0, 1]], [_result(TASK_A)], str(tmp_path))
    assert [r["success"] for r in expanded] == [True, False]
    assert expanded[1]["deduped_from"] == "a.hlsl:main:1"


def test_expand_results_reports_tasks_missing_from_every_group(tmp_path):
    expanded = expand_results([TASK_A, TASK_B], [[0]], [_result(TASK_A)], str(tmp_path))
    assert expanded[0]["success"] is True
    assert expanded[1]["success"] is False
    assert expanded[1]["log"] == "Compilation result missing."


def test_select_canary_members_skips_groups_with_a_failed_representative():
    results = [_result(TASK_A, success=False), _result(TASK_B, success=False)]
    assert select_canary_members([[0, 1]], results, 100.0) == []


def test_select_canary_members_returns_nothing_without_members():
    assert select_canary_members([[0]], [_result(TASK_A)], 100.0) == []


def test_blob_matches_requires_both_blobs_to_exist(tmp_path):
    present = tmp_path / "present.pso"
    present.write_bytes(b"x")
    assert dedupe._blob_matches(str(present), str(present)) is True
    assert dedupe._blob_matches(None, str(present)) is False
    assert dedupe._blob_matches(str(present), str(tmp_path / "missing.pso")) is False


def test_verify_reports_a_failed_recompile(tmp_path, monkeypatch):
    tasks = [TASK_A, TASK_B]
    results = [_result(TASK_A), _result(TASK_B)]
    monkeypatch.setattr(compile_shaders, "compile_shader", lambda *call_args: _result(TASK_B, success=False))
    mismatches = verify_deduped_members(tasks, [[0, 1]], results, _args(tmp_path), 100.0)
    assert len(mismatches) == 1
    assert "recompile failed" in mismatches[0]
    assert "b.hlsl:main:2" in mismatches[0]


def test_strip_defines_removes_debug_defines_and_adds_avoid_flow_control():
    defines = ["KEEP=1", "D3DCOMPILE_DEBUG", "_debug=1"]
    stripped = compile_shaders._strip_defines_for_compile(defines, True, None, "a.hlsl", "main:1")
    assert stripped == ["KEEP=1", "D3DCOMPILE_AVOID_FLOW_CONTROL"]
    assert defines == ["KEEP=1", "D3DCOMPILE_DEBUG", "_debug=1"]


def test_strip_defines_does_not_duplicate_avoid_flow_control():
    defines = ["D3DCOMPILE_AVOID_FLOW_CONTROL", "DEBUG"]
    stripped = compile_shaders._strip_defines_for_compile(defines, True, {"DEBUG"}, "a.hlsl", "main:1")
    assert stripped == ["D3DCOMPILE_AVOID_FLOW_CONTROL"]


def test_strip_defines_is_a_no_op_when_disabled():
    defines = ["DEBUG"]
    assert compile_shaders._strip_defines_for_compile(defines, False, None, "a.hlsl", "main:1") is defines


def test_preprocess_invocation_mirrors_the_compile_define_and_include_layout(tmp_path):
    cmd, cwd, shader_path = compile_shaders.build_preprocess_invocation(
        "fxc.exe", "a.hlsl", "main:1", ["A=1", "DEBUG"], str(tmp_path), "out.hlsl", strip_debug_defines=True
    )
    assert cmd[:4] == ["fxc.exe", "/P", "out.hlsl", "a.hlsl"]
    assert "/T" not in cmd
    assert "/E" not in cmd
    assert cmd[cmd.index("/D") + 1] == "A=1"
    assert "D3DCOMPILE_AVOID_FLOW_CONTROL" in cmd
    assert "DEBUG" not in cmd
    assert cmd[cmd.index("/I") + 1] == os.path.abspath(tmp_path)
    assert cwd == os.path.abspath(tmp_path)
    assert shader_path == os.path.join(str(tmp_path), "a.hlsl")


def test_compile_shader_passes_partial_precision(tmp_path, monkeypatch):
    (tmp_path / "a.hlsl").write_text("float x;\n")
    (tmp_path / "out").mkdir()
    captured = {}

    def fake_popen(cmd, **_kwargs):
        captured["cmd"] = cmd
        process = MagicMock()
        process.communicate.return_value = ("", "")
        process.returncode = 0
        return process

    monkeypatch.setattr(compile_shaders, "validate_shader_inputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compile_shaders.subprocess, "Popen", fake_popen)
    result = compile_shaders.compile_shader(
        "fxc.exe",
        "a.hlsl",
        "PSHADER",
        "main:1",
        [],
        str(tmp_path / "out"),
        str(tmp_path),
        force_partial_precision=True,
    )
    assert result["success"] is True
    assert "/Gfp" in captured["cmd"]


def test_deduped_failure_report_ignores_runs_without_deduped_members():
    errors = {}
    compile_shaders._add_deduped_failure_errors([_result(TASK_A, success=False)], errors)
    assert errors == {}


def test_main_exits_nonzero_for_an_interrupted_run_with_a_failed_canary(monkeypatch):
    args = argparse.Namespace(
        config="config.yaml",
        output_dir="out",
        suppress_warnings="",
        max_warnings=0,
        timing_report="",
        dedupe_verify_failed=True,
    )
    monkeypatch.setattr(compile_shaders, "parse_arguments", lambda default_jobs: args)
    monkeypatch.setattr(compile_shaders, "setup_environment", lambda parsed: (1, 1, False))
    monkeypatch.setattr(compile_shaders, "run_compilation", lambda *call_args: [_result(TASK_A)])
    monkeypatch.setattr(compile_shaders, "analyze_and_report_results", lambda *call_args: (0, 0, 0))
    monkeypatch.setattr(compile_shaders, "_maybe_write_timing_report", lambda *call_args: None)
    compile_shaders.stop_event.set()
    try:
        assert compile_shaders.main() == 1
    finally:
        compile_shaders.stop_event.clear()


def test_normalize_log_skips_missing_output_paths():
    assert dedupe._normalize_log("a.hlsl(1,2): warning X1: m", None, "") == "a.hlsl(L): warning X1: m"
