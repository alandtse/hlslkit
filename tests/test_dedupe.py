"""Tests for the opt-in compile content dedupe and verification path."""

import argparse
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hlslkit import compile_shaders, dedupe
from hlslkit.dedupe import (
    expand_results,
    group_variants,
    strip_line_directives,
    variant_key,
    verify_deduped_members,
)

TASKS = [
    ("a.hlsl", "VSHADER", "main:1", ["A"]),
    ("b.hlsl", "VSHADER", "main:2", ["B"]),
    ("c.hlsl", "PSHADER", "main:3", ["C"]),
]


def _result(task, log="", success=True, cmd=None, duration=1.5):
    return {
        "file": task[0],
        "entry": task[2],
        "type": task[1],
        "log": log,
        "success": success,
        "cmd": [] if cmd is None else cmd,
        "duration_seconds": duration,
    }


def _group(tasks, texts, optimization_level="1", force_partial_precision=False):
    mapping = {id(task): text for task, text in zip(tasks, texts)}
    return group_variants(
        tasks,
        lambda task: mapping[id(task)],
        optimization_level=optimization_level,
        force_partial_precision=force_partial_precision,
        max_workers=1,
    )


def _write_blob(output_dir, task, data):
    path = compile_shaders.shader_output_path(output_dir, task[0], task[1], task[2])
    assert path is not None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Path(path).write_bytes(data)
    return path


def _run_args(tmp_path, dedupe_enabled, verify_pct=0.0):
    return argparse.Namespace(
        fxc="fxc.exe",
        shader_dir="shaders",
        output_dir=str(tmp_path / "out"),
        extra_includes="",
        debug=False,
        strip_debug_defines=False,
        optimization_level="1",
        force_partial_precision=False,
        debug_defines_set=None,
        dedupe=dedupe_enabled,
        dedupe_verify=verify_pct,
        dedupe_verify_failed=False,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('#line 1 "x"\nfloat x;\n#line 2 "y"\n', "float x;\n"),
        ("#pragma line 1\n", "#pragma line 1\n"),
        ("// #line 3\n", "// #line 3\n"),
        ("  #line 4\n", "  #line 4\n"),
        ("#line 1\r\n\r\nfloat x;\r\n\r\n", "\r\nfloat x;\r\n\r\n"),
        ("", ""),
        ("a\r#line 9\nb", "a\r#line 9\nb"),
        ("\x0c#line 4\nx\n", "\x0c#line 4\nx\n"),
        ("\u2028#line 4\nx\n", "\u2028#line 4\nx\n"),
    ],
)
def test_strip_line_directives(text, expected):
    """Only raw lines starting with #line are removed; everything else is exact."""
    assert strip_line_directives(text) == expected


def test_variant_key_identical_text_and_flags_match():
    assert variant_key("same", "VSHADER", "1", False) == variant_key("same", "VSHADER", "1", False)


@pytest.mark.parametrize(
    "other",
    [
        ("same", "PSHADER", "1", False),
        ("same", "VSHADER", "2", False),
        ("same", "VSHADER", "1", True),
        ("different", "VSHADER", "1", False),
    ],
)
def test_variant_key_differs_when_text_model_optimization_or_precision_differ(other):
    assert variant_key("same", "VSHADER", "1", False) != variant_key(*other)


def test_preprocess_variant_returns_raw_text_and_cleans_temp(tmp_path, monkeypatch):
    shader = tmp_path / "a.hlsl"
    shader.write_text("float x;\n", encoding="utf-8")
    text = '#line 1 "a"\r\nfloat x;\r\n'
    captured = {}

    class FakeProcess:
        returncode = 0

        def communicate(self):
            output_path = captured["cmd"][captured["cmd"].index("/P") + 1]
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            return "", ""

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(compile_shaders.subprocess, "Popen", fake_popen)

    result = dedupe.preprocess_variant("fxc.exe", str(shader), "main:1", ["A=1"], str(tmp_path))

    cmd = captured["cmd"]
    assert result == text
    assert cmd[cmd.index("/D") : cmd.index("/D") + 2] == ["/D", "A=1"]
    assert captured["kwargs"]["cwd"] == str(tmp_path)
    assert not Path(cmd[cmd.index("/P") + 1]).parent.exists()


def test_preprocess_variant_returns_none_on_nonzero_exit(tmp_path, monkeypatch):
    shader = tmp_path / "a.hlsl"
    shader.write_text("float x;\n", encoding="utf-8")

    class FakeProcess:
        returncode = 1

        def communicate(self):
            return "", "fxc failed"

    monkeypatch.setattr(compile_shaders.subprocess, "Popen", lambda cmd, **kwargs: FakeProcess())

    assert dedupe.preprocess_variant("fxc.exe", str(shader), "main:1", [], str(tmp_path)) is None


def test_preprocess_variant_returns_none_when_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(compile_shaders.stop_event, "is_set", lambda: True)
    popen_mock = MagicMock()
    monkeypatch.setattr(compile_shaders.subprocess, "Popen", popen_mock)

    assert dedupe.preprocess_variant("fxc.exe", "a.hlsl", "main:1", [], str(tmp_path)) is None
    popen_mock.assert_not_called()


def test_group_identical_text_and_flags_group_together():
    tasks = [
        ("a.hlsl", "VSHADER", "main:1", ["A"]),
        ("b.hlsl", "VSHADER", "main:2", ["B"]),
        ("d.hlsl", "VSHADER", "main:4", ["D"]),
    ]
    groups, failures = _group(tasks, ["same", "same", "same"])
    assert failures == 0
    assert groups == [[0, 1, 2]]


def test_group_different_shader_model_does_not_group():
    groups, _failures = _group([TASKS[0], TASKS[2]], ["same", "same"])
    assert sorted(groups) == [[0], [1]]


def test_group_preprocessing_failure_is_a_singleton():
    tasks = [
        ("a.hlsl", "VSHADER", "main:1", ["A"]),
        ("b.hlsl", "VSHADER", "main:2", ["B"]),
        ("d.hlsl", "VSHADER", "main:4", ["D"]),
    ]
    groups, failures = _group(tasks, ["same", None, "same"])
    assert failures == 1
    assert sorted(groups) == [[0, 2], [1]]


def test_expand_results_preserves_order_and_reuses_representative_fields(tmp_path):
    groups = [[0, 1], [2]]
    representative_a = _result(TASKS[0], log="log-a", cmd=["fxc"])
    representative_c = _result(TASKS[2], log="log-c")
    _write_blob(str(tmp_path), TASKS[0], b"blob")

    expanded = expand_results(TASKS, groups, [representative_c, representative_a], str(tmp_path))

    assert [result["entry"] for result in expanded] == ["main:1", "main:2", "main:3"]
    member = expanded[1]
    assert member["file"] == TASKS[1][0]
    assert member["type"] == TASKS[1][1]
    assert member["log"] == "log-a"
    assert member["success"] is True
    assert member["cmd"] == ["fxc"]
    assert member["duration_seconds"] == 0.0
    assert member["deduped_from"] == "a.hlsl:main:1"
    assert expanded[0]["duration_seconds"] == 1.5


def test_expand_results_copies_blob_to_member_path(tmp_path):
    tasks = [TASKS[0], TASKS[1]]
    _write_blob(str(tmp_path), tasks[0], b"representative-blob")

    expand_results(tasks, [[0, 1]], [_result(tasks[0])], str(tmp_path))

    member_blob = compile_shaders.shader_output_path(str(tmp_path), tasks[1][0], tasks[1][1], tasks[1][2])
    assert member_blob is not None
    assert Path(member_blob).read_bytes() == b"representative-blob"


def test_expand_results_failing_group_fails_every_member(tmp_path):
    tasks = [TASKS[0], TASKS[1]]
    expanded = expand_results(tasks, [[0, 1]], [_result(tasks[0], log="boom", success=False)], str(tmp_path))

    assert [result["success"] for result in expanded] == [False, False]
    assert expanded[1]["deduped_from"] == "a.hlsl:main:1"
    member_blob = compile_shaders.shader_output_path(str(tmp_path), tasks[1][0], tasks[1][1], tasks[1][2])
    assert member_blob is not None
    assert not Path(member_blob).exists()


def test_expand_results_missing_representative_result_fails_members(tmp_path):
    tasks = [TASKS[0], TASKS[1]]
    expanded = expand_results(tasks, [[0, 1]], [], str(tmp_path))

    assert [result["success"] for result in expanded] == [False, False]
    assert expanded[0]["log"] == "Compilation result missing."
    assert expanded[1]["deduped_from"] == "a.hlsl:main:1"


def test_failed_deduped_group_is_named_in_error_report(tmp_path):
    tasks = [TASKS[0], TASKS[1]]
    expanded = expand_results(
        tasks, [[0, 1]], [_result(tasks[0], log="not a compiler line", success=False)], str(tmp_path)
    )

    new_warnings, _all_warnings, errors, _suppressed = compile_shaders.process_warnings_and_errors(expanded, {}, [], {})
    assert "a.hlsl:main:1" in errors
    assert "b.hlsl:main:2" in errors

    compile_shaders.log_new_issues(new_warnings, errors, expanded, str(tmp_path), {})
    report = (tmp_path / "new_issues.log").read_text(encoding="utf-8")
    assert "main:1" in report
    assert "main:2" in report


def test_parseable_error_log_names_every_member(tmp_path):
    tasks = [TASKS[0], TASKS[1]]
    log = "a.hlsl(1): error X1000: bad shader\n"
    expanded = expand_results(tasks, [[0, 1]], [_result(tasks[0], log=log, success=False)], str(tmp_path))

    _new_warnings, _all_warnings, errors, _suppressed = compile_shaders.process_warnings_and_errors(
        expanded, {}, [], {}
    )
    assert set(errors) == {"a.hlsl:main:1", "b.hlsl:main:2"}


def _verify_args(tmp_path, verify_pct=2.0):
    return argparse.Namespace(
        fxc="fxc.exe",
        extra_includes="",
        output_dir=str(tmp_path / "out"),
        debug=False,
        strip_debug_defines=False,
        optimization_level="1",
        force_partial_precision=False,
        debug_defines_set=None,
        shader_dir="shaders",
        dedupe_verify=verify_pct,
    )


def _canary_setup(tmp_path):
    tasks = [TASKS[0], TASKS[1]]
    groups = [[0, 1]]
    representative = _result(tasks[0], log="clean", duration=1.0)
    results = [
        representative,
        dict(representative, file=tasks[1][0], entry=tasks[1][2], duration_seconds=0.0, deduped_from="a.hlsl:main:1"),
    ]
    _write_blob(str(tmp_path / "out"), tasks[0], b"blob")
    return tasks, groups, results


def test_verify_matching_blob_and_log_passes(tmp_path, monkeypatch):
    tasks, groups, results = _canary_setup(tmp_path)
    args = _verify_args(tmp_path)

    def fake_compile(*call_args):
        _write_blob(call_args[5], tasks[1], b"blob")
        return _result(tasks[1], log="clean")

    monkeypatch.setattr(compile_shaders, "compile_shader", fake_compile)
    assert verify_deduped_members(tasks, groups, results, args, 100.0) == []


def test_verify_ignores_the_output_path_echoed_in_the_log(tmp_path, monkeypatch):
    tasks, groups, results = _canary_setup(tmp_path)
    args = _verify_args(tmp_path)
    representative_blob = compile_shaders.shader_output_path(args.output_dir, tasks[0][0], tasks[0][1], tasks[0][2])
    results[0]["log"] = f"compilation object save succeeded; see {representative_blob}"

    def fake_compile(*call_args):
        member_blob = _write_blob(call_args[5], tasks[1], b"blob")
        return _result(tasks[1], log=f"compilation object save succeeded; see {member_blob}")

    monkeypatch.setattr(compile_shaders, "compile_shader", fake_compile)
    assert verify_deduped_members(tasks, groups, results, args, 100.0) == []


def test_verify_blob_mismatch_is_reported(tmp_path, monkeypatch):
    tasks, groups, results = _canary_setup(tmp_path)
    args = _verify_args(tmp_path)

    def fake_compile(*call_args):
        _write_blob(call_args[5], tasks[1], b"different")
        return _result(tasks[1], log="clean")

    monkeypatch.setattr(compile_shaders, "compile_shader", fake_compile)
    mismatches = verify_deduped_members(tasks, groups, results, args, 100.0)
    assert any("blob differs" in mismatch for mismatch in mismatches)
    assert any("b.hlsl:main:2" in mismatch and "a.hlsl:main:1" in mismatch for mismatch in mismatches)


def test_verify_log_mismatch_is_reported(tmp_path, monkeypatch):
    tasks, groups, results = _canary_setup(tmp_path)
    args = _verify_args(tmp_path)

    def fake_compile(*call_args):
        _write_blob(call_args[5], tasks[1], b"blob")
        return _result(tasks[1], log="different log")

    monkeypatch.setattr(compile_shaders, "compile_shader", fake_compile)
    mismatches = verify_deduped_members(tasks, groups, results, args, 100.0)
    assert any("compiler log differs" in mismatch for mismatch in mismatches)


def test_verify_zero_percent_skips_compilation(tmp_path, monkeypatch):
    tasks, groups, results = _canary_setup(tmp_path)
    args = _verify_args(tmp_path, verify_pct=0)
    compile_mock = MagicMock()
    monkeypatch.setattr(compile_shaders, "compile_shader", compile_mock)

    assert verify_deduped_members(tasks, groups, results, args, 0) == []
    compile_mock.assert_not_called()


def test_run_compilation_dedupe_compiles_one_representative_and_expands(tmp_path, monkeypatch):
    tasks = [TASKS[0], TASKS[1]]
    monkeypatch.setattr(compile_shaders, "initialize_compilation", lambda *args: (1, 1, "manual", tasks))
    monkeypatch.setattr(dedupe, "preprocess_variant", lambda *args, **kwargs: "same text")
    compiled = []

    def fake_compile(*call_args):
        task = (call_args[1], call_args[2], call_args[3], call_args[4])
        compiled.append((task[0], task[2]))
        _write_blob(call_args[5], task, b"blob")
        return _result(task, log="clean")

    monkeypatch.setattr(compile_shaders, "compile_shader", fake_compile)
    args = _run_args(tmp_path, dedupe_enabled=True)

    results = compile_shaders.run_compilation(args, 1, 1, False)

    assert compiled == [("a.hlsl", "main:1")]
    assert [result["entry"] for result in results] == ["main:1", "main:2"]
    assert results[1]["deduped_from"] == "a.hlsl:main:1"
    member_blob = compile_shaders.shader_output_path(args.output_dir, tasks[1][0], tasks[1][1], tasks[1][2])
    assert member_blob is not None
    assert Path(member_blob).read_bytes() == b"blob"


def test_run_compilation_without_dedupe_never_preprocesses(tmp_path, monkeypatch):
    tasks = [TASKS[0], TASKS[1]]
    monkeypatch.setattr(compile_shaders, "initialize_compilation", lambda *args: (1, 1, "manual", tasks))
    preprocess_mock = MagicMock()
    monkeypatch.setattr(dedupe, "preprocess_variant", preprocess_mock)
    compiled = []

    def fake_compile(*call_args):
        task = (call_args[1], call_args[2], call_args[3], call_args[4])
        compiled.append(task[2])
        _write_blob(call_args[5], task, b"blob")
        return _result(task, log="clean")

    monkeypatch.setattr(compile_shaders, "compile_shader", fake_compile)
    args = _run_args(tmp_path, dedupe_enabled=False)

    results = compile_shaders.run_compilation(args, 1, 1, False)

    preprocess_mock.assert_not_called()
    assert compiled == ["main:1", "main:2"]
    assert all("deduped_from" not in result for result in results)


def test_run_compilation_canary_mismatch_sets_failure_flag(tmp_path, monkeypatch):
    tasks = [TASKS[0], TASKS[1]]
    monkeypatch.setattr(compile_shaders, "initialize_compilation", lambda *args: (1, 1, "manual", tasks))
    monkeypatch.setattr(dedupe, "preprocess_variant", lambda *args, **kwargs: "same text")
    calls = []

    def fake_compile(*call_args):
        task = (call_args[1], call_args[2], call_args[3], call_args[4])
        calls.append(task[2])
        data = b"representative" if len(calls) == 1 else b"different"
        _write_blob(call_args[5], task, data)
        return _result(task, log="clean")

    monkeypatch.setattr(compile_shaders, "compile_shader", fake_compile)
    args = _run_args(tmp_path, dedupe_enabled=True, verify_pct=100.0)

    compile_shaders.run_compilation(args, 1, 1, False)

    assert args.dedupe_verify_failed is True


def test_main_returns_nonzero_when_canary_failed(monkeypatch):
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
    monkeypatch.setattr(compile_shaders, "run_compilation", lambda *call_args: [])
    monkeypatch.setattr(compile_shaders, "analyze_and_report_results", lambda *call_args: (0, 0, 0))
    monkeypatch.setattr(compile_shaders, "_maybe_write_cache_manifest", lambda parsed: None)
    monkeypatch.setattr(compile_shaders, "_maybe_write_timing_report", lambda *call_args: None)

    assert compile_shaders.main() == 1


def test_parse_arguments_dedupe_defaults_off():
    with patch("sys.argv", ["compile_shaders.py", "--config", "test.yaml", "--shader-dir", "shaders"]):
        args = compile_shaders.parse_arguments(default_jobs=4)
    assert args.dedupe is False
    assert args.dedupe_verify == 2.0


def test_parse_arguments_dedupe_flags():
    with patch(
        "sys.argv",
        ["compile_shaders.py", "--config", "test.yaml", "--shader-dir", "shaders", "--dedupe", "--dedupe-verify", "10"],
    ):
        args = compile_shaders.parse_arguments(default_jobs=4)
    assert args.dedupe is True
    assert args.dedupe_verify == 10.0


@pytest.mark.parametrize("value", ["-1", "101"])
def test_parse_arguments_rejects_out_of_range_dedupe_verify(value):
    with (
        patch(
            "sys.argv",
            ["compile_shaders.py", "--config", "test.yaml", "--shader-dir", "shaders", "--dedupe-verify", value],
        ),
        pytest.raises(SystemExit),
    ):
        compile_shaders.parse_arguments(default_jobs=4)


def test_variant_key_keeps_invalid_utf8_bytes_distinct():
    """Texts that differ only in undecodable bytes must not share a key."""
    first = variant_key("x\udc80", "PSHADER", "1", False)
    second = variant_key("x\udc81", "PSHADER", "1", False)
    assert first != second


def test_preprocess_variant_decodes_invalid_utf8_losslessly(tmp_path, monkeypatch):
    """Invalid bytes must survive as distinct text instead of collapsing to U+FFFD."""
    shader = tmp_path / "a.hlsl"
    shader.write_text("void main() {}")

    def fake_popen(cmd, **_kwargs):
        Path(cmd[cmd.index("/P") + 1]).write_bytes(b"// \xff\nvoid main() {}\n")
        process = MagicMock()
        process.communicate.return_value = ("", "")
        process.returncode = 0
        return process

    monkeypatch.setattr(dedupe.subprocess, "Popen", fake_popen)
    text = dedupe.preprocess_variant("fxc.exe", str(shader), "main:1", [], str(tmp_path))
    assert text is not None
    assert "�" not in text
    assert variant_key(text, "PSHADER", "1", False) != variant_key("// \xfe\nvoid main() {}\n", "PSHADER", "1", False)


def test_normalize_log_ignores_diagnostic_locations_but_not_messages():
    """Members may report shifted line numbers; the code and message still count."""
    first = "a.hlsl(10,5): warning X4000: use of potentially uninitialized variable (f)"
    shifted = "a.hlsl(12,5-9): warning X4000: use of potentially uninitialized variable (f)"
    other = "a.hlsl(10,5): warning X3206: implicit truncation"
    assert dedupe._normalize_log(first) == dedupe._normalize_log(shifted)
    assert dedupe._normalize_log(first) != dedupe._normalize_log(other)
