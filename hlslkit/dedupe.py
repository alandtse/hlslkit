"""Content deduplication for shader compilation variants.

Many variants preprocess to identical text, so one fxc invocation can serve all
of them. This module mirrors the Open Shaders plugin's ``Util::CompileDedupe``
grouping rule: strip ``#line`` directives from each variant's ``fxc /P`` output,
then group by the resulting text plus the compile flags that affect code
generation. A group's first variant is compiled and its blob and diagnostics are
reused for the rest.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
from collections import deque
from collections.abc import Callable
from pathlib import Path

from hlslkit import compile_shaders
from hlslkit.shader_digest import hash_bytes

# About-percent verification still needs a hard ceiling: 2% of a large cache is
# thousands of extra compiles, which would erase the feature's whole speedup.
_CANARY_MAX = 20
_CANARY_SEED = 0
_DEFAULT_VERIFY_PCT = 2.0
_DIAGNOSTIC_LOCATION = re.compile(r"\(\d+(?:,\d+(?:-\d+)?)?\)")


def strip_line_directives(text: str) -> str:
    """Drop lines that start exactly with ``#line``, keeping all else intact.

    Mirrors the C++ rule ``line.starts_with("#line")`` on the raw line: no
    leading-whitespace trimming and no whitespace or line-ending normalization.

    Args:
        text: Preprocessed shader text.

    Returns:
        str: The text without ``#line`` directives, byte-for-byte otherwise.
    """
    # splitlines() also breaks on \r, \f, \v and Unicode separators, which would drop a
    # "#line" the C++ rule keeps; only "\n" ends a line there.
    lines = re.findall(r"[^\n]*\n|[^\n]+", text)
    return "".join(line for line in lines if not line.startswith("#line"))


def variant_key(
    normalized_text: str,
    shader_type: str,
    optimization_level: str,
    force_partial_precision: bool,
) -> tuple[int, str, str, bool]:
    """Return a hashable identity for a compile variant.

    Args:
        normalized_text: Preprocessed text after :func:`strip_line_directives`.
        shader_type: Shader type (VSHADER, PSHADER, CSHADER).
        optimization_level: fxc optimization level.
        force_partial_precision: Whether ``/Gfp`` is passed.

    Returns:
        tuple[int, str, str, bool]: Text hash, shader model, optimization level
        and partial-precision flag.
    """
    return (
        hash_bytes(normalized_text.encode("utf-8", errors="surrogateescape")),
        compile_shaders.shader_model(shader_type),
        str(optimization_level),
        bool(force_partial_precision),
    )


def preprocess_variant(
    fxc_path: str,
    shader_file: str,
    entry: str,
    defines: list[str],
    shader_dir: str,
    strip_debug_defines: bool = False,
    debug_defines: set[str] | None = None,
    extra_includes: list[str] | None = None,
) -> str | None:
    """Run ``fxc /P`` for one variant and return its raw preprocessed text.

    Any failure (non-zero exit, missing output, exception) returns None so the
    caller compiles that variant normally instead of skipping it.

    Args:
        fxc_path: Path to fxc.exe.
        shader_file: Shader file as given in the task.
        entry: Entry id, used only in debug logging.
        defines: Preprocessor defines before debug stripping.
        shader_dir: Directory containing shader files, or the shader file itself.
        strip_debug_defines: Apply the same debug-define stripping as compile_shader.
        debug_defines: Defines treated as debug defines.
        extra_includes: Additional include directories.

    Returns:
        str | None: Raw ``/P`` output, or None when preprocessing failed.
    """
    if compile_shaders.stop_event.is_set():
        return None

    temp_dir = tempfile.mkdtemp(prefix="hlslkit-preprocess-")
    output_path = os.path.join(temp_dir, "preprocessed.hlsl")
    process = None
    try:
        cmd, compile_cwd, shader_path = compile_shaders.build_preprocess_invocation(
            fxc_path,
            shader_file,
            entry,
            defines,
            shader_dir,
            output_path,
            strip_debug_defines=strip_debug_defines,
            debug_defines=debug_defines,
            extra_includes=extra_includes,
        )
        if not os.path.exists(shader_path):
            logging.debug(f"Cannot preprocess missing shader {shader_file}:{entry}")
            return None

        logging.debug(f"Preprocessing command: {' '.join(cmd)}")
        process = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=compile_cwd,
        )
        with compile_shaders.running_processes_lock:
            compile_shaders.running_processes.add(process)
        _stdout, _stderr = process.communicate()
        if process.returncode != 0 or not os.path.exists(output_path):
            logging.debug(f"Preprocessing failed for {shader_file}:{entry}")
            return None
        with open(output_path, encoding="utf-8", errors="surrogateescape", newline="") as f:
            return f.read()
    except Exception:
        logging.exception(f"Preprocessing failed for {shader_file}:{entry}")
        return None
    finally:
        with compile_shaders.running_processes_lock:
            if process in compile_shaders.running_processes:
                compile_shaders.running_processes.remove(process)
        shutil.rmtree(temp_dir, ignore_errors=True)


def group_variants(
    tasks: list[tuple],
    preprocess: Callable[[tuple], str | None],
    optimization_level: str,
    force_partial_precision: bool,
    max_workers: int,
) -> tuple[list[list[int]], int]:
    """Group task indices by identical preprocessed text and compile flags.

    Args:
        tasks: (file, type, entry, defines) tasks.
        preprocess: Callable taking a task and returning preprocessed text or None.
        optimization_level: fxc optimization level shared by every task.
        force_partial_precision: Whether ``/Gfp`` is shared by every task.
        max_workers: Worker-pool size to reuse for preprocessing.

    Returns:
        tuple[list[list[int]], int]: Groups of task indices in task order (each
        group's first index is its representative) and the number of
        preprocessing failures.
    """
    keys: list[object] = [None] * len(tasks)
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(preprocess, task): index for index, task in enumerate(tasks)}
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                text = future.result()
            except Exception:
                logging.exception(f"Dedupe preprocessing raised for {tasks[index][0]}:{tasks[index][2]}")
                text = None
            if text is None:
                failures += 1
                # A unique key keeps a failed variant as its own group so it is
                # still compiled normally rather than merged or skipped.
                keys[index] = ("preprocess-failed", index)
            else:
                keys[index] = variant_key(
                    strip_line_directives(text),
                    tasks[index][1],
                    optimization_level,
                    force_partial_precision,
                )

    groups_by_key: dict[object, list[int]] = {}
    for index, key in enumerate(keys):
        groups_by_key.setdefault(key, []).append(index)
    return list(groups_by_key.values()), failures


def _index_results(results: list[dict]) -> dict[tuple[str, str, str], deque[dict]]:
    """Index results by (file, entry, type) so representatives can be matched."""
    indexed: dict[tuple[str, str, str], deque[dict]] = {}
    for result in results:
        identity = (result["file"], result["entry"], result["type"])
        indexed.setdefault(identity, deque()).append(result)
    return indexed


def _failed_result(task: tuple, log: str) -> dict[str, object]:
    """Build a failed result for a representative whose future produced none."""
    return {
        "file": task[0],
        "entry": task[2],
        "type": task[1],
        "log": log,
        "success": False,
        "cmd": [],
        "duration_seconds": 0.0,
    }


def expand_results(
    tasks: list[tuple],
    groups: list[list[int]],
    results: list[dict],
    output_dir: str,
) -> list[dict]:
    """Expand representative results to one entry per original task in order.

    Args:
        tasks: Original (file, type, entry, defines) tasks.
        groups: Task-index groups from :func:`group_variants`.
        results: Results for the compiled representatives, in completion order.
        output_dir: Directory the representatives were compiled into.

    Returns:
        list[dict]: One result per original task, ordered like ``tasks``.
    """
    indexed_results = _index_results(results)
    expanded: list[dict | None] = [None] * len(tasks)

    for group in groups:
        representative_index = group[0]
        representative_task = tasks[representative_index]
        identity = (representative_task[0], representative_task[2], representative_task[1])
        queued = indexed_results.get(identity)
        representative_result = queued.popleft() if queued else None
        if representative_result is None:
            representative_result = _failed_result(representative_task, "Compilation result missing.")

        expanded[representative_index] = representative_result
        representative_key = f"{representative_task[0]}:{representative_task[2]}"
        representative_blob = compile_shaders.shader_output_path(
            output_dir, representative_task[0], representative_task[1], representative_task[2]
        )

        for member_index in group[1:]:
            member_task = tasks[member_index]
            member_result = dict(representative_result)
            member_result["file"] = member_task[0]
            member_result["entry"] = member_task[2]
            member_result["type"] = member_task[1]
            member_result["duration_seconds"] = 0.0
            member_result["deduped_from"] = representative_key
            if representative_result.get("success"):
                member_blob = compile_shaders.shader_output_path(
                    output_dir, member_task[0], member_task[1], member_task[2]
                )
                if representative_blob is None or member_blob is None:
                    member_result["success"] = False
                else:
                    try:
                        os.makedirs(os.path.dirname(member_blob), exist_ok=True)
                        shutil.copyfile(representative_blob, member_blob)
                    except OSError:
                        logging.exception(
                            f"Failed to copy deduplicated blob from {representative_key} to {member_task[0]}:{member_task[2]}"
                        )
                        member_result["success"] = False
            expanded[member_index] = member_result

    # Groups cover every task index; fill any gap so the result shape always
    # matches the original task list even if a future vanished mid-run.
    for index, result in enumerate(expanded):
        if result is None:
            expanded[index] = _failed_result(tasks[index], "Compilation result missing.")
    return [result for result in expanded if result is not None]


def select_canary_members(groups: list[list[int]], results: list[dict], verify_pct: float) -> list[tuple[int, int]]:
    """Pick (member, representative) index pairs to verify.

    Args:
        groups: Task-index groups from :func:`group_variants`.
        results: Expanded results, one per original task.
        verify_pct: Percentage of non-representative members to recompile.

    Returns:
        list[tuple[int, int]]: Deterministically sampled pairs, empty when
        verification is disabled or there are no deduplicated members.
    """
    if verify_pct <= 0:
        return []
    candidates = [(member, group[0]) for group in groups if results[group[0]].get("success") for member in group[1:]]
    if not candidates:
        return []
    count = min(_CANARY_MAX, max(1, math.ceil(len(candidates) * verify_pct / 100)))
    # Seeded sampling is deliberately deterministic; nothing security-sensitive.
    return random.Random(_CANARY_SEED).sample(candidates, count)  # noqa: S311


def _normalize_log(log: str, *output_paths: str | None) -> str:
    """Blank out the parts of an fxc log that legitimately differ inside a group.

    fxc prints the /Fo path on success and on a write failure, and the (line,col)
    of each diagnostic follows the #line directives that dedupe ignores; the
    diagnostic codes and messages themselves are still compared.
    """
    for path in output_paths:
        if path:
            log = log.replace(path, "<output>")
    return _DIAGNOSTIC_LOCATION.sub("(L)", log)


def _blob_matches(representative_blob: str | None, member_blob: str | None) -> bool:
    """Whether both blobs exist and contain identical bytes."""
    if not representative_blob or not member_blob:
        return False
    try:
        return Path(representative_blob).read_bytes() == Path(member_blob).read_bytes()
    except OSError:
        return False


def verify_deduped_members(
    tasks: list[tuple],
    groups: list[list[int]],
    results: list[dict],
    args: argparse.Namespace,
    verify_pct: float = _DEFAULT_VERIFY_PCT,
) -> list[str]:
    """Recompile sampled deduplicated variants and compare them to their representative.

    Args:
        tasks: Original (file, type, entry, defines) tasks.
        groups: Task-index groups from :func:`group_variants`.
        results: Expanded results, one per original task.
        args: Parsed CLI arguments (supplies fxc and compile settings).
        verify_pct: Percentage of non-representative members to check.

    Returns:
        list[str]: Human-readable mismatch messages; empty when all checks pass.
    """
    chosen = select_canary_members(groups, results, verify_pct)
    if not chosen:
        return []

    mismatches: list[str] = []
    temp_dir = tempfile.mkdtemp(prefix="hlslkit-dedupe-verify-")
    try:
        for member_index, representative_index in chosen:
            member_task = tasks[member_index]
            representative_task = tasks[representative_index]
            member_name = f"{member_task[0]}:{member_task[2]}"
            representative_name = f"{representative_task[0]}:{representative_task[2]}"
            outcome = compile_shaders.compile_shader(
                *compile_shaders._compile_task_args(member_task, args, args.shader_dir, temp_dir)
            )

            if not outcome.get("success"):
                mismatches.append(
                    f"Dedupe verification failed for {member_name} (deduped from {representative_name}): "
                    f"recompile failed"
                )
                continue
            member_blob = compile_shaders.shader_output_path(temp_dir, member_task[0], member_task[1], member_task[2])
            representative_blob = compile_shaders.shader_output_path(
                os.path.abspath(args.output_dir), representative_task[0], representative_task[1], representative_task[2]
            )
            if not _blob_matches(representative_blob, member_blob):
                mismatches.append(
                    f"Dedupe verification mismatch for {member_name} (deduped from {representative_name}): "
                    f"compiled blob differs"
                )
            member_log = _normalize_log(str(outcome.get("log", "")), representative_blob, member_blob)
            representative_log = _normalize_log(
                str(results[representative_index].get("log", "")), representative_blob, member_blob
            )
            if member_log != representative_log:
                mismatches.append(
                    f"Dedupe verification mismatch for {member_name} (deduped from {representative_name}): "
                    f"compiler log differs"
                )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return mismatches
