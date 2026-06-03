"""Build an HLSL ``#include`` dependency graph and compute affected shaders.

This module enables *incremental* shader validation: given a set of changed
``.hlsl``/``.hlsli`` files, it determines which entry-point shaders must be
recompiled because they transitively include a changed file.

Design notes:

- Includes are parsed with a regex and **preprocessor guards are ignored**
  (an ``#include`` inside an ``#ifdef`` is still treated as a live edge). This
  is intentional: the goal is a *conservative superset* of affected shaders.
  Over-validating wastes time; under-validating ships a broken shader.
- Include paths are resolved the way ``fxc.exe`` resolves them: relative to the
  including file's directory first, then relative to the shader-root include
  dir. Both candidates are recorded as edges when they exist on disk, so the
  graph never misses a real dependency due to ambiguous resolution.
- All graph keys are POSIX-style relative paths (forward slashes) rooted at the
  shader directory, matching the ``file`` field used in hlslkit YAML configs.
"""

import logging
import os
import re

# Matches  #include "Common/Color.hlsli"  (double-quoted, local includes).
# Angle-bracket system includes are intentionally not matched: shader sources
# here use quoted relative includes exclusively.
INCLUDE_REGEX = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)

SHADER_EXTENSIONS = (".hlsl", ".hlsli")


def normalize_rel(rel_path: str) -> str:
    """Normalize a relative path to POSIX form for use as a stable graph key."""
    return os.path.normpath(rel_path).replace(os.sep, "/")


# Backwards-compatible private alias used throughout this module.
_normalize = normalize_rel


def _scan_includes(file_path: str) -> list[str]:
    """Return the raw include strings found in a single shader file.

    Errors reading the file are logged and treated as "no includes" so a single
    unreadable file never aborts graph construction.
    """
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as e:
        logging.warning(f"include_graph: could not read {file_path}: {e}")
        return []
    return INCLUDE_REGEX.findall(content)


def build_include_graph(shader_dir: str) -> dict[str, set[str]]:
    """Build the forward include graph for a shader tree.

    Args:
        shader_dir: Root directory containing the assembled shader sources.

    Returns:
        Mapping of ``relpath -> set(relpath)`` where each value is the set of
        files directly included by the key file. Keys are POSIX-relative to
        ``shader_dir``. Files with no includes still appear as keys (empty set)
        so every shader is represented as a graph node.
    """
    shader_dir = os.path.abspath(shader_dir)
    graph: dict[str, set[str]] = {}

    for dirpath, _dirnames, filenames in os.walk(shader_dir):
        for name in filenames:
            if not name.lower().endswith(SHADER_EXTENSIONS):
                continue
            abs_path = os.path.join(dirpath, name)
            rel_key = _normalize(os.path.relpath(abs_path, shader_dir))
            deps: set[str] = graph.setdefault(rel_key, set())

            for raw in _scan_includes(abs_path):
                resolved = _resolve_include(raw, abs_path, shader_dir)
                if resolved is not None:
                    deps.add(resolved)
                else:
                    # An unresolved include is usually a path we don't model
                    # (e.g. an absolute or generated path). Log at debug so it
                    # is discoverable without spamming normal runs.
                    logging.debug(f"include_graph: unresolved include '{raw}' in {rel_key}")

    return graph


def _resolve_include(raw: str, including_file_abs: str, shader_dir: str) -> str | None:
    """Resolve an include string to a normalized relpath under ``shader_dir``.

    Mirrors fxc resolution order: relative to the including file's directory
    first, then relative to the shader-root include directory. Returns ``None``
    if neither candidate exists on disk (or escapes the shader tree).
    """
    candidates = [
        os.path.join(os.path.dirname(including_file_abs), raw),
        os.path.join(shader_dir, raw),
    ]
    for cand in candidates:
        cand_abs = os.path.abspath(cand)
        if not os.path.isfile(cand_abs):
            continue
        rel = os.path.relpath(cand_abs, shader_dir)
        if rel.startswith(".."):
            # Include resolves outside the shader tree; we can't key it.
            continue
        return _normalize(rel)
    return None


def invert_graph(graph: dict[str, set[str]]) -> dict[str, set[str]]:
    """Invert a forward include graph into ``included -> set(includers)``."""
    reverse: dict[str, set[str]] = {key: set() for key in graph}
    for includer, deps in graph.items():
        for dep in deps:
            reverse.setdefault(dep, set()).add(includer)
    return reverse


def compute_affected_files(graph: dict[str, set[str]], changed: list[str]) -> set[str]:
    """Compute every file affected by a set of changed files.

    A file is affected if it *is* a changed file or transitively includes one.
    The changed files themselves are always included in the result (a changed
    entry-point shader must be recompiled even if nothing includes it).

    Args:
        graph: Forward include graph from :func:`build_include_graph`.
        changed: Changed files as POSIX-relative paths under the shader root.

    Returns:
        Set of affected files as normalized POSIX-relative paths.
    """
    reverse = invert_graph(graph)
    affected: set[str] = set()
    stack = [_normalize(c) for c in changed]

    while stack:
        current = stack.pop()
        if current in affected:
            continue
        affected.add(current)
        # Everything that includes `current` is transitively affected.
        for includer in reverse.get(current, ()):
            if includer not in affected:
                stack.append(includer)

    return affected


def select_affected_entrypoints(
    graph: dict[str, set[str]],
    changed: list[str],
    entrypoint_files: set[str],
) -> set[str]:
    """Return the entry-point files that must be recompiled for ``changed``.

    Args:
        graph: Forward include graph from :func:`build_include_graph`.
        changed: Changed files as POSIX-relative paths under the shader root.
        entrypoint_files: The set of compilable entry-point files (the ``file``
            keys from the YAML config), normalized to POSIX-relative form.

    Returns:
        The subset of ``entrypoint_files`` transitively affected by ``changed``.
    """
    affected = compute_affected_files(graph, changed)
    return {f for f in entrypoint_files if _normalize(f) in affected}
