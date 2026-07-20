"""XXH3-128 content-digest manifest for a compiled shader disk cache.

Byte-for-byte mirror of ``Util::ContentHash`` and ``GetShaderContentDigest``
in alandtse/open-shaders' ``src/Utils/ContentHash.h`` / ``src/ShaderCache.cpp``,
so a cache built here validates against the runtime's manifest-first
disk-cache check instead of falling back to its (fragile) mtime comparison.
Any change to the digest algorithm on either side must be mirrored on the
other -- a mismatch is silently safe (the runtime just recompiles that
shader) but defeats the entire point of shipping a manifest.

Windows-only by design: the include-closure sort order must match the C++
side's native (backslash) path separators and case-insensitive comparison,
which only holds running on Windows -- matching this project's windows-2025
CI runner for the prebuilt-cache release workflow.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import xxhash

SCHEMA_VERSION = 1
_CACHE_EXTENSIONS = (".pso", ".vso", ".cso")
_INCLUDE_KEYWORD = "include"


def hash_bytes(data: bytes) -> int:
    """XXH3-128 of raw bytes as a single 128-bit int: (high64 << 64) | low64."""
    return xxhash.xxh3_128(data).intdigest()


def hash_string(text: str) -> int:
    return hash_bytes(text.encode("utf-8"))


def combine_hashes(a: int, b: int) -> int:
    """Order-sensitive combine, matching ContentHash.h's CombineHashes: hashes
    the native little-endian byte layout of {a.high, a.low, b.high, b.low}."""
    mask = 0xFFFFFFFFFFFFFFFF
    a_hi, a_lo = (a >> 64) & mask, a & mask
    b_hi, b_lo = (b >> 64) & mask, b & mask
    return hash_bytes(struct.pack("<4Q", a_hi, a_lo, b_hi, b_lo))


def to_hex(digest: int) -> str:
    return format(digest, "032x")


def hash_file_content(path: Path) -> int | None:
    """CRLF -> LF normalized content hash, matching ContentHash.h's HashFile.

    None on any read failure; callers must treat that as "digest unknown",
    never "unchanged".
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    out = bytearray()
    i, n = 0, len(raw)
    while i < n:
        if raw[i] == 0x0D and i + 1 < n and raw[i + 1] == 0x0A:
            i += 1
            continue
        out.append(raw[i])
        i += 1
    return hash_bytes(bytes(out))


def _normalized_key(path: Path) -> str:
    """Matches ShaderCache.cpp's NormalizedPathKey: path-normalized then
    lowercased. Relies on Windows' native backslash separators to get the
    same relative sort order the C++ side gets from
    std::filesystem::path::string() -- this only holds on Windows.
    """
    return os.path.normpath(str(path)).lower()


def _parse_includes(path: Path) -> list[str]:
    """Extract quoted or angle-bracket #include targets, one scan per line.

    Mirrors GetMaxShaderMTimeInternal's textual (preprocessor-blind) scan in
    ShaderCache.cpp exactly, including its permissiveness (no word-boundary
    check after "include", angle-bracket accepted). Intentionally NOT the
    same rules as include_graph.py's quoted-only regex: this module's
    contract is byte-for-byte parity with the runtime, not with hlslkit's
    own validation-focused include graph.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    names: list[str] = []
    for line in text.splitlines():
        rest = line.lstrip(" \t")
        if not rest.startswith("#"):
            continue
        rest = rest[1:].lstrip(" \t")
        if not rest.startswith(_INCLUDE_KEYWORD):
            continue
        rest = rest[len(_INCLUDE_KEYWORD) :]
        open_idx = None
        open_char = None
        for i, c in enumerate(rest):
            if c in ('"', "<"):
                open_idx = i
                open_char = c
                break
        if open_idx is None:
            continue
        close_char = '"' if open_char == '"' else ">"
        close_idx = rest.find(close_char, open_idx + 1)
        if close_idx == -1 or close_idx == open_idx + 1:
            continue
        names.append(rest[open_idx + 1 : close_idx])
    return names


def _resolve_include(name: str, shader_root: Path, including_file: Path) -> Path | None:
    """Root-relative first, then the including file's own directory.

    Matches ShaderCache.cpp's resolution order exactly -- the opposite
    priority from include_graph.py's fxc-mirroring order, which tries the
    including file's directory first.
    """
    root_candidate = shader_root / name
    if root_candidate.is_file():
        return root_candidate
    parent_candidate = including_file.parent / name
    if parent_candidate.is_file():
        return parent_candidate
    return None


def compute_shader_content_digest(
    path: Path,
    shader_root: Path,
    _results: dict[str, int | None] | None = None,
) -> int | None:
    """Merkle-style digest over a shader source and its recursive quoted or
    angle-bracket includes, path-sorted so ordering doesn't affect the
    result. None only if the root file is unreadable; an unreadable
    descendant is skipped, not fatal to the whole digest -- matches
    GetShaderContentDigestInternal in ShaderCache.cpp, including its
    cycle guard (a cycle contributes nothing extra to its own combine).
    """
    if _results is None:
        _results = {}
    key = _normalized_key(path)
    if key in _results:
        return _results[key]
    _results[key] = None  # in-progress marker; also the final value on failure

    self_hash = hash_file_content(path)
    if self_hash is None:
        return None

    resolved = []
    for name in _parse_includes(path):
        target = _resolve_include(name, shader_root, path)
        if target is not None:
            resolved.append(target)
    resolved.sort(key=_normalized_key)

    combined = self_hash
    for inc in resolved:
        child = compute_shader_content_digest(inc, shader_root, _results)
        if child is not None:
            combined = combine_hashes(combined, child)

    _results[key] = combined
    return combined


def build_manifest_entries(cache_dir: Path, shader_root: Path, global_defines_state: str) -> dict[str, str]:
    """Compute manifest entries for every compiled blob under cache_dir.

    Args:
        cache_dir: Root of the compiled cache (ShaderCache/<Name>/<descriptor>.ext).
        shader_root: Root of the shader source tree (Data/Shaders equivalent).
        global_defines_state: Must match what GetGlobalDefinesDigest() builds
            at comparison time for this install -- "" for a default SE
            install, "VR;" for VR. This tool never targets Developer Mode or
            a custom Shader Defines string, since a release-parity prebuilt
            cache assumes neither is active on the machine that built it.

    Returns:
        Mapping of blob path (relative to cache_dir, POSIX separators,
        matching GetManifestKey) to its 32-hex-char combined digest. A blob
        whose source shader can't be found or read is left unrecorded --
        the runtime then falls back to its mtime check for that blob, which
        is always safe, just not the fast path.
    """
    defines_digest = hash_string(global_defines_state)
    digest_cache: dict[str, int | None] = {}
    entries: dict[str, str] = {}
    for blob in sorted(cache_dir.rglob("*")):
        if not blob.is_file() or blob.suffix.lower() not in _CACHE_EXTENSIONS:
            continue
        shader_name = blob.parent.name
        source = shader_root / f"{shader_name}.hlsl"
        if not source.is_file():
            continue
        digest = compute_shader_content_digest(source, shader_root, digest_cache)
        if digest is None:
            continue
        combined = combine_hashes(digest, defines_digest)
        relative_key = blob.relative_to(cache_dir).as_posix()
        entries[relative_key] = to_hex(combined)
    return entries


def write_manifest(cache_dir: Path, shader_root: Path, global_defines_state: str, manifest_path: Path) -> int:
    """Write Manifest.json in the schema Util::ShaderCacheManifest::Manifest
    expects. Returns the number of entries written."""
    if sys.platform != "win32":
        # this is the one thing a future maintainer must know before "fixing" this check.
        raise RuntimeError(  # noqa: TRY003
            "shader_digest requires Windows: its include-closure sort order "
            "depends on native backslash path separators to match the C++ "
            "runtime's digest exactly. Running it elsewhere would silently "
            "produce a manifest that never validates."
        )
    entries = build_manifest_entries(cache_dir, shader_root, global_defines_state)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"schemaVersion": SCHEMA_VERSION, "entries": entries}, sort_keys=True),
        encoding="utf-8",
    )
    return len(entries)
