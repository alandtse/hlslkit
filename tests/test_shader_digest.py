"""Tests for hlslkit.shader_digest.

The hardcoded hex vectors in the cross-language parity tests were computed
independently in alandtse/open-shaders' C++ runtime (Util::ContentHash,
built against the same xxHash library) and must never be "fixed" to match
whatever this module currently produces -- a mismatch here means the two
implementations have drifted apart, which is exactly what these tests exist
to catch.
"""

import json

from hlslkit.shader_digest import (
    SCHEMA_VERSION,
    build_manifest_entries,
    combine_hashes,
    compute_shader_content_digest,
    hash_file_content,
    hash_string,
    to_hex,
    write_manifest,
)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)


def test_hash_string_matches_cpp_xxh3_128():
    """Cross-validated against Util::ContentHash::HashString("abc").ToHex()."""
    assert to_hex(hash_string("abc")) == "06b05ab6733a618578af5f94892f3950"


def test_combine_hashes_matches_cpp_combinehashes():
    """Cross-validated against Util::ContentHash::CombineHashes(HashString("a"), HashString("b"))."""
    a = hash_string("a")
    b = hash_string("b")
    assert to_hex(combine_hashes(a, b)) == "7abd5dc6efa97faba54ed1749c10d8a0"


def test_combine_hashes_is_order_sensitive():
    a = hash_string("a")
    b = hash_string("b")
    assert combine_hashes(a, b) != combine_hashes(b, a)


def test_hash_string_is_deterministic_and_content_sensitive():
    assert hash_string("abc") == hash_string("abc")
    assert hash_string("abc") != hash_string("abd")


def test_hash_file_content_normalizes_crlf_to_lf(tmp_path):
    crlf = tmp_path / "crlf.hlsli"
    lf = tmp_path / "lf.hlsli"
    _write(crlf, "line1\r\nline2\r\n")
    _write(lf, "line1\nline2\n")
    assert hash_file_content(crlf) == hash_file_content(lf)


def test_hash_file_content_detects_real_changes(tmp_path):
    path = tmp_path / "changed.hlsli"
    _write(path, "value = 1\r\n")
    before = hash_file_content(path)
    _write(path, "value = 2\r\n")
    after = hash_file_content(path)
    assert before != after


def test_hash_file_content_missing_file_is_none(tmp_path):
    assert hash_file_content(tmp_path / "does-not-exist.hlsl") is None


def test_digest_follows_quoted_and_root_relative_includes(tmp_path):
    """Common/Math.hlsli included via a root-relative quoted path."""
    _write(tmp_path / "Common" / "Math.hlsli", "// leaf\n")
    _write(tmp_path / "Lighting.hlsl", '#include "Common/Math.hlsli"\n')

    leaf_hash = hash_file_content(tmp_path / "Common" / "Math.hlsli")
    root_hash = hash_file_content(tmp_path / "Lighting.hlsl")
    expected = combine_hashes(root_hash, leaf_hash)

    digest = compute_shader_content_digest(tmp_path / "Lighting.hlsl", tmp_path)
    assert digest == expected


def test_digest_follows_sibling_relative_includes(tmp_path):
    """A bare "Math.hlsli" resolves against the including file's own directory
    when it doesn't exist at the shader root."""
    _write(tmp_path / "Common" / "Math.hlsli", "// leaf\n")
    _write(tmp_path / "Common" / "BRDF.hlsli", '#include "Math.hlsli"\n')

    leaf_hash = hash_file_content(tmp_path / "Common" / "Math.hlsli")
    self_hash = hash_file_content(tmp_path / "Common" / "BRDF.hlsli")
    expected = combine_hashes(self_hash, leaf_hash)

    digest = compute_shader_content_digest(tmp_path / "Common" / "BRDF.hlsli", tmp_path)
    assert digest == expected


def test_digest_ignores_preprocessor_guards(tmp_path):
    """An #include inside an #ifdef is still a live edge (conservative:
    never under-invalidate), matching GetMaxShaderMTimeInternal's textual scan."""
    _write(tmp_path / "Common" / "Math.hlsli", "// leaf\n")
    _write(tmp_path / "Guarded.hlsl", '#ifdef SOME_FEATURE\n#include "Common/Math.hlsli"\n#endif\n')

    self_hash = hash_file_content(tmp_path / "Guarded.hlsl")
    leaf_hash = hash_file_content(tmp_path / "Common" / "Math.hlsli")
    expected = combine_hashes(self_hash, leaf_hash)

    digest = compute_shader_content_digest(tmp_path / "Guarded.hlsl", tmp_path)
    assert digest == expected


def test_digest_changes_when_an_included_file_changes(tmp_path):
    _write(tmp_path / "Common" / "Math.hlsli", "// v1\n")
    _write(tmp_path / "Lighting.hlsl", '#include "Common/Math.hlsli"\n')
    before = compute_shader_content_digest(tmp_path / "Lighting.hlsl", tmp_path)

    _write(tmp_path / "Common" / "Math.hlsli", "// v2\n")
    after = compute_shader_content_digest(tmp_path / "Lighting.hlsl", tmp_path)

    assert before != after


def test_digest_is_stable_regardless_of_include_declaration_order(tmp_path):
    _write(tmp_path / "Common" / "A.hlsli", "// a\n")
    _write(tmp_path / "Common" / "B.hlsli", "// b\n")
    _write(tmp_path / "One.hlsl", '#include "Common/A.hlsli"\n#include "Common/B.hlsli"\n')
    _write(tmp_path / "Two.hlsl", '#include "Common/B.hlsli"\n#include "Common/A.hlsli"\n')

    # Same include SET, declared in opposite order in the source text; since
    # both shader files are otherwise identical (only their #include lines
    # differ in order) the combine must sort children so both files end up
    # combining the same two child hashes in the same resulting order.
    one_includes_hash = compute_shader_content_digest(tmp_path / "One.hlsl", tmp_path)
    two_includes_hash = compute_shader_content_digest(tmp_path / "Two.hlsl", tmp_path)

    a_hash = hash_file_content(tmp_path / "Common" / "A.hlsli")
    b_hash = hash_file_content(tmp_path / "Common" / "B.hlsli")
    one_self = hash_file_content(tmp_path / "One.hlsl")
    two_self = hash_file_content(tmp_path / "Two.hlsl")

    expected_one = combine_hashes(combine_hashes(one_self, a_hash), b_hash)
    expected_two = combine_hashes(combine_hashes(two_self, a_hash), b_hash)

    assert one_includes_hash == expected_one
    assert two_includes_hash == expected_two


def test_digest_handles_missing_root_gracefully(tmp_path):
    assert compute_shader_content_digest(tmp_path / "missing.hlsl", tmp_path) is None


def test_digest_skips_unreadable_include_without_failing(tmp_path):
    """An #include for an uninstalled/optional feature that resolves to no
    file on disk at all is simply skipped, matching the runtime's textual scan."""
    _write(tmp_path / "One.hlsl", '#include "Feature/NotInstalled.hlsli"\n')
    digest = compute_shader_content_digest(tmp_path / "One.hlsl", tmp_path)
    assert digest == hash_file_content(tmp_path / "One.hlsl")


def test_digest_handles_include_cycles_without_hanging(tmp_path):
    _write(tmp_path / "A.hlsli", '#include "B.hlsli"\n')
    _write(tmp_path / "B.hlsli", '#include "A.hlsli"\n')
    digest = compute_shader_content_digest(tmp_path / "A.hlsli", tmp_path)
    assert digest is not None  # must terminate and produce a real value


def test_build_manifest_entries_skips_blobs_with_no_matching_source(tmp_path):
    cache_dir = tmp_path / "ShaderCache"
    shader_dir = tmp_path / "Shaders"
    _write(shader_dir / "Lighting.hlsl", "// v1\n")
    _write(cache_dir / "Lighting" / "1A2B.pso", b"\x00\x01")
    _write(cache_dir / "Orphaned" / "0.pso", b"\x00\x01")  # no Orphaned.hlsl

    entries = build_manifest_entries(cache_dir, shader_dir, "")

    assert "Lighting/1A2B.pso" in entries
    assert not any(k.startswith("Orphaned/") for k in entries)


def test_build_manifest_entries_uses_resolve_source_name_when_given(tmp_path):
    """ImageSpace-style case: the cache directory (named by runtime technique)
    doesn't match any source file directly, but a caller-supplied resolver
    (e.g. build-shader-cache.py's IMAGESPACE_DIRS table) maps it to the real
    source stem it was actually compiled from."""
    cache_dir = tmp_path / "ShaderCache"
    shader_dir = tmp_path / "Shaders"
    _write(shader_dir / "ISCompositeLensFlareVolumetricLighting.hlsl", "// v1\n")
    _write(cache_dir / "ISCompositeLensFlare" / "73.vso", b"\x00\x01")

    without_resolver = build_manifest_entries(cache_dir, shader_dir, "")
    assert without_resolver == {}

    with_resolver = build_manifest_entries(
        cache_dir,
        shader_dir,
        "",
        resolve_source_name=lambda name: {"ISCompositeLensFlare": "ISCompositeLensFlareVolumetricLighting"}.get(
            name, name
        ),
    )
    assert "ISCompositeLensFlare/73.vso" in with_resolver


def test_build_manifest_entries_uses_posix_separators(tmp_path):
    cache_dir = tmp_path / "ShaderCache"
    shader_dir = tmp_path / "Shaders"
    _write(shader_dir / "Lighting.hlsl", "// v1\n")
    _write(cache_dir / "Lighting" / "1A2B.pso", b"\x00\x01")

    entries = build_manifest_entries(cache_dir, shader_dir, "")
    assert set(entries) == {"Lighting/1A2B.pso"}


def test_build_manifest_entries_folds_global_defines_state(tmp_path):
    cache_dir = tmp_path / "ShaderCache"
    shader_dir = tmp_path / "Shaders"
    _write(shader_dir / "Lighting.hlsl", "// v1\n")
    _write(cache_dir / "Lighting" / "1A2B.pso", b"\x00\x01")

    se_entries = build_manifest_entries(cache_dir, shader_dir, "")
    vr_entries = build_manifest_entries(cache_dir, shader_dir, "VR;")

    assert se_entries["Lighting/1A2B.pso"] != vr_entries["Lighting/1A2B.pso"]


def test_write_manifest_matches_runtime_schema(tmp_path):
    cache_dir = tmp_path / "ShaderCache"
    shader_dir = tmp_path / "Shaders"
    manifest_path = cache_dir / "Manifest.json"
    _write(shader_dir / "Lighting.hlsl", "// v1\n")
    _write(cache_dir / "Lighting" / "1A2B.pso", b"\x00\x01")

    count = write_manifest(cache_dir, shader_dir, "", manifest_path)

    assert count == 1
    parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert parsed["schemaVersion"] == SCHEMA_VERSION
    assert set(parsed["entries"]) == {"Lighting/1A2B.pso"}
    assert len(parsed["entries"]["Lighting/1A2B.pso"]) == 32
