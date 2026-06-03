"""Tests for the HLSL include dependency graph and incremental selection."""

from hlslkit.compile_shaders import filter_tasks_by_changed_files
from hlslkit.include_graph import (
    build_include_graph,
    compute_affected_files,
    invert_graph,
    select_affected_entrypoints,
)


def _write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_tree(root):
    """Create a small shader tree with a representative include topology.

    Topology (A -> B means "A includes B"):
        Lighting.hlsl   -> Common/BRDF.hlsli -> Common/Math.hlsli
        Water.hlsl      -> Common/Math.hlsli
        Guarded.hlsl    -> Common/Math.hlsli   (inside an #ifdef guard)
        Feature/FooCS.hlsl -> Feature/Local.hlsli
        Standalone.hlsl (no includes)
        Common/Unused.hlsli (included by nothing)
    """
    _write(root / "Common" / "Math.hlsli", "// leaf\n")
    # Sibling-relative include ("Math.hlsli") must resolve against the file dir.
    _write(root / "Common" / "BRDF.hlsli", '#include "Math.hlsli"\n')
    _write(root / "Lighting.hlsl", '#include "Common/BRDF.hlsli"\n')
    _write(root / "Water.hlsl", '#include "Common/Math.hlsli"\n')
    _write(
        root / "Guarded.hlsl",
        '#ifdef SOME_FEATURE\n#include "Common/Math.hlsli"\n#endif\n',
    )
    _write(root / "Feature" / "Local.hlsli", "// leaf\n")
    _write(root / "Feature" / "FooCS.hlsl", '#include "Feature/Local.hlsli"\n')
    _write(root / "Standalone.hlsl", "// nothing\n")
    _write(root / "Common" / "Unused.hlsli", "// orphan\n")


def test_build_graph_resolves_root_and_sibling_includes(tmp_path):
    _make_tree(tmp_path)
    graph = build_include_graph(str(tmp_path))

    assert graph["Lighting.hlsl"] == {"Common/BRDF.hlsli"}
    # Sibling include "Math.hlsli" resolves relative to Common/.
    assert graph["Common/BRDF.hlsli"] == {"Common/Math.hlsli"}
    assert graph["Water.hlsl"] == {"Common/Math.hlsli"}
    assert graph["Common/Math.hlsli"] == set()
    # Every shader file is a node, even leaves and orphans.
    assert "Standalone.hlsl" in graph
    assert "Common/Unused.hlsli" in graph


def test_invert_graph(tmp_path):
    _make_tree(tmp_path)
    reverse = invert_graph(build_include_graph(str(tmp_path)))
    assert reverse["Common/Math.hlsli"] == {"Common/BRDF.hlsli", "Water.hlsl", "Guarded.hlsl"}
    assert reverse["Common/BRDF.hlsli"] == {"Lighting.hlsl"}


def test_affected_is_transitive(tmp_path):
    _make_tree(tmp_path)
    graph = build_include_graph(str(tmp_path))
    affected = compute_affected_files(graph, ["Common/Math.hlsli"])
    # Math itself, its direct includers, and the transitive Lighting.hlsl.
    assert affected == {
        "Common/Math.hlsli",
        "Common/BRDF.hlsli",
        "Lighting.hlsl",
        "Water.hlsl",
        "Guarded.hlsl",
    }


def test_guarded_include_is_conservative(tmp_path):
    """An #include inside an #ifdef is still treated as a live dependency."""
    _make_tree(tmp_path)
    graph = build_include_graph(str(tmp_path))
    assert "Common/Math.hlsli" in graph["Guarded.hlsl"]


def test_leaf_change_affects_single_entrypoint(tmp_path):
    _make_tree(tmp_path)
    graph = build_include_graph(str(tmp_path))
    entrypoints = {"Lighting.hlsl", "Water.hlsl", "Feature/FooCS.hlsl", "Standalone.hlsl", "Guarded.hlsl"}
    affected = select_affected_entrypoints(graph, ["Feature/Local.hlsli"], entrypoints)
    assert affected == {"Feature/FooCS.hlsl"}


def test_shared_change_affects_many_entrypoints(tmp_path):
    _make_tree(tmp_path)
    graph = build_include_graph(str(tmp_path))
    entrypoints = {"Lighting.hlsl", "Water.hlsl", "Feature/FooCS.hlsl", "Standalone.hlsl", "Guarded.hlsl"}
    affected = select_affected_entrypoints(graph, ["Common/Math.hlsli"], entrypoints)
    assert affected == {"Lighting.hlsl", "Water.hlsl", "Guarded.hlsl"}


def test_changed_entrypoint_itself_is_selected(tmp_path):
    _make_tree(tmp_path)
    graph = build_include_graph(str(tmp_path))
    entrypoints = {"Lighting.hlsl", "Water.hlsl", "Standalone.hlsl"}
    # Standalone is included by nothing, but a direct edit must still compile it.
    affected = select_affected_entrypoints(graph, ["Standalone.hlsl"], entrypoints)
    assert affected == {"Standalone.hlsl"}


# --- filter_tasks_by_changed_files (integration with config task tuples) ---


def _tasks():
    """Config-style tasks: (file_name, shader_type, entry, defines)."""
    return [
        ("Lighting.hlsl", "PSHADER", "Lighting:Pixel:0", ["A=1"]),
        ("Lighting.hlsl", "VSHADER", "Lighting:Vertex:0", ["A=1"]),
        ("Water.hlsl", "PSHADER", "Water:Pixel:0", []),
        ("Feature/FooCS.hlsl", "CSHADER", "FooCS:Compute:0", []),
        ("Standalone.hlsl", "PSHADER", "Standalone:Pixel:0", []),
    ]


def test_filter_selects_affected_variants(tmp_path):
    _make_tree(tmp_path)
    tasks, noop = filter_tasks_by_changed_files(_tasks(), ["Common/Math.hlsli"], str(tmp_path))
    assert noop is False
    selected_files = {t[0] for t in tasks}
    assert selected_files == {"Lighting.hlsl", "Water.hlsl"}
    # Both Lighting variants survive.
    assert len([t for t in tasks if t[0] == "Lighting.hlsl"]) == 2


def test_filter_leaf_selects_one_shader(tmp_path):
    _make_tree(tmp_path)
    tasks, noop = filter_tasks_by_changed_files(_tasks(), ["Feature/Local.hlsli"], str(tmp_path))
    assert noop is False
    assert {t[0] for t in tasks} == {"Feature/FooCS.hlsl"}


def test_filter_unknown_file_falls_back_to_full(tmp_path):
    """A changed path not in the tree must NOT narrow validation."""
    _make_tree(tmp_path)
    full = _tasks()
    tasks, noop = filter_tasks_by_changed_files(full, ["DoesNotExist.hlsli"], str(tmp_path))
    assert noop is False
    assert tasks == full


def test_filter_orphan_change_is_noop(tmp_path):
    """Editing a file that no entry-point includes compiles nothing (cleanly)."""
    _make_tree(tmp_path)
    tasks, noop = filter_tasks_by_changed_files(_tasks(), ["Common/Unused.hlsli"], str(tmp_path))
    assert noop is True
    assert tasks == []


def test_filter_backslash_paths_normalized(tmp_path):
    _make_tree(tmp_path)
    tasks, noop = filter_tasks_by_changed_files(_tasks(), ["Common\\Math.hlsli"], str(tmp_path))
    assert noop is False
    assert {t[0] for t in tasks} == {"Lighting.hlsl", "Water.hlsl"}
