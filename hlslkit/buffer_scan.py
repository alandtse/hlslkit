import argparse
import difflib
import io
import logging
import os
import re
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from re import Match, Pattern
from typing import Any, Optional, TypeAlias

import jellyfish
import pcpp
from py_markdown_table.markdown_table import markdown_table

try:
    import pathspec
except ImportError:
    pathspec = None

# === Module-level Debug Storage ===
DEBUG_INFO = []


def add_debug_info(message: str) -> None:
    """Add debug information to be included in the output."""
    DEBUG_INFO.append(message)


def clear_debug_info() -> None:
    """Clear collected debug information."""
    DEBUG_INFO.clear()


# Type aliases
StructDict: TypeAlias = dict[str, Any]
FieldDict: TypeAlias = dict[str, Any]
ResultMap: TypeAlias = dict[str, dict[str, Any]]
CompilationUnits: TypeAlias = dict[tuple[str, frozenset[str]], dict[str, set[str]]]


# === Data Classes ===
@dataclass
class StructCandidate:
    """Represents a candidate C++ struct for matching with an HLSL struct."""

    name: str
    data: StructDict
    score: float
    align_matches: list[tuple[FieldDict | None, FieldDict | None]]
    report: dict[str, Any]


@dataclass
class StructMatch:
    """Represents a match between an HLSL struct and a C++ struct."""

    hlsl_name: str
    hlsl_file: str
    hlsl_line: int
    cpp_name: str
    cpp_file: str
    cpp_line: int
    score: float
    align_matches: list[tuple[FieldDict | None, FieldDict | None]]
    report: dict[str, Any]
    candidates: list[StructCandidate]

    @property
    def is_matched(self) -> bool:
        """True if there's a valid C++ match."""
        return bool(self.cpp_name and self.score > 0)


@dataclass
class AnalysisLink:
    """Represents a link to struct analysis results."""

    link: str
    is_match: bool
    cpp_name: str
    cpp_file: str
    cpp_line: int
    score: float
    status: str


# === Module-level Constants (Magic Numbers) ===
# Alignment and size constants
ALIGN_TO_16 = 16  # 16-byte alignment for cbuffers/ConstantBuffers
ALIGN_TO_4 = 4  # 4-byte alignment for HLSL fields
POINTER_SIZE = 8  # Assume 64-bit pointers for C++

# Similarity thresholds
NAME_SIM_THRESHOLD = 0.70  # Threshold for fuzzy LCS field name similarity
JARO_WINKLER_PREFIX_SCALE = 0.1  # Scaling factor for Jaro-Winkler prefix
JARO_WINKLER_THRESHOLD = 0.7  # Threshold for Winkler boost
MAX_PREFIX_LENGTH = 4  # Max prefix length for Winkler boost

# Default size for unknown types
DEFAULT_TYPE_SIZE = 4


class FileScanner:
    """Class to handle file scanning operations."""

    def __init__(self, cwd: str):
        """Initialize the scanner with current working directory.

        Args:
            cwd: Current working directory path.
        """
        self.cwd = cwd
        self.excluded_dirs = get_excluded_dirs(cwd)

    def _get_short_path(self, full_path: str) -> str:
        """Get shortened path relative to skyrim-community-shaders or cwd."""
        full_path = full_path.replace("\\", "/")
        short_path_start = full_path.lower().find("skyrim-community-shaders")
        if short_path_start != -1:
            return full_path[short_path_start + len("skyrim-community-shaders") + 1 :]
        return os.path.relpath(full_path, self.cwd).replace("\\", "/")

    def scan_for_buffers(
        self,
        pattern: Pattern[str],
        feature_pattern: Pattern[str],
        shader_pattern: Pattern[str],
        hlsl_types: dict[str, str],
        defines_list: list[dict[str, str]],
    ) -> tuple[list[dict[str, Any]], dict[tuple[str, frozenset[str]], dict[str, set[str]]]]:
        """Scan HLSL files for buffer definitions and track compilation units.

        Args:
            pattern: Compiled regex pattern to match HLSL files.
            feature_pattern: Compiled regex pattern to extract feature names.
            shader_pattern: Compiled regex pattern for buffer definitions.
            hlsl_types: Mapping of HLSL register types to descriptions.
            defines_list: List of preprocessor define dictionaries.

        Returns:
            A tuple containing:
            - List of buffer entries
            - Dictionary of compilation units
        """
        result_map: dict[str, dict[str, Any]] = {}
        compilation_units: dict[tuple[str, frozenset[str]], dict[str, set[str]]] = {}

        for root, dirs, files in os.walk(self.cwd):
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            feature = ""
            root_normalized = root.replace("\\", "/")
            feature_match = feature_pattern.search(root_normalized)
            if feature_match:
                feature = feature_match.group("feature")

            for file in files:
                if file.lower().endswith((".hlsl", ".hlsli")):
                    full_path = os.path.join(root, file).replace("\\", "/")
                    short_path = self._get_short_path(full_path)
                    logging.debug(f"Processing file: {full_path}")

                    for defines in defines_list:
                        process_file(
                            full_path,
                            self.cwd,
                            defines,
                            shader_pattern,
                            hlsl_types,
                            feature,
                            short_path,
                            result_map,
                            compilation_units,
                        )

        results = list(result_map.values())
        logging.debug(f"Scan found {len(results)} buffers")
        return results, compilation_units

    def scan_for_structs(
        self,
    ) -> tuple[dict[str, list[StructDict]], dict[str, list[StructDict]]]:
        """Scan directory for HLSL and C++ files and extract structs."""
        hlsl_structs: dict[str, list[StructDict]] = {}
        cpp_structs: dict[str, list[StructDict]] = {}

        for root, dirs, files in os.walk(self.cwd):
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            for f in files:
                if not f.lower().endswith((".cpp", ".h", ".hpp", ".hlsl", ".hlsli")):
                    continue
                full_path = os.path.normpath(os.path.join(root, f))
                try:
                    with open(full_path, encoding="utf-8", errors="ignore") as file:
                        content = file.read()
                    short_path = self._get_short_path(full_path)
                    is_hlsl = full_path.lower().endswith((".hlsl", ".hlsli"))
                    structs = extract_structs(content, is_hlsl, short_path)
                    target = hlsl_structs if is_hlsl else cpp_structs
                    for name, data in structs.items():
                        if is_shader_io_struct(name):
                            logging.debug(f"Skipping shader IO buffer: {name} in {short_path}")
                            continue

                        if not isinstance(data, dict):
                            logging.error(f"Invalid struct data for {name} in {short_path}: {data}")
                            continue
                        data["name"] = name

                        # Check if we already have a real definition and this is a template
                        if name in target and data.get("is_template", False):
                            # Check if any existing definition is not a template (has fields)
                            has_real_definition = any(
                                existing.get("fields") and not existing.get("is_template", False)
                                for existing in target[name]
                            )
                            if has_real_definition:
                                logging.debug(f"Skipping template {name} - real struct definition already exists")
                                continue

                        if name not in target:
                            target[name] = []
                        target[name].append(data)
                        logging.debug(f"Added {name} from {short_path} to {'hlsl' if is_hlsl else 'cpp'} structs")
                except Exception as e:
                    logging.warning(f"Failed to read or process file {full_path}: {e}")

        return hlsl_structs, cpp_structs


def create_link(text: str, line: int | None = None) -> str:
    """Generate a GitHub link for a given file path and optional line number.

    Args:
        text: The file path to convert to a GitHub URL.
        line: Line number to link to.

    Returns:
        str: A URL pointing to the file in the skyrim-community-shaders repository.
    """
    base_url = f"https://github.com/doodlum/skyrim-community-shaders/blob/dev/{urllib.parse.quote(text)}"
    if line is not None:
        base_url += f"#L{line}"
    return base_url


def finditer_with_line_numbers(
    pattern: str | Pattern[str],
    string: str,
    flags: int = 0,
    line_map: dict[int, int] | None = None,
) -> list[tuple[int, Match[str]]]:
    """Find matches of a pattern in a string, returning match objects with adjusted line numbers.

    This function accounts for `#line` directives and a line map for preprocessed content.

    Args:
        pattern: The regular expression pattern to match.
        string: The input string to search.
        flags: Regular expression flags. Defaults to 0.
        line_map: Mapping of preprocessed to original line numbers.

    Returns:
        list[tuple[int, Match[str]]]: A list of tuples containing the adjusted line number and match object.
    """
    # Handle pcpp info on skipped lines
    line_offsets: dict[int, int] = {}
    for line_number, line in enumerate(string.splitlines()):
        line_adjust = r"^#line (?P<line>[0-9]+) \"(?P<filename>.*)\""
        line_match = re.match(line_adjust, line)
        if line_match:
            offset = int(line_match.group("line")) - line_number - 1
            line_offsets[line_number] = offset

    matches = list(re.finditer(pattern, string, flags))
    if not matches:
        return []

    end = matches[-1].start()
    newline_table = {-1: 0}
    for i, m in enumerate(re.finditer("\\n", string), 1):
        offset = m.start()
        if offset > end:
            break
        newline_table[offset] = i

    result: list[tuple[int, Match[str]]] = []
    for m in matches:
        newline_offset = string.rfind("\n", 0, m.start())
        line_number = newline_table[newline_offset] + 1  # Add 1 since line numbers are 1-based
        # Apply line_map if provided
        if line_map and line_number in line_map:
            line_number = line_map[line_number]
        # Apply #line directive offset
        found_offset = 0
        for k, v in line_offsets.items():
            if k <= line_number:
                found_offset = v
            else:
                break
        adjusted_line = line_number + found_offset
        result.append((adjusted_line, m))
    return result


def capture_pattern(text: str, pattern: str) -> list[tuple[int, Match[str]]]:
    """Capture matches of a pattern in the given text, accounting for line directives.

    Args:
        text: The text to search.
        pattern: The regular expression pattern to match.

    Returns:
        list[tuple[int, Match[str]]]: A list of tuples containing the line number and match object.
    """
    # line adjust
    # line 288 "features/Grass Lighting/Shaders/RunGrass.hlsl"
    line_adjust = r"^#line (?P<line>[0-9]+) \"(?P<filename>.*)\""
    # Compile the regular expression pattern.
    regex = re.compile(pattern)
    results: list[tuple[int, Match[str]]] = []
    offset = 1
    # Iterate over the lines of the text.
    for line_number, line in enumerate(text.splitlines()):
        line_match = re.match(line_adjust, line)
        if line_match:
            offset = int(line_match.group("line")) - line_number - 1

        # Match the pattern against the line.
        match = regex.match(line)

        # If there is a match, return the line number.
        if match:
            results.append((line_number + offset, match))

    # If no match is found, return None.
    return results


def get_hlsl_types() -> dict[str, str]:
    """Return a mapping of HLSL register types to their descriptions.

    Returns:
        dict[str, str]: A dictionary mapping register types to their descriptions.
    """
    # https://learn.microsoft.com/en-us/windows/win32/direct3d12/resource-binding-in-hlsl
    return {"t": "SRV", "u": "UAV", "s": "Sampler", "b": "CBV"}


def get_defines_list() -> list[dict[str, str]]:
    """Return a list of preprocessor define combinations for HLSL compilation.

    Returns:
        list[dict[str, str]]: A list of dictionaries containing preprocessor defines.
    """
    return [
        {"PSHADER": ""},
        {"PSHADER": "", "VR": ""},
        {"VSHADER": ""},
        {"VSHADER": "", "VR": ""},
    ]


def get_excluded_dirs(cwd: str) -> set[str]:
    """Get directories to exclude based on .gitignore and default exclusions.

    Args:
        cwd: Current working directory.

    Returns:
        set[str]: Set of directory names to exclude.
    """
    default_exclusions = {"build", "extern", "tools", "include"}
    excluded_dirs = set(default_exclusions)

    gitignore_path = os.path.join(cwd, ".gitignore")
    if not os.path.isfile(gitignore_path) or pathspec is None:
        logging.warning(f".gitignore not found or pathspec not available, using default exclusions: {excluded_dirs}")
        return excluded_dirs

    try:
        with open(gitignore_path) as file:
            spec = pathspec.PathSpec.from_lines("gitwildmatch", file)
        for pattern in spec.patterns:
            # Extract directory names from patterns
            pattern_str = str(pattern)
            if pattern_str.endswith("/"):
                dir_name = pattern_str.rstrip("/").split("/")[-1]
                if dir_name:
                    excluded_dirs.add(dir_name)
    except Exception as e:
        logging.warning(f"Failed to parse .gitignore: {e}. Using default exclusions: {excluded_dirs}")

    return excluded_dirs


def preprocess_content(content: str, defines: dict[str, str]) -> str:
    """Preprocess HLSL content to include/exclude code based on defines.

    Args:
        content (str): The HLSL content to preprocess.
        defines (dict[str, str]): Preprocessor defines to apply.

    Returns:
        str: Preprocessed content with applied defines.
    """
    lines = content.splitlines()
    output = []
    include = True
    skip_depth = 0

    for line in lines:
        line = line.strip()
        if line.startswith("#ifdef") or line.startswith("#ifndef"):
            macro = line.split()[1]
            is_ifdef = line.startswith("#ifdef")
            should_include = (macro in defines) if is_ifdef else (macro not in defines)
            if not include:
                skip_depth += 1
            elif not should_include:
                include = False
                skip_depth += 1
        elif line.startswith("#else"):
            if skip_depth == 0:
                include = not include
        elif line.startswith("#endif"):
            if skip_depth > 0:
                skip_depth -= 1
            if skip_depth == 0:
                include = True
        elif include:
            output.append(line)

    return "\n".join(output)


def process_file(
    path: str,
    cwd: str,
    defines: dict[str, str],
    shader_pattern: Pattern[str],
    hlsl_types: dict[str, str],
    feature: str,
    short_path: str,
    result_map: dict[str, dict[str, Any]],
    compilation_units: dict[tuple[str, frozenset[str]], dict[str, set[str]]],
) -> None:
    """Process a shader file to extract buffers and update result maps."""

    def _should_skip_buffer(buffer_name, mapped_line, original_lines):
        if buffer_name.lower() in BASE_TYPE_SIZES:
            logging.debug(f"Skipping buffer {buffer_name} (built-in type) in {path}:{mapped_line}")
            return True

        if is_shader_io_struct(buffer_name):
            logging.debug(f"Skipping shader IO buffer: {buffer_name} in {path}:{mapped_line}")
            return True
        if mapped_line is None:
            logging.debug(f"Skipping buffer {buffer_name} from include at preprocessed line {mapped_line}")
            return True
        if mapped_line - 1 >= len(original_lines):
            return True
        context_window = 5
        start = max(0, mapped_line - context_window - 1)
        end = min(len(original_lines), mapped_line + context_window)
        buffer_decl_filter = re.compile(
            rf"""
(?:\s*
    (?:cbuffer|struct)\s+{re.escape(buffer_name)}\s*:\s*register\s*\([butsg]\d+\)
    |
    (?:[\w<>]+\s+)?{re.escape(buffer_name)}\s*:\s*register\s*\([butsg]\d+\)
)
""",
            re.IGNORECASE | re.VERBOSE,
        )
        for i in range(start, end):
            line = original_lines[i]
            if buffer_decl_filter.search(line):
                return False
        logging.debug(f"Skipping buffer {buffer_name} in {path}:{mapped_line}")
        return True

    path = os.path.normpath(path).replace("\\", "/")
    cwd = os.path.normpath(cwd).replace("\\", "/")
    if not path.startswith(cwd) or not os.path.isfile(path):
        logging.error(f"Skipping invalid or missing file: {path}")
        return

    try:
        with open(path, encoding="utf-8", errors="ignore") as file:
            original_contents = file.read()

        # Preprocess with pcpp, adding include paths
        preprocessor = pcpp.Preprocessor()
        include_dirs: list[str] = [
            os.path.join(cwd, "package", "Shaders"),
            os.path.dirname(path),
        ]

        common_dir = os.path.join(cwd, "package", "Shaders", "Common")
        if os.path.isdir(common_dir):
            for root, _, _ in os.walk(common_dir):
                include_dirs.append(root)

        features_dir = os.path.join(cwd, "features")
        if os.path.isdir(features_dir):
            for feature_dir in Path(features_dir).glob("*/Shaders"):
                include_dirs.append(str(feature_dir))

        aio_shaders_dir = os.path.join(cwd, "Shaders")
        if os.path.isdir(aio_shaders_dir) and "aio" in cwd.lower():
            include_dirs.append(aio_shaders_dir)
        include_dirs = list(set(include_dirs))  # remove duplicates

        for inc_dir in include_dirs:
            preprocessor.add_path(os.path.normpath(inc_dir))

        for key, value in defines.items():
            preprocessor.define(f"{key} {value or '1'}")

        preprocessor.parse(original_contents, path)
        with io.StringIO() as io_buffer:
            preprocessor.write(io_buffer)
            contents = io_buffer.getvalue()

        # Build line map: preprocessed line -> original line
        line_map: dict[int, int] = {}
        preprocessed_line = 0
        current_file = path
        current_line = 0
        original_lines = original_contents.splitlines()
        orig_line_idx = 0
        for line in contents.splitlines():
            line_match = re.match(r'^#line\s+(\d+)\s+"([^"]*)"', line)
            if line_match:
                current_line = int(line_match.group(1))
                current_file = line_match.group(2).replace("\\", "/")
                orig_line_idx = min(current_line - 1, len(original_lines) - 1)
                continue
            if current_file == path:
                line_map[preprocessed_line] = current_line
                preprocessed_line += 1
            if current_file == path and orig_line_idx < len(original_lines):
                if line.strip() == original_lines[orig_line_idx].strip():
                    current_line = orig_line_idx + 1
                orig_line_idx += 1
            current_line += 1

        if not line_map:
            logging.debug(f"Empty line map for {path}, assuming identity mapping")
            contents = original_contents
            line_map = {i + 1: i + 1 for i in range(len(contents.splitlines()))}
        logging.debug(f"Preprocessing {path} with defines: {defines}")

        # Process buffers
        capture_list: list[tuple[int, Match[str]]] = finditer_with_line_numbers(
            shader_pattern, contents, line_map=line_map
        )
        for line_number, result in capture_list:
            mapped_line = line_map.get(line_number)
            buffer_name = result.group("name")
            if _should_skip_buffer(buffer_name, mapped_line, original_lines):
                continue
            path_with_line_no = f"{short_path}:{mapped_line}"
            key = f"{path_with_line_no}"  # Consistent key format
            entry = result_map.get(key)
            if not entry:
                # Extract template information from the full type match
                full_type = result.group("type")
                template_type = ""

                # Check for ConstantBuffer<Type> pattern
                const_buffer_match = re.search(r"ConstantBuffer<(\w+)>", full_type)
                if const_buffer_match:
                    template_type = const_buffer_match.group(1)
                # Check for other template patterns like Texture2D<half2>
                elif result.group("template_name"):
                    template_type = result.group("template_name")

                # Create a key for this define combination
                define_key = "_".join(sorted(defines.keys())) if defines else "no_defines"

                entry = {
                    "Register": f"{result.group('buffer_type').lower()}{result.group('buffer_number')}",
                    "Feature": feature,
                    "Type": f"`{full_type}`",
                    "Name": buffer_name,  # This should be the actual buffer variable name
                    "File": f"[{path_with_line_no}]({create_link(short_path, mapped_line)})",
                    "File Path": short_path,  # Store the file path separately
                    "Register Type": hlsl_types.get(result.group("buffer_type").lower(), "Unknown"),
                    "Buffer Type": result.group("buffer_type"),
                    "Number": int(result.group("buffer_number")),
                    "PSHADER": False,
                    "VSHADER": False,
                    "VR": False,
                    "Matching Struct Analysis": "",
                    "Original Line": mapped_line,  # Store original line number,
                    "Template Type": template_type,
                    "Define Combinations": set(),  # Track define combinations
                }
                result_map[key] = entry
            # Record this define combination with consistent ordering
            if defines:
                # Create consistent define key with standard ordering
                define_parts = []
                if "PSHADER" in defines:
                    define_parts.append("PSHADER")
                if "VSHADER" in defines:
                    define_parts.append("VSHADER")
                if "VR" in defines:
                    define_parts.append("VR")
                define_key = "_".join(define_parts) if define_parts else "no_defines"
            else:
                define_key = "no_defines"

            if "Define Combinations" not in entry:
                entry["Define Combinations"] = set()
            entry["Define Combinations"].add(define_key)

            # Update boolean flags
            for define in defines:
                entry[define] = True
            if "PSHADER" in defines:
                entry["PSHADER"] = True
            if "VSHADER" in defines:
                entry["VSHADER"] = True
            if "VR" in defines:
                entry["VR"] = True

            compilation_unit_key = (short_path.lower(), frozenset(defines.keys()))
            if compilation_unit_key not in compilation_units:
                compilation_units[compilation_unit_key] = {}
            reg = f"{result.group('buffer_type').lower()}{result.group('buffer_number')}"
            if reg not in compilation_units[compilation_unit_key]:
                compilation_units[compilation_unit_key][reg] = set()
            compilation_units[compilation_unit_key][reg].add(feature)

    except Exception:
        logging.exception("Failed to process file %s", path)


def scan_files(
    cwd: str,
    pattern: Pattern[str],
    feature_pattern: Pattern[str],
    shader_pattern: Pattern[str],
    hlsl_types: dict[str, str],
    defines_list: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], CompilationUnits]:
    """Scan HLSL files for buffer definitions and track compilation units.

    Args:
        cwd: The current working directory.
        pattern: Compiled regex pattern to match HLSL files.
        feature_pattern: Compiled regex pattern to extract feature names.
        shader_pattern: Compiled regex pattern for buffer definitions.
        hlsl_types: Mapping of HLSL register types to descriptions.
        defines_list: List of preprocessor define dictionaries.

    Returns:
        tuple[list[dict[str, Any]], CompilationUnits]: A tuple containing the list of buffer entries and compilation units.
    """
    result_map = {}
    compilation_units = {}
    excluded_dirs = get_excluded_dirs(cwd)
    for root, dirs, files in os.walk(cwd):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        feature = ""
        root_normalized = root.replace("\\", "/")
        feature_match = feature_pattern.search(root_normalized)
        if feature_match:
            feature = feature_match.group("feature")
        for file in files:
            if file.lower().endswith((".hlsl", ".hlsli")):
                full_path = os.path.join(root, file).replace("\\", "/")
                short_path_start = full_path.lower().find("skyrim-community-shaders")
                short_path = (
                    full_path[short_path_start + len("skyrim-community-shaders") + 1 :]
                    if short_path_start != -1
                    else os.path.relpath(full_path, cwd)
                )
                logging.debug(f"Processing file: {full_path}")
                for defines in defines_list:
                    process_file(
                        full_path,
                        cwd,
                        defines,
                        shader_pattern,
                        hlsl_types,
                        feature,
                        short_path,
                        result_map,
                        compilation_units,
                    )
    results = list(result_map.values())
    logging.debug(f"Scan found {len(results)} buffers")
    return results, compilation_units


def print_buffers_and_conflicts(
    result_map: dict[str, dict[str, Any]],
    compilation_units: dict[tuple[str, frozenset[str]], dict[str, set[str]]],
    show_conflicts: bool = False,
) -> None:
    """Print the buffers and any conflicts as a markdown table and report.

    Args:
        result_map: Dictionary of unique buffer entries.
        compilation_units: Dictionary of register usage per compilation unit.
        show_conflicts: Whether to print conflict analysis sections.
    """
    if not result_map:
        print("No results found.")
        return

    sorted_results = sorted(
        result_map.values(),
        key=lambda x: (str(x.get("Register", "")), str(x.get("Name", ""))),
    )
    rows: list[dict[str, str]] = [
        {
            "Register": str(entry.get("Register", "")),
            "Feature": str(entry.get("Feature", "")),
            "Type": str(entry.get("Type", "")),
            "Name": str(entry.get("Name", "")),
            "File": str(entry.get("File", "")),
            "Register Type": str(entry.get("Register Type", "")),
            "Buffer Type": str(entry.get("Buffer Type", "")),
            "Number": str(entry.get("Number", "")),
            "PSHADER": str(entry.get("PSHADER", False)),
            "VSHADER": str(entry.get("VSHADER", False)),
            "VR": str(entry.get("VR", False)),
            "Struct Analysis": str(entry.get("Matching Struct Analysis", "")),
        }
        for entry in sorted_results
    ]

    # Filter out entries with blank Type, Name, or File
    filtered_results = [
        entry
        for entry in rows
        if entry.get("Type", "").strip() and entry.get("Name", "").strip() and entry.get("File", "").strip()
    ]

    print("<!--")
    print("DEBUG INFORMATION (hidden):")
    # Print any debug information that was collected during analysis
    for debug_msg in DEBUG_INFO:
        print(debug_msg)
    print("-->")
    print()
    print(f"# Buffer Table (generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    table = markdown_table(filtered_results).set_params(quote=False, row_sep="markdown").get_markdown()
    print(table)

    if show_conflicts:
        conflicts: dict[str, list[dict[str, Any]]] = {}
        for (path, defines), registers in compilation_units.items():
            for reg, features in registers.items():
                if len(features) > 1:
                    if reg not in conflicts:
                        conflicts[reg] = []
                    conflicts[reg].append({"path": path, "defines": defines, "features": features})

        if conflicts:
            print("\n# Conflicts")
            for reg, conflict_list in sorted(conflicts.items()):
                print(f"\n## {reg}")
                for conflict in conflict_list:
                    print(f"- **Path**: {conflict['path']}")
                    print(f"  **Defines**: {', '.join(sorted(str(d) for d in conflict['defines']))}")
                    print(f"  **Features**: {', '.join(sorted(str(f) for f in conflict['features']))}")

        # Additional analysis: Check for register conflicts within same define context
        register_conflicts = {}
        for entry in sorted_results:
            reg = entry.get("Register", "")
            if not reg:
                continue

            # Group by register and define combination
            define_combos = entry.get("Define Combinations", set())
            for combo in define_combos:
                key = f"{reg}_{combo}"
                if key not in register_conflicts:
                    register_conflicts[key] = []
                register_conflicts[key].append({
                    "name": entry.get("Name", ""),
                    "file": entry.get("File Path", ""),
                    "feature": entry.get("Feature", ""),
                    "type": entry.get("Type", ""),
                })

        # Report register conflicts
        true_conflicts = {k: v for k, v in register_conflicts.items() if len(v) > 1}
        if true_conflicts:
            print("\n# Register Conflicts (Same Register + Define Combination)")
            for key, conflicts in sorted(true_conflicts.items()):
                reg, combo = key.rsplit("_", 1)
                print(f"\n## Register {reg} with defines: {combo.replace('_', ', ')}")
                for conflict in conflicts:
                    print(
                        f"- **{conflict['name']}** ({conflict['type']}) in `{conflict['file']}` - Feature: {conflict['feature']}"
                    )
        else:
            print("\n# Register Conflicts")
            print(
                "No register conflicts detected - all buffers use unique register slots within their define contexts."
            )


def calculate_struct_size(fields: list[FieldDict], align_to_16: bool = False) -> int:
    """Calculate total size of a struct, accounting for alignment.

    Args:
        fields: List of field dictionaries.
        align_to_16: Whether to align to 16-byte boundaries (e.g., for cbuffers).

    Returns:
        int: Total size in bytes.
    """
    total_size = 0
    for field in fields:
        size = field["size"]
        if align_to_16:
            # Align each field to 16-byte boundary for cbuffers
            total_size = (total_size + (ALIGN_TO_16 - 1)) & ~(ALIGN_TO_16 - 1)
        total_size += size
    if align_to_16:
        total_size = (total_size + (ALIGN_TO_16 - 1)) & ~(ALIGN_TO_16 - 1)
    # logging.debug(
    #     f"Calculated struct size: {total_size} bytes for fields: {[(f['name'], f['type'], f['size']) for f in fields]}"
    # )
    return total_size


def calculate_hlsl_struct_size(fields: list[FieldDict]) -> int:
    """Calculate HLSL struct size, handling packoffset and 16-byte alignment.

    Args:
        fields: List of field dictionaries with type, size, and optional packoffset.

    Returns:
        int: Total size in bytes, aligned to 16-byte boundaries.
    """
    offset = 0
    max_offset = 0
    for field in fields:
        packoffset = field.get("packoffset")
        field_size, _ = get_field_size(field["type"], field.get("array_size", 1))
        if packoffset:
            match = re.match(r"c(\d+)\.([xyzw])", packoffset)
            if match:
                register = int(match.group(1))
                component = {"x": 0, "y": 4, "z": 8, "w": 12}[match.group(2)]
                field_offset = register * ALIGN_TO_16 + component
                offset = max(offset, field_offset)
            else:
                raise ValueError
        else:
            offset = (offset + (ALIGN_TO_4 - 1)) & ~(ALIGN_TO_4 - 1)  # Align to 4-byte boundary
        offset += field_size
        max_offset = max(max_offset, offset)
    return (max_offset + (ALIGN_TO_16 - 1)) & ~(ALIGN_TO_16 - 1)  # Align to 16-byte boundary


def clean_body(body: str) -> str:
    """Clean comments and empty lines from struct body."""
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    lines = body.splitlines()
    cleaned_lines = [re.sub(r"//.*$", "", line).strip() for line in lines if re.sub(r"//.*$", "", line).strip()]
    return "\n".join(cleaned_lines)


def parse_field(field: str, struct_name: str, is_hlsl: bool) -> FieldDict | None:
    field = field.strip()
    if not field:
        return None
    if is_hlsl:
        field_match = re.match(
            r"(?:(?:row|column)_major\s+)?(?P<type>[\w:]+(?:<\w+>)?(?:\w+)?)\s+(?P<name>\w+)(?:\s*:\s*packoffset\((?P<packoffset>[^)]+)\))?(?:\s*\[(?P<array>\d+(?:\]\[\d+)?)\])?",
            field,
        )
    else:
        field_match = re.match(
            r"(?P<type>[\w:]+(?:\w+)?(?:<\w+>)?\s*\*?)\s+(?P<name>\w+)(?:\s*\[(?P<array>\d+(?:\]\[\d+)?)\])?",
            field,
        )
    if not field_match:
        logging.debug(f"Failed to parse {'HLSL' if is_hlsl else 'C++'} field in {struct_name}: {field}")
        return None
    field_type = field_match.group("type")
    field_name = field_match.group("name")
    array_size = 1
    if field_match.group("array"):
        dims = [int(dim) for dim in re.findall(r"\d+", field_match.group("array"))]
        array_size = 1
        for dim in dims:
            array_size *= dim
    display_name = f"{field_name}[{array_size}]" if array_size > 1 else field_name
    field_size, is_unknown = get_field_size(field_type, array_size)
    result: FieldDict = {
        "name": display_name,
        "type": field_type,
        "size": field_size,
        "array_size": array_size,
        "is_unknown_type": is_unknown,
    }
    if is_hlsl and (packoffset := field_match.group("packoffset")):
        result["packoffset"] = packoffset
    return result


def is_shader_io_struct(name: str) -> bool:
    name_upper = name.upper()
    return name_upper.endswith("_INPUT") or name_upper.endswith("_OUTPUT")


def extract_hlsl_structs(content: str, file_path: str) -> dict[str, dict]:
    """Extract HLSL structs, cbuffers, ConstantBuffers, and template buffers."""
    structs = {}
    struct_pattern = r"(struct|cbuffer|ConstantBuffer<(?P<template>\w+)>)\s+(?P<name>\w+)\s*(?::\s*register\s*\(\w\d+\s*\))?\s*{(?P<body>[^{}]*?)}"
    template_pattern = (
        r"(?:RW)?(?:StructuredBuffer)<(?P<template>\w+)>\s+(?P<name>\w+)\s*:\s*register\s*\([a-z]\d+\s*\)"
    )
    content_clean = preprocess_content(content, {})
    content_clean = clean_body(content_clean)

    # Extract structs, cbuffers, and ConstantBuffers
    for line_number, match in finditer_with_line_numbers(struct_pattern, content, re.MULTILINE | re.DOTALL):
        name = match.group("name")
        if name.lower() in BASE_TYPE_SIZES:
            logging.debug(f"Skipping struct/buffer {name} (built-in type) in {file_path}:{line_number}")
            continue
        body = clean_body(match.group("body").strip())
        is_cbuffer = match.group(1) == "cbuffer"
        is_constant_buffer = match.group(1).startswith("ConstantBuffer")
        fields = [f for field in body.split(";") if (f := parse_field(field, name, True))]
        size = calculate_struct_size(fields, align_to_16=is_cbuffer or is_constant_buffer)
        structs[name] = {
            "fields": fields,
            "file": file_path,
            "line": line_number,
            "is_cbuffer": is_cbuffer,
            "is_constant_buffer": is_constant_buffer,
            "is_template": False,
            "size": size,
        }
        logging.debug(
            f"Found HLSL {'cbuffer' if is_cbuffer else 'ConstantBuffer' if is_constant_buffer else 'struct'} {name} in {file_path}:{line_number} with {len(fields)} fields, total size: {size} bytes"
        )

    # Second pass: extract and process template buffers
    for line_number, match in finditer_with_line_numbers(template_pattern, content, re.MULTILINE):
        template_type = match.group("template")
        template_name = match.group("name")

        # Skip if template type is a base type
        if template_type in BASE_TYPE_SIZES:
            logging.debug(f"Skipping base type template {template_type} in {file_path}:{line_number}")
            continue

        # Only add template type if we don't already have a real definition
        if template_type not in structs:
            structs[template_type] = {
                "fields": [],
                "file": file_path,
                "line": line_number,
                "is_cbuffer": False,
                "is_template": True,
            }
            logging.debug(f"Found template struct {template_type} in {file_path}:{line_number}")
        else:
            # Check if existing definition has actual fields (real struct vs empty template)
            existing_struct = structs[template_type]
            if existing_struct.get("fields") and not existing_struct.get("is_template", False):
                logging.debug(
                    f"Skipping template {template_type} - real struct definition already exists with {len(existing_struct['fields'])} fields"
                )
                continue
            else:
                logging.debug(f"Template {template_type} found but existing definition is also a template or empty")

        # Also add the instance as its own entry (only if not already exists)
        if template_name not in structs:
            structs[template_name] = {
                "fields": [],
                "file": file_path,
                "line": line_number,
                "is_cbuffer": False,
                "is_template": True,
                "template_type": template_type,
            }

    return structs


def extract_cpp_structs(content: str, file_path: str) -> dict[str, dict]:
    """Extract C++ structs, skipping any that contain pointers or only static members.

    Args:
        content (str): The C++ file content.
        file_path (str): Path to the file for logging and linking.

    Returns:
        dict[str, dict]: Dictionary of struct names to their metadata.
    """
    structs = {}
    struct_pattern = r"struct\s+(?:alignas\(\d+\)\s+)?(?P<name>\w+)\s*{(?P<body>[^{}]*?)}"

    for line_number, match in finditer_with_line_numbers(struct_pattern, content, re.MULTILINE | re.DOTALL):
        name = match.group("name")
        if name.lower() in BASE_TYPE_SIZES:
            logging.debug(f"Skipping struct/buffer {name} (built-in type) in {file_path}:{line_number}")
            continue

        body = clean_body(match.group("body").strip())

        # Parse all fields, including static ones
        all_fields = []
        non_static_fields = []

        for field_line in body.split(";"):
            field_line = field_line.strip()
            if not field_line:
                continue

            # Check if this is a static declaration
            is_static = field_line.startswith("static ")

            if is_static:
                logging.debug(f"Skipping static member in {name}: {field_line}")
                continue

            # Parse non-static field
            parsed_field = parse_field(field_line, name, False)
            if parsed_field:
                all_fields.append(parsed_field)
                non_static_fields.append(parsed_field)

        # Skip struct if it has no non-static fields
        if not non_static_fields:
            logging.debug(f"Skipping C++ struct {name} in {file_path}:{line_number + 1} (no non-static fields)")
            continue

        # Skip struct if any non-static field contains a pointer
        if any("*" in field["type"] for field in non_static_fields):
            logging.debug(f"Skipping C++ struct {name} in {file_path}:{line_number + 1} (contains pointers)")
            continue

        adjusted_line = line_number + 1
        structs[name] = {
            "fields": non_static_fields,  # Only store non-static fields
            "file": file_path,
            "line": adjusted_line,
            "is_cbuffer": False,
            "is_template": False,
            "body": body,  # Store raw body for alignment check
        }
        size = calculate_struct_size(non_static_fields)
        logging.debug(
            f"Found C++ struct {name} in {file_path}:{adjusted_line} with {len(non_static_fields)} non-static fields: {[(f['name'], f['type'], f['size']) for f in non_static_fields]}, total size: {size} bytes"
        )

    return structs


def extract_structs(content: str, is_hlsl: bool, file_path: str) -> dict[str, dict]:
    """Extract structs from HLSL or C++ content.

    Args:
        content (str): The file content to parse.
        is_hlsl (bool): Whether the content is HLSL (True) or C++ (False).
        file_path (str): Path to the file for logging and linking.

    Returns:
        dict[str, dict]: Dictionary of struct names to their metadata.
    """
    logging.debug(f"Extracting structs from {file_path}")
    return extract_hlsl_structs(content, file_path) if is_hlsl else extract_cpp_structs(content, file_path)


def get_struct_signature(fields: list[dict]) -> str:
    """Generate a signature for a struct based on field names, types, and sizes.

    Args:
        fields (list[dict]): List of field dictionaries.

    Returns:
        str: Struct signature string.
    """
    signature = []
    for field in fields:
        norm_type = normalize_field_type(field["type"])
        size = field["size"]
        signature.append(f"{norm_type}:{field['name']}:{size}")
    sig_str = ";".join(signature)
    return sig_str


BASE_TYPE_SIZES = {
    "float": 4,
    "float2": 8,
    "float3": 12,
    "float4": 16,
    "uint": 4,
    "uint2": 8,
    "uint3": 12,
    "uint4": 16,
    "int": 4,
    "int4": 16,
    "bool": 4,
}


def parse_type_with_array(field_type: str) -> tuple[str, int]:
    """
    Parse field type and extract array size if present.

    Args:
        field_type (str): The type name, e.g., 'float4[4]'.

    Returns:
        tuple[str, int]: A tuple containing the base type and array size (default 1).
    """
    match = re.match(r"(.+?)\[(\d+)\]$", field_type)
    if match:
        return match.group(1), int(match.group(2))
    return field_type, 1


def extract_matrix_size(field_type: str) -> tuple[int, int] | None:
    """
    Extract matrix dimensions from names like XMFLOAT3X4.

    Args:
        field_type (str): The field type, potentially namespaced (e.g., REX::XMFLOAT3X4).

    Returns:
        tuple[int, int] | None: (rows, columns) if matched, else None.
    """
    match = re.search(r"FLOAT(\d)X(\d)", field_type.upper())
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def normalize_field_type(field_type: str) -> str:
    """
    Normalize a field type string to a canonical HLSL-like form (e.g., float4x4).

    Args:
        field_type (str): Original field type (e.g., 'REX::W32::XMFLOAT4X4', 'float3').

    Returns:
        str: Normalized type, e.g., 'float4x4' or 'float3'.
    """
    base_type, _ = parse_type_with_array(field_type)

    # Handle XMFLOAT matrix types
    if dims := extract_matrix_size(base_type):
        if "XM" in base_type.upper():
            return f"xmfloat{dims[0]}x{dims[1]}"
        return f"float{dims[0]}x{dims[1]}"

    # Strip namespaces and return last component
    return base_type.split("::")[-1].lower()


def get_field_size(field_type: str, array_size: int = 1) -> tuple[int, bool]:
    """Calculate the size in bytes of a field based on its type and array size.

    Args:
        field_type: The type name (e.g., 'float4', 'XMFLOAT4X4[3]', or pointer types).
        array_size: Optional additional array multiplier (default: 1).

    Returns:
        A tuple containing:
        - The size in bytes
        - A flag indicating if the type is unknown (True) or recognized (False)
    """
    base_type, parsed_array_size = parse_type_with_array(field_type)
    array_size *= parsed_array_size

    if base_type.endswith("*"):
        return POINTER_SIZE * array_size, False  # Assume 64-bit pointer

    norm_type = normalize_field_type(base_type)

    # Handle floatNxM matrix
    if match := re.match(r"float(\d)x(\d)", norm_type):
        rows, cols = int(match.group(1)), int(match.group(2))
        return 4 * rows * cols * array_size, False  # 4 bytes per float

    size = BASE_TYPE_SIZES.get(norm_type, DEFAULT_TYPE_SIZE)
    is_unknown = size == DEFAULT_TYPE_SIZE and norm_type not in BASE_TYPE_SIZES
    return size * array_size, is_unknown


def is_padding_field(field: dict) -> bool:
    """Check if a field is a padding field.

    Args:
        field: Field dictionary containing name and type.

    Returns:
        bool: True if the field is a padding field.
    """
    name = field["name"].lower()
    # Remove array size from name for comparison
    base_name = name.split("[")[0] if "[" in name else name
    return (
        base_name == "pad"
        or base_name.startswith("_pad")
        or base_name.startswith("pad")
        or base_name.endswith("pad")
        or base_name.startswith("padding")
        or base_name.endswith("padding")
        or base_name.startswith("_padding")
        or base_name.endswith("_padding")
    )


def are_fields_equivalent(cpp_field: dict, hlsl_field: dict) -> bool:
    """Check if fields are equivalent despite different types.

    This function uses the following criteria to determine equivalence:
    1. Name similarity (using compute_name_similarity)
    2. Type compatibility (using normalize_field_type)
    3. Size matching

    Args:
        cpp_field: C++ field dictionary with name, type and size.
        hlsl_field: HLSL field dictionary with name, type and size.

    Returns:
        bool: True if fields are equivalent, False otherwise.
    """
    # Get field similarities using our helper function
    type_sim, name_sim, size_match = get_field_similarity(cpp_field, hlsl_field)

    # Fields are equivalent if they have both matching names and types
    return name_sim >= NAME_SIM_THRESHOLD and type_sim == 1.0 and size_match


def normalize_array_types(hlsl_type: str, cpp_type: str) -> tuple[str, str]:
    """Normalize HLSL vector types and C++ array types for comparison."""
    # HLSL vector to C++ array mapping
    hlsl_to_cpp = {
        "float2": "float[2]",
        "float3": "float[3]",
        "float4": "float[4]",
        "int2": "int[2]",
        "int3": "int[3]",
        "int4": "int[4]",
        "uint2": "uint[2]",
        "uint3": "uint[3]",
        "uint4": "uint[4]",
    }

    # Normalize HLSL type to equivalent C++ array
    normalized_hlsl = hlsl_to_cpp.get(hlsl_type, hlsl_type)

    # Extract base type and size from C++ array notation
    cpp_array_match = re.match(r"(\w+)\[(\d+)\]", cpp_type)
    if cpp_array_match:
        base_type, size = cpp_array_match.groups()
        normalized_cpp = f"{base_type}[{size}]"
    else:
        normalized_cpp = cpp_type

    return normalized_hlsl, normalized_cpp


def get_field_similarity(cpp_field: dict, hlsl_field: dict) -> tuple[float, float, bool]:
    """Enhanced field similarity that handles array type equivalence.

    Args:
        cpp_field: C++ field metadata containing name, type and size.
        hlsl_field: HLSL field metadata containing name, type and size.

    Returns:
        tuple[float, float, bool]: A tuple containing:
            - type_sim: Type similarity (0.0 to 1.0)
            - name_sim: Name similarity score (0.0 to 1.0)
            - size_match: Whether field sizes match exactly
    """
    # Use compute_name_similarity which handles array notation internally
    name_sim = compute_name_similarity(hlsl_field["name"], cpp_field["name"])

    # Normalize types for comparison
    hlsl_norm, cpp_norm = normalize_array_types(hlsl_field["type"], cpp_field["type"])

    # Check for exact type match after normalization
    if hlsl_norm == cpp_norm:
        type_sim = 1.0
    else:
        # Fall back to string similarity for types that don't normalize
        type_sim = jellyfish.jaro_winkler_similarity(hlsl_field["type"], cpp_field["type"])
        if type_sim < 0.7:
            type_sim = 0.0

    # Direct size comparison
    size_match = cpp_field["size"] == hlsl_field["size"]

    return type_sim, name_sim, size_match


def compute_struct_alignment(
    cpp_data: dict[str, Any], hlsl_data: dict[str, Any], struct_name_weight: float = 0.5
) -> tuple[float, list[tuple[FieldDict | None, FieldDict | None]], dict[str, Any]]:
    """Compute alignment between HLSL and C++ struct fields.

    This function analyzes field alignment between C++ and HLSL structs, considering:
    - Field name similarity using compute_name_similarity
    - Field type compatibility using normalize_field_type
    - Field size matching using calculate_struct_size
    - Struct name similarity using compute_name_similarity

    Args:
        cpp_data: Dictionary containing C++ struct metadata and fields.
        hlsl_data: Dictionary containing HLSL struct metadata and fields.
        struct_name_weight: Weight given to struct name similarity (0.0 to 1.0).

    Returns:
        A tuple containing:
        - score: Overall alignment score (0.0 to 1.0)
        - align_matches: List of field alignment tuples
        - report: Detailed alignment report dictionary
    """
    if not isinstance(cpp_data, dict) or not isinstance(hlsl_data, dict):
        raise InvalidStructDictType(cpp_data if not isinstance(cpp_data, dict) else hlsl_data)

    cpp_fields = cpp_data.get("fields", [])
    hlsl_fields = hlsl_data.get("fields", [])

    # Use existing helper functions for field matching
    align_matches, report = _compute_alignment_report(cpp_fields, hlsl_fields, cpp_data, hlsl_data, struct_name_weight)

    score = report.get("score", 0.0)
    return score, align_matches, report


def compute_match_score(
    hlsl_name: str,
    cpp_name: str,
    hlsl_fields: list[FieldDict],
    cpp_fields: list[FieldDict],
    lcs_pairs: Optional[list[tuple[int, int]]] = None,
    struct_name_weight: float = 0.5,
) -> float:
    """Compute overall match score using LCS information.

    Args:
        hlsl_name: Name of HLSL struct
        cpp_name: Name of C++ struct
        hlsl_fields: List of HLSL fields
        cpp_fields: List of C++ fields
        lcs_pairs: List of matched field index pairs from LCS
        struct_name_weight: Weight for struct name similarity

    Returns:
        float: Match score between 0 and 1
    """
    # Name similarity
    name_sim = compute_name_similarity(hlsl_name, cpp_name)

    # Field matching score based on LCS
    lcs_length = len(lcs_pairs) if lcs_pairs else 0
    max_fields = max(len(hlsl_fields), len(cpp_fields))
    field_match_score = lcs_length / max_fields if max_fields > 0 else 0.0

    # Field order preservation score
    order_score = 1.0
    if lcs_pairs:
        prev_hlsl, prev_cpp = lcs_pairs[0]
        for hlsl_idx, cpp_idx in lcs_pairs[1:]:
            if hlsl_idx < prev_hlsl or cpp_idx < prev_cpp:
                order_score *= 0.9  # Penalize out-of-order matches
            prev_hlsl, prev_cpp = hlsl_idx, cpp_idx

    # Combine scores
    return name_sim * struct_name_weight + field_match_score * (1 - struct_name_weight) * order_score


def _compute_alignment_report(
    cpp_fields: list[FieldDict],
    hlsl_fields: list[FieldDict],
    cpp_data: dict[str, Any],
    hlsl_data: dict[str, Any],
    struct_name_weight: float = 0.5,
) -> tuple[list[tuple[FieldDict | None, FieldDict | None]], dict[str, Any]]:
    """Core logic for struct alignment, always returns a full report with all statistics."""
    align_matches: list[tuple[FieldDict | None, FieldDict | None]] = []
    report: dict[str, Any] = {
        "name_sim": 0.0,
        "type_sim": 0.0,
        "size_sim": 0.0,
        "exact_matches": 0,
        "high_sim_matches": 0,
        "total_fields": 0,
        "missing_fields": 0,
    }
    # Use fuzzy LCS to align fields by name similarity
    lcs_pairs = fuzzy_lcs(hlsl_fields, cpp_fields, name_sim_threshold=NAME_SIM_THRESHOLD)
    # logging.debug(f"LCS pairs found: {lcs_pairs}")
    # for i, j in lcs_pairs:
    #     hlsl_name = hlsl_fields[i]["name"] if i < len(hlsl_fields) else "N/A"
    #     cpp_name = cpp_fields[j]["name"] if j < len(cpp_fields) else "N/A"
    #     sim = compute_name_similarity(hlsl_name, cpp_name)
    #     logging.debug(f"  Pair ({i},{j}): {hlsl_name} <-> {cpp_name} (sim: {sim:.3f})")

    hlsl_matched = {i for i, _ in lcs_pairs}
    cpp_matched = {j for _, j in lcs_pairs}
    i, j = 0, 0
    lcs_idx = 0
    while i < len(hlsl_fields) or j < len(cpp_fields):
        if lcs_idx < len(lcs_pairs):
            lcs_i, lcs_j = lcs_pairs[lcs_idx]
        else:
            lcs_i, lcs_j = None, None
        if i < len(hlsl_fields) and (lcs_idx < len(lcs_pairs) and i == lcs_i and j == lcs_j):
            # Matched pair
            align_matches.append((hlsl_fields[i], cpp_fields[j]))
            type_sim, name_sim, _ = get_field_similarity(cpp_fields[j], hlsl_fields[i])
            if name_sim == 1.0 and type_sim == 1.0:
                report["exact_matches"] += 1
            elif name_sim >= 0.9 and type_sim >= 0.9:
                report["high_sim_matches"] += 1
            i += 1
            j += 1
            lcs_idx += 1
        elif i < len(hlsl_fields) and i not in hlsl_matched:
            align_matches.append((hlsl_fields[i], None))
            i += 1
        elif j < len(cpp_fields) and j not in cpp_matched:
            align_matches.append((None, cpp_fields[j]))
            j += 1
        else:
            i += 1
            j += 1
    report["missing_fields"] = len([
        f for f in hlsl_fields if not is_padding_field(f) and f not in [m[0] for m in align_matches if m[0]]
    ])
    report["total_fields"] = len([f for f in hlsl_fields if not is_padding_field(f)])
    cpp_total_size = calculate_struct_size(cpp_fields)
    hlsl_total_size = calculate_struct_size(hlsl_fields)
    size_ratio = (
        min(cpp_total_size, hlsl_total_size) / max(cpp_total_size, hlsl_total_size)
        if cpp_total_size > 0 and hlsl_total_size > 0
        else 0.0
    )
    # Calculate struct name similarity using our helper function
    cpp_name = cpp_data.get("name", "")
    hlsl_name = hlsl_data.get("name", "")
    name_sim = compute_name_similarity(str(cpp_name), str(hlsl_name))
    report["name_sim"] = name_sim * struct_name_weight

    # Calculate overall type similarity across fields
    total_fields = max(1, report["total_fields"])
    report["type_sim"] = (report["exact_matches"] + report["high_sim_matches"] * 0.8) / total_fields
    report["size_sim"] = size_ratio

    # Calculate field similarity scores using LCS information
    lcs_score = compute_match_score(
        hlsl_data.get("name", ""),
        cpp_data.get("name", ""),
        hlsl_fields,
        cpp_fields,
        lcs_pairs,
        struct_name_weight,
    )

    # Score is always computed, no thresholding here
    report["score"] = (
        lcs_score * 0.6  # Weight LCS matching more heavily
        + report["type_sim"] * 0.2
        + report["size_sim"] * 0.2
    )
    report["cpp_total_fields"] = len([f for f in cpp_fields if not is_padding_field(f)])
    total_diff, name_diff, type_diff = count_field_differences(align_matches)
    report["field_diff_count"] = total_diff
    report["field_name_diff_count"] = name_diff
    report["field_type_diff_count"] = type_diff
    report["field_matches"] = len([m for m in align_matches if m[0] and m[1]])
    report["unmatched_hlsl_fields"] = [f["name"] for f, c in align_matches if f and not c]
    report["unmatched_cpp_fields"] = [c["name"] for f, c in align_matches if c and not f]
    report["size_difference"] = abs(cpp_total_size - hlsl_total_size)
    report["field_names_hlsl"] = [f["name"] for f in hlsl_fields]
    report["field_names_cpp"] = [f["name"] for f in cpp_fields]
    return align_matches, report


# For compatibility, keep the old name as an alias
def align_structs(
    cpp_data: dict[str, Any], hlsl_data: dict[str, Any], struct_name_weight: float = 0.5
) -> tuple[float, list[tuple[FieldDict | None, FieldDict | None]], dict[str, Any]]:
    return compute_struct_alignment(cpp_data, hlsl_data, struct_name_weight)


def fuzzy_lcs(
    hlsl_fields: list[dict],
    cpp_fields: list[dict],
    name_sim_threshold: float = NAME_SIM_THRESHOLD,
) -> list[tuple[int, int]]:
    """Fuzzy sequence matching using difflib for field alignment.

    Args:
        hlsl_fields: List of HLSL field dictionaries
        cpp_fields: List of C++ field dictionaries
        name_sim_threshold: Minimum similarity threshold

    Returns:
        list[tuple[int, int]]: List of matching index pairs
    """
    matches = []
    used_cpp_indices = set()

    # For each HLSL field, find its best C++ match
    for i, hlsl_field in enumerate(hlsl_fields):
        best_match_idx = -1
        best_similarity = 0.0

        for j, cpp_field in enumerate(cpp_fields):
            if j in used_cpp_indices:
                continue

            sim = compute_name_similarity(hlsl_field["name"], cpp_field["name"])

            if sim >= name_sim_threshold and sim > best_similarity:
                best_similarity = sim
                best_match_idx = j

        # Add the best match if found
        if best_match_idx >= 0:
            matches.append((i, best_match_idx))
            used_cpp_indices.add(best_match_idx)
            # logging.debug(
            #     f"  Match found ({i},{best_match_idx}): {hlsl_field['name']} <-> {cpp_fields[best_match_idx]['name']} (sim: {best_similarity:.3f})"
            # )

    return matches


def strip_array_notation(name: str) -> str:
    """Remove any array notation from a field name, e.g., 'pad[3]' -> 'pad'."""
    return re.sub(r"\[.*?\]$", "", name)


def emphasize_if(condition: bool, value: str) -> str:
    return f"<ins>**_{value}_**</ins>" if condition and value else value


def compute_name_similarity(hlsl_field_name: str, cpp_field_name: str) -> float:
    """Compute name similarity using difflib and handle array notation.

    Args:
        hlsl_field_name: HLSL field name
        cpp_field_name: C++ field name

    Returns:
        float: Similarity score between 0 and 1
    """
    # Normalize to lowercase for case-insensitive comparison
    hlsl_name_lower = hlsl_field_name.lower()
    cpp_name_lower = cpp_field_name.lower()

    # Try direct sequence matching first
    matcher = difflib.SequenceMatcher(None, hlsl_name_lower, cpp_name_lower)
    normal_sim = matcher.ratio()

    # Enhanced: Also get LCS-based similarity
    matching_blocks = matcher.get_matching_blocks()
    lcs_length = sum(size for _, _, size in matching_blocks[:-1])  # Exclude final dummy block
    max_length = max(len(hlsl_name_lower), len(cpp_name_lower))
    lcs_sim = lcs_length / max_length if max_length > 0 else 0.0

    # Try with array notation stripped (preserving existing logic)
    hlsl_stripped = strip_array_notation(hlsl_name_lower)
    cpp_stripped = strip_array_notation(cpp_name_lower)
    stripped_matcher = difflib.SequenceMatcher(None, hlsl_stripped, cpp_stripped)
    stripped_sim = stripped_matcher.ratio()

    # Enhanced: LCS similarity for stripped names too
    stripped_blocks = stripped_matcher.get_matching_blocks()
    stripped_lcs_length = sum(size for _, _, size in stripped_blocks[:-1])
    stripped_max_length = max(len(hlsl_stripped), len(cpp_stripped))
    stripped_lcs_sim = stripped_lcs_length / stripped_max_length if stripped_max_length > 0 else 0.0

    # Look for substring relationships (preserving existing logic)
    if hlsl_stripped in cpp_stripped or cpp_stripped in hlsl_stripped:
        min_len = min(len(hlsl_stripped), len(cpp_stripped))
        max_len = max(len(hlsl_stripped), len(cpp_stripped))
        if max_len > 0 and min_len / max_len >= 0.7:  # At least 70% coverage
            substring_sim = 0.9  # Boost similarity for significant substring matches
            return max(normal_sim, stripped_sim, lcs_sim, stripped_lcs_sim, substring_sim)

    # Prefix/suffix boost
    shorter, longer = (
        (hlsl_stripped, cpp_stripped) if len(hlsl_stripped) <= len(cpp_stripped) else (cpp_stripped, hlsl_stripped)
    )
    if longer.startswith(shorter) or longer.endswith(shorter) or shorter in longer:
        # Full shorter string matches as prefix/suffix/substring
        prefix_sim = 0.85
        return max(normal_sim, stripped_sim, lcs_sim, stripped_lcs_sim, prefix_sim)

    # Return highest similarity score with small penalty for stripped matches
    best_score = max(normal_sim, lcs_sim, stripped_sim - 0.01, stripped_lcs_sim - 0.01)
    return best_score


def generate_comparison_table(
    hlsl_name: str,
    cpp_name: str,
    hlsl_data: StructDict,
    cpp_data: StructDict,
    align_matches: list[tuple[FieldDict | None, FieldDict | None]],
    report: dict[str, int | float],
    candidates: list[tuple[str, StructDict, float]],
    status: str = "",
    depth: int = 2,
    section_id: Optional[str] = None,
    show_top_candidate: bool = False,
) -> str:
    """Generate a comparison table for a pair of HLSL and C++ structs."""
    if not isinstance(hlsl_data, dict):
        logging.error(f"Invalid hlsl_data type: {type(hlsl_data)}, expected dict")
        return f"Error: Invalid HLSL data for {hlsl_name}\n"

    # Determine if we have a real match
    has_cpp_match = bool(cpp_name and cpp_data and report.get("score", 0) > 0)
    is_rejected_candidate = "(top candidate - rejected)" in cpp_name if cpp_name else False

    # If showing top candidate and no accepted match, use the top candidate from candidates list
    if show_top_candidate and not has_cpp_match and not is_rejected_candidate and candidates:
        top_candidate = candidates[0]  # candidates should be [(name, data, score, align_matches, report), ...]
        cpp_name = f"{top_candidate[0]} (top candidate - rejected)"
        cpp_data = top_candidate[1]
        # Use pre-computed alignment data if available (candidates with 5 elements)
        if len(top_candidate) >= 5:
            align_matches = top_candidate[3]
            report = top_candidate[4]
        else:
            # Fallback: compute alignment for legacy candidate format
            result = align_structs(hlsl_data, cpp_data, 0.5)
            if result:
                _, align_matches, report = result
            else:
                # Ensure we have some basic data even if alignment fails
                align_matches = []
                report = {"score": top_candidate[2], "field_diff_count": 0}
        is_rejected_candidate = True

    table = "---\n\n" if depth == 2 else ""
    # Add a section anchor for cross-linking
    if section_id:
        table += f'<a id="{section_id}"></a>\n'
    table += f"{'#' * depth} HLSL `{hlsl_name}` ({os.path.basename(hlsl_data.get('file', ''))})\n"
    table += f"**HLSL File:** [{hlsl_data.get('file', '')}:{hlsl_data.get('line', '')}]({create_link(hlsl_data.get('file', ''), hlsl_data.get('line', ''))})\n"

    # Handle unmatched case
    if not has_cpp_match and not is_rejected_candidate:
        table += "\n**No matching C++ struct found**\n"

        # Summary section for unmatched structs
        hlsl_fields = hlsl_data.get("fields", [])
        hlsl_total_fields = len([f for f in hlsl_fields if not is_padding_field(f)])
        table += f"\n{'#' * (depth + 1)} Summary:\n"
        table += f"- Total HLSL Fields: {hlsl_total_fields}\n"
        table += "- Status: Unmatched\n"
    else:
        # Handle matched or rejected candidate case
        if is_rejected_candidate:
            table += f"**C++**: `{cpp_name}`\n"
        else:
            table += f"**C++**: `{cpp_name}`\n"
        table += f"**C++ File:** [{cpp_data.get('file', '')}:{cpp_data.get('line', '')}]({create_link(cpp_data.get('file', ''), cpp_data.get('line', ''))})\n"
        table += f"**Match Score:** {report.get('score', 0):.2f}\n"

        # Field comparison table (for both matched and rejected candidates)
        # Always try to show field comparison for matched/rejected candidates
        if has_cpp_match or is_rejected_candidate:
            # If we don't have alignment data, compute it now
            if not align_matches and cpp_data:
                result = align_structs(hlsl_data, cpp_data, 0.5)
                if result:
                    _, align_matches, computed_report = result
                    # Update report with computed data if we don't have good data
                    if not report or report.get("score", 0) == 0:
                        report = computed_report

            if align_matches:
                field_rows = []
                rows_with_differences = report.get("field_diff_count", 0)

                for hlsl_field, cpp_field in align_matches:
                    hlsl_field_name = hlsl_field["name"] if hlsl_field else ""
                    hlsl_field_type = hlsl_field["type"] if hlsl_field else ""
                    cpp_field_name = cpp_field["name"] if cpp_field else ""
                    cpp_field_type = cpp_field["type"] if cpp_field else ""

                    # Compute similarities and apply emphasis
                    if hlsl_field and cpp_field:
                        type_sim = jellyfish.jaro_winkler_similarity(hlsl_field_type, cpp_field_type)
                        # For emphasis, use exact string comparison to catch array notation differences
                        exact_name_match = hlsl_field_name == cpp_field_name

                    # Apply emphasis using helper
                    if not hlsl_field or not cpp_field:
                        hlsl_field_type = emphasize_if(True, hlsl_field_type)
                        hlsl_field_name = emphasize_if(True, hlsl_field_name)
                        cpp_field_type = emphasize_if(True, cpp_field_type)
                        cpp_field_name = emphasize_if(True, cpp_field_name)
                    else:
                        # Use exact match for emphasis to catch array notation differences
                        hlsl_field_name = emphasize_if(not exact_name_match, hlsl_field_name)
                        cpp_field_name = emphasize_if(not exact_name_match, cpp_field_name)
                        hlsl_field_type = emphasize_if(type_sim < 1, hlsl_field_type)
                        cpp_field_type = emphasize_if(type_sim < 1, cpp_field_type)

                    field_rows.append({
                        "HLSL Type": hlsl_field_type,
                        "HLSL Field": hlsl_field_name,
                        "C++ Type": cpp_field_type,
                        "C++ Field": cpp_field_name,
                    })

                # Determine if we should default to closed
                should_close = rows_with_differences == 0
                table += f"\n<details{'>' if should_close else ' open>'}\n"
                table += f"<summary>{rows_with_differences} Field Differences</summary>\n\n"

                # Generate table using py_markdown_table
                if field_rows:
                    table += markdown_table(field_rows).set_params(row_sep="markdown", quote=False).get_markdown()

                table += "\n</details>\n"

        # Summary section for matched structs
        table += f"\n{'#' * (depth + 1)} Summary:\n"
        table += f"- Exact Field Matches: {report.get('exact_matches', 0)}\n"
        table += f"- High Similarity Field Matches: {report.get('high_sim_matches', 0)}\n"

        hlsl_total_fields = report.get("total_fields", 0)
        cpp_total_fields = report.get("cpp_total_fields", 0)
        table += f"- Total HLSL Fields: {hlsl_total_fields}\n"
        table += f"- Total C++ Fields: {cpp_total_fields}\n"
        table += f"- Field Name Differences: {report.get('field_name_diff_count', 0)}\n"
        table += f"- Field Type Differences: {report.get('field_type_diff_count', 0)}\n"

    # Always show candidates in a details section
    if candidates:
        total_candidates = len(candidates)
        table += f"\n<details>\n<summary>Top 5 of {total_candidates} Candidates Reviewed</summary>\n\n"
        candidate_rows = []
        for candidate in candidates[:5]:
            # Handle both old and new candidate formats
            if len(candidate) == 3:
                cand_name, cand_data, cand_score = candidate
            else:
                cand_name, cand_data, cand_score = candidate[0], candidate[1], candidate[2]

            # Count fields in candidate
            cand_fields = cand_data.get("fields", [])
            field_count = len([f for f in cand_fields if not is_padding_field(f)])

            candidate_rows.append({
                "Candidate Name": cand_name,
                "File": f"[{cand_data.get('file', '')}:{cand_data.get('line', '')}]({create_link(cand_data.get('file', ''), cand_data.get('line', ''))})",
                "Fields": str(field_count),
                "Similarity": f"{cand_score:.2f}",
            })

        table += markdown_table(candidate_rows).set_params(row_sep="markdown", quote=False).get_markdown()
        table += "\n</details>\n"

    return table


def count_field_differences(align_matches) -> tuple[int, int, int]:
    """
    Count field differences, returning (total_diff, name_diff, type_diff).
    """
    total_diff = 0
    name_diff = 0
    type_diff = 0
    for hlsl_field, cpp_field in align_matches:
        if hlsl_field and cpp_field:
            name_sim = compute_name_similarity(hlsl_field["name"], cpp_field["name"])
            type_sim = jellyfish.jaro_winkler_similarity(hlsl_field["type"], cpp_field["type"])
            if name_sim < 1:
                name_diff += 1
                total_diff += 1
            elif type_sim < 0.7:
                type_diff += 1
                total_diff += 1
        else:
            # Unmatched field counts as both a name and type diff
            name_diff += 1
            type_diff += 1
            total_diff += 1
    return total_diff, name_diff, type_diff


class InvalidStructDictType(Exception):
    def __init__(self, obj):
        super().__init__(f"Expected StructDict (dict) but got {type(obj)}")


class StructAnalyzer:
    """Class to handle struct comparison and analysis."""

    def __init__(
        self,
        hlsl_structs: dict[str, list[StructDict]],
        cpp_structs: dict[str, list[StructDict]],
    ):
        self.hlsl_structs: dict[str, list[StructDict]] = hlsl_structs
        self.cpp_structs: dict[str, list[StructDict]] = cpp_structs
        self.composite_buffers: dict[str, list[str]] = {}
        self.comparison_tables: list[str] = []
        self.buffer_locations: dict[tuple[str, str], tuple[str, int]] = {}
        self.analysis_results: dict[str, dict[str, Any]] = {}  # Store analysis links
        hlsl_count = sum(len(struct_list) for struct_list in hlsl_structs.values())
        cpp_count = sum(len(struct_list) for struct_list in cpp_structs.values())
        logging.info(f"Initialized with {hlsl_count} HLSL structs and {cpp_count} C++ structs")

    def add_buffer_location(self, file: str, buffer_name: str, line: int) -> None:
        """Add a buffer location to the map.

        Args:
            file: File path
            buffer_name: Name of the buffer
            line: Line number
        """
        key = f"{file.lower()}:{buffer_name.lower()}"
        self.buffer_locations[key] = (file, buffer_name)
        logging.debug(f"Added buffer location: {key} -> ({file}, {buffer_name})")

    def get_buffer_location(self, file: str, buffer_name: str, line: int) -> tuple[str, int] | None:
        """Get the location of a buffer.

        Args:
            file: File path
            buffer_name: Name of the buffer
            line: Line number

        Returns:
            tuple[str, int] | None: (file, line) tuple if found, None otherwise
        """
        key = f"{file.lower()}:{buffer_name.lower()}"
        return self.buffer_locations.get(key)

    def _is_composite_buffer(self, struct_data: StructDict) -> bool:
        if not struct_data.get("is_cbuffer", False):
            return False
        fields = struct_data.get("fields", [])
        if not fields:
            return False
        # At least one field is a user-defined struct type
        return any(field["type"] in self.hlsl_structs and field["type"] not in BASE_TYPE_SIZES for field in fields)

    def _process_composite_buffer(
        self,
        buffer_name: str,
        buffer_data: StructDict,
        matches: list,
        matched_cpp_structs: set[str],
    ) -> None:
        fields = buffer_data.get("fields", [])
        if not fields:
            return

        self.composite_buffers[buffer_name] = [field["type"] for field in fields]
        logging.debug(
            f"Processing composite buffer {buffer_name} with contained structs: {self.composite_buffers[buffer_name]}"
        )

        # --- Process sub-buffers first ---
        for field in fields:
            struct_type = field["type"]
            if struct_type in BASE_TYPE_SIZES:
                logging.warning(
                    f"Field {field['name']} with built-in type {struct_type} in composite buffer {buffer_name} "
                    f"at {buffer_data['file']}:{buffer_data['line']} will be skipped"
                )
                continue
            if struct_type in self.hlsl_structs:
                for struct_data in self.hlsl_structs[struct_type]:
                    if not isinstance(struct_data, dict):
                        logging.error(f"Invalid struct_data for {struct_type}: {struct_data}")
                        continue
                    struct_fields = self.get_nested_fields(struct_data)
                    candidates = self.find_struct_candidates(
                        struct_type, struct_data, struct_fields, matched_cpp_structs
                    )
                    # Use shared helper to find best match without score boosts for sub-structs
                    best_match, sorted_candidates = self._find_best_match_from_candidates(
                        struct_type, struct_data, struct_fields, candidates, apply_score_boosts=False
                    )

                    if best_match and self._is_match_good_enough(
                        best_match.score, best_match.report, sorted_candidates, struct_type, best_match.cpp_name
                    ):
                        unique_id = f"{best_match.cpp_file}:{best_match.cpp_name}"
                        matched_cpp_structs.add(unique_id)
                        matches.append(best_match)
                        logging.info(
                            f"Best match for sub-struct {struct_type}: {best_match.cpp_name} (score={best_match.score:.3f}) from {best_match.cpp_file}:{best_match.cpp_line}"
                        )
                    else:
                        logging.info(
                            f"No suitable C++ struct found for '{struct_type}' "
                            f"in {struct_data.get('file', 'unknown')}:{struct_data.get('line', 0)}"
                        )
            else:
                logging.warning(
                    f"Field type {struct_type} in composite buffer {buffer_name} "
                    f"at {buffer_data['file']}:{buffer_data['line']} not found in hlsl_structs"
                )

        # --- Now update fields and size ---
        hlsl_fields = self.get_nested_fields(buffer_data)
        buffer_data["fields"] = hlsl_fields

        buffer_data["size"] = calculate_struct_size(hlsl_fields, align_to_16=True)
        logging.debug(
            f"Updated composite buffer {buffer_name} size to {buffer_data['size']} bytes (fields: {[(f['name'], f['type'], f['size']) for f in hlsl_fields]})"
        )

        # --- Now match the composite buffer itself ---
        candidates = self.find_struct_candidates(buffer_name, buffer_data, hlsl_fields, matched_cpp_structs)

        # Use shared helper to find best match without score boosts for composite buffer
        best_match, sorted_candidates = self._find_best_match_from_candidates(
            buffer_name, buffer_data, hlsl_fields, candidates, apply_score_boosts=False
        )

        if best_match and self._is_match_good_enough(
            best_match.score, best_match.report, sorted_candidates, buffer_name, best_match.cpp_name
        ):
            unique_id = f"{best_match.cpp_file}:{best_match.cpp_name}"
            matched_cpp_structs.add(unique_id)
            matches.append(best_match)
        else:
            logging.info(
                f"No suitable C++ struct found for composite buffer '{buffer_name}' "
                f"in {buffer_data['file']}:{buffer_data['line']}"
            )

    def _generate_comparison_tables(self, matches: list[StructMatch], print_tables: bool = True) -> None:
        """Generate comparison tables for all matches.

        Args:
            matches: List of StructMatch objects.
            print_tables: Whether to print the tables immediately.
        """
        self.comparison_tables = []
        if not matches and print_tables:
            print("\nNo matching structs found between HLSL and C++.")
            return

        composite_matches: dict[str, list[StructMatch]] = {}
        regular_matches: list[StructMatch] = []

        logging.debug(f"Generating {len(matches)} comparison tables")

        for match in matches:
            hlsl_name = match.hlsl_name
            found_in_composite = False
            for composite_name, contained_structs in self.composite_buffers.items():
                if hlsl_name in contained_structs or hlsl_name == composite_name:
                    if composite_name not in composite_matches:
                        composite_matches[composite_name] = []
                    composite_matches[composite_name].append(match)
                    found_in_composite = True
                    break
            if not found_in_composite:
                regular_matches.append(match)

        for match in regular_matches:
            self._generate_single_comparison_table(match, print_tables)

        for composite_name, comp_matches in composite_matches.items():
            if print_tables:
                print(f"\n## Composite Buffer: {composite_name}\n")
                # Find the match for the composite buffer itself
                comp_match = next((m for m in comp_matches if m.hlsl_name == composite_name), None)
                if comp_match:
                    self._generate_single_comparison_table(comp_match, print_tables)
                else:
                    # Select the first StructDict for the composite buffer
                    comp_data = self.hlsl_structs.get(composite_name, [{}])[0]
                    if not comp_data:
                        logging.warning(f"No data found for composite buffer {composite_name}")
                        continue
                    print(
                        f"**HLSL File:** [{comp_data.get('file', '')}:{comp_data.get('line', '')}]({create_link(comp_data.get('file', ''), comp_data.get('line', ''))})\n"
                    )
                    print("**Best Match:** None\n")
            for match in comp_matches:
                if match.hlsl_name != composite_name:
                    self._generate_single_comparison_table(match, print_tables)

    def _is_match_good_enough(
        self,
        score: float,
        report: dict,
        candidates: list,
        hlsl_name: str = "",
        cpp_name: str = "",
    ) -> bool:
        """Improved match quality assessment with more nuanced criteria."""
        logging.debug(f"Checking match for {hlsl_name} vs {cpp_name}:")

        # 1. Absolute minimum score threshold (lowered)
        if score < 0.5:  # Reduced from 0.6
            logging.debug(f"Rejected match for {hlsl_name} vs {cpp_name}: score {score:.3f} below threshold 0.5")
            return False

        # 2. Enhanced field count analysis
        total_fields = report.get("total_fields", 1)
        cpp_total_fields = report.get("cpp_total_fields", 1)
        field_count_ratio = min(total_fields, cpp_total_fields) / max(total_fields, cpp_total_fields)

        # More lenient for small structs
        min_ratio = 0.4 if max(total_fields, cpp_total_fields) <= 3 else 0.6
        if field_count_ratio < min_ratio:
            logging.debug(
                f"Rejected match for {hlsl_name} vs {cpp_name}: field count ratio {field_count_ratio:.2f} < {min_ratio}"
            )
            return False

        # 3. Improved exact match requirements
        exact_matches = report.get("exact_matches", 0)
        high_sim_matches = report.get("high_sim_matches", 0)
        total_good_matches = exact_matches + high_sim_matches

        min_fields = min(total_fields, cpp_total_fields)

        # Require at least 30% good matches for small structs, 50% for larger ones
        required_match_ratio = 0.3 if min_fields <= 3 else 0.5
        good_match_ratio = total_good_matches / min_fields if min_fields > 0 else 0

        if good_match_ratio < required_match_ratio:
            logging.debug(
                f"Rejected match for {hlsl_name} vs {cpp_name}: good match ratio {good_match_ratio:.2f} < {required_match_ratio}"
            )
            return False

        # 4. Name similarity boost for very similar names
        name_sim = compute_name_similarity(hlsl_name, cpp_name)
        if name_sim > 0.8:
            # Very similar names get more lenient treatment
            return score > 0.4 and good_match_ratio > 0.2

        # 5. Size difference check (more lenient)
        size_difference = report.get("size_difference", 0)
        max_size_diff = 128 if max(total_fields, cpp_total_fields) > 5 else 64
        if size_difference > max_size_diff:
            logging.debug(
                f"Rejected match for {hlsl_name} vs {cpp_name}: size difference {size_difference} > {max_size_diff} bytes"
            )
            return False

        # 6. Relative candidate quality (more lenient)
        if len(candidates) > 1:
            best_score = score
            # Handle both StructCandidate objects and old tuple format
            if hasattr(candidates[1], "score"):
                second_best_score = candidates[1].score
            else:
                second_best_score = candidates[1][2] if len(candidates[1]) > 2 else 0
            score_gap = best_score - second_best_score

            # Only reject if the best candidate isn't significantly better AND the score is low
            if score_gap < 0.05 and best_score < 0.6:
                logging.debug(f"Rejected match for {hlsl_name} vs {cpp_name}: insufficient score gap {score_gap:.3f}")
                return False

        return True

    def compare_all_structs(self, result_map: ResultMap) -> dict[str, dict[str, str | bool]]:
        """Compare HLSL and C++ structs, generating alignment reports."""
        analysis_links: dict[str, dict[str, str | bool]] = {}
        matches: list[StructMatch] = []
        matched_cpp_structs = set()

        match_counts = {
            "Matched": 0,  # Perfect matches
            "Mismatched": 0,  # Partial matches
            "Unmatched": 0,  # No matches
        }

        if not self.hlsl_structs:
            print("\nNo HLSL structs found.")
            self.analysis_links = analysis_links
            return analysis_links

        for hlsl_name, hlsl_struct_list in self.hlsl_structs.items():
            for hlsl_data in hlsl_struct_list:
                if not isinstance(hlsl_data, dict):
                    logging.error(f"Invalid hlsl_data for {hlsl_name}: {hlsl_data}")
                    continue

                logging.debug(
                    f"Processing HLSL struct: {hlsl_name} : size={hlsl_data.get('size')} : {hlsl_data.get('file', '')}:{hlsl_data.get('line', '')}"
                )

                if self._is_composite_buffer(hlsl_data):
                    self._process_composite_buffer(hlsl_name, hlsl_data, matches, matched_cpp_structs)
                    continue

                if hlsl_data.get("is_template") and "template_type" in hlsl_data:
                    template_type = hlsl_data["template_type"]
                    if template_type in self.hlsl_structs:
                        hlsl_data["fields"] = self.hlsl_structs[template_type][0]["fields"]
                hlsl_fields = self.get_nested_fields(hlsl_data)

                if not hlsl_fields:
                    continue

                # Find all potential candidates (including exact name matches)
                candidates = self.find_struct_candidates(hlsl_name, hlsl_data, hlsl_fields, matched_cpp_structs)

                # Use shared helper to find best match with score boosts
                best_match, sorted_candidates = self._find_best_match_from_candidates(
                    hlsl_name, hlsl_data, hlsl_fields, candidates, apply_score_boosts=True
                )

                # ONLY AFTER finding the best match, check if it's good enough
                if best_match:
                    # Check if the match is good enough
                    if self._is_match_good_enough(
                        best_match.score, best_match.report, sorted_candidates, hlsl_name, best_match.cpp_name
                    ):
                        matches.append(best_match)
                        logging.debug(
                            f"Accepted match for {hlsl_name}: {best_match.cpp_name} (score={best_match.score:.3f})"
                        )
                    else:
                        # For rejected matches, create a match with empty cpp info but preserve candidate data
                        matches.append(
                            StructMatch(
                                hlsl_name=hlsl_name,
                                hlsl_file=hlsl_data["file"],
                                hlsl_line=hlsl_data["line"],
                                cpp_name="",  # Empty cpp_name indicates rejection
                                cpp_file="",  # Empty cpp_file
                                cpp_line=0,  # Empty cpp_line
                                score=0.0,  # Zero score for status determination
                                align_matches=[],  # Empty align_matches for rejected
                                report=best_match.report,  # Preserve the actual report from best match
                                candidates=sorted_candidates,  # Preserve sorted candidates with full alignment info
                            )
                        )
                        logging.debug(
                            f"Rejected match for {hlsl_name}: {best_match.cpp_name} (score={best_match.score:.3f}) - quality too low"
                        )
                else:
                    # No candidates found
                    matches.append(
                        StructMatch(
                            hlsl_name=hlsl_name,
                            hlsl_file=hlsl_data["file"],
                            hlsl_line=hlsl_data["line"],
                            cpp_name="",  # Empty cpp_name
                            cpp_file="",  # Empty cpp_file
                            cpp_line=0,  # Empty cpp_line
                            score=0.0,  # Zero score
                            align_matches=[],  # Empty align_matches
                            report={  # Empty report
                                "score": 0.0,
                                "exact_matches": 0,
                                "high_sim_matches": 0,
                                "total_fields": len(hlsl_fields),
                                "missing_fields": len(hlsl_fields),
                            },
                            candidates=sorted_candidates,  # Empty sorted_candidates
                        )
                    )

        self.matches = matches

        self._generate_comparison_tables(matches, print_tables=False)
        # Store analysis_links as an attribute for later use in print_comparison_tables
        self.analysis_links = analysis_links

        # Update analysis links in result_map
        for match in matches:
            key = f"{match.hlsl_file.lower()}:{match.hlsl_name.lower()}"

            # New logic for status:
            if not match.cpp_name:
                status = "Unmatched"
                display_name = "Unmatched"
                match_counts["Unmatched"] += 1
            else:
                field_diff_count = match.report.get("field_diff_count", 0)
                cpp_total_fields = match.report.get("cpp_total_fields", 0)
                hlsl_total_fields = match.report.get("total_fields", 0)
                min_fields = min(hlsl_total_fields, cpp_total_fields)
                diff_ratio = field_diff_count / max(1, min_fields)
                debug_msg = f"DEBUG: {match.hlsl_name} vs {match.cpp_name}: field_diff_count={field_diff_count}, min_fields={min_fields}, diff_ratio={diff_ratio}, score={match.score}"
                add_debug_info(debug_msg)
                if diff_ratio > 0.5 or match.score < 0.75:
                    status = "Unmatched"
                    display_name = "Unmatched"
                    match_counts["Unmatched"] += 1
                elif field_diff_count == 0:
                    status = "Matched"
                    display_name = match.cpp_name
                    match_counts["Matched"] += 1
                elif field_diff_count <= 1:
                    status = f"Mismatched ({match.cpp_name})"
                    display_name = f"Mismatched ({match.cpp_name})"
                    match_counts["Mismatched"] += 1
                else:
                    status = f"Mismatched ({match.cpp_name})"
                    display_name = f"Mismatched ({match.cpp_name})"
                    match_counts["Mismatched"] += 1

            analysis_links[key] = {
                "link": f"[{display_name}](#hlsl-{match.hlsl_name.lower()}-{os.path.basename(match.hlsl_file).lower()})",
                "is_match": bool(match.cpp_name),
                "cpp_name": match.cpp_name,
                "cpp_file": match.cpp_file,
                "cpp_line": match.cpp_line,
                "score": match.score,
                "status": status,
            }
            # Look for entries using template type for struct analysis
            for result_key, entry in result_map.items():
                file_path = entry.get("File Path", "").lower()
                buffer_name = entry.get("Name", "").lower()
                template_type = entry.get("Template Type", "")

                # Use template type for struct analysis if available, otherwise use buffer name
                struct_name = template_type.lower() if template_type else buffer_name

                # Check if this buffer entry should link to this struct analysis
                if file_path == match.hlsl_file.lower() and struct_name == match.hlsl_name.lower():
                    entry["Matching Struct Analysis"] = analysis_links[key]["link"]
                    logging.debug(
                        f"Updated result_map for {result_key} using template type {template_type} with link: {analysis_links[key]['link']}"
                    )
                    break
            else:
                # Fallback: try name-only match for user-defined types
                for result_key, entry in result_map.items():
                    buffer_name = entry.get("Name", "").lower()
                    template_type = entry.get("Template Type", "")
                    struct_name = template_type.lower() if template_type else buffer_name

                    if struct_name == match.hlsl_name.lower() and match.hlsl_name.lower() not in BASE_TYPE_SIZES:
                        entry["Matching Struct Analysis"] = analysis_links[key]["link"]
                        logging.debug(f"Fallback name-only match for {match.hlsl_name} to result_map key {result_key}")
                        break
                else:
                    logging.warning(f"No buffer table entry found for struct {key}")

        for entry in result_map.values():
            if "Matching Struct Analysis" not in entry:
                entry["Matching Struct Analysis"] = "Unmatched"
                name = entry.get("Name", "unknown")
                file_path = entry.get("File Path", "unknown")
                logging.debug(f"Buffer {name} in {file_path} not matched to any struct")

        return analysis_links

    def _generate_single_comparison_table(self, match: StructMatch, print_tables: bool) -> None:
        """Generate a single comparison table.

        Args:
            match: StructMatch containing match data.
            print_tables: Whether to print the table immediately.
        """
        # Use hlsl_data from match if available, else find it
        hlsl_data = next(
            (
                m
                for m in self.hlsl_structs.get(match.hlsl_name, [])
                if m["file"] == match.hlsl_file and m["line"] == match.hlsl_line
            ),
            {},
        )
        if not hlsl_data:
            logging.warning(f"No HLSL data found for {match.hlsl_name} in {match.hlsl_file}:{match.hlsl_line}")

        # Only use cpp_data if there is a valid match (score > 0 and cpp_name is not empty)
        if match.cpp_name and match.score > 0:
            cpp_data = next(
                (
                    s
                    for s in self.cpp_structs.get(match.cpp_name, [])
                    if s["file"] == match.cpp_file and s["line"] == match.cpp_line
                ),
                {},
            )
        else:
            cpp_data = {}

        # Validate types
        if not isinstance(hlsl_data, dict):
            raise InvalidStructDictType(hlsl_data)
        if not isinstance(cpp_data, dict):
            raise InvalidStructDictType(cpp_data)

        logging.debug(f"Generating table for HLSL {match.hlsl_name} {match.hlsl_file}:{match.hlsl_line} ")

        # Convert StructCandidate objects to the format expected by generate_comparison_table
        candidates = [(c.name, c.data, c.score) for c in match.candidates]

        table_data = {
            "hlsl_name": match.hlsl_name,
            "cpp_name": match.cpp_name if cpp_data else "",
            "hlsl_data": hlsl_data,
            "cpp_data": cpp_data,
            "align_matches": match.align_matches if cpp_data else [],
            "report": match.report,
            "candidates": candidates,
        }
        self.comparison_tables.append(table_data)

        if print_tables:
            print(
                generate_comparison_table(
                    match.hlsl_name,
                    match.cpp_name if cpp_data else "",
                    hlsl_data,
                    cpp_data,
                    match.align_matches if cpp_data else [],
                    match.report,
                    candidates,
                    status=self.analysis_links.get(f"{match.hlsl_file.lower()}:{match.hlsl_name.lower()}", {}).get(
                        "status", ""
                    ),
                    section_id=f"hlsl-{match.hlsl_name.lower()}-{os.path.basename(match.hlsl_file).lower()}",
                )
            )

    def print_comparison_tables(self, only_matched: bool = False, show_top_candidate: bool = False) -> None:
        """Print comparison tables for all HLSL structs, with sub-buffers nested under their parents, using result_map as the source."""
        print("\n# Struct Comparison Results")
        printed_keys = set()
        # Build a lookup for table data
        table_lookup = {f"{t['hlsl_name']}:{t['hlsl_data']['file']}": t for t in self.comparison_tables}
        logging.debug(f"Table lookup keys: {list(table_lookup.keys())}")
        # Build a lookup for composite buffer relationships
        composite_to_subs: dict[str, list[str]] = {}
        for buffer_name, sub_names in self.composite_buffers.items():
            for sub_name in sub_names:
                # Find the file for the sub-buffer
                for t in self.comparison_tables:
                    if t["hlsl_name"] == sub_name:
                        sub_key = f"{sub_name}:{t['hlsl_data']['file']}"
                        parent_key = f"{buffer_name}:{t['hlsl_data']['file']}"
                        composite_to_subs.setdefault(parent_key, []).append(
                            sub_key
                        )  # Use buffer_locations as the source
        logging.debug(f"Buffer locations: {list(self.buffer_locations.items())}")
        for _key, entry in self.buffer_locations.items():
            file, buffer_name = entry
            composite_key = f"{buffer_name}:{file}"
            # If this is a composite buffer, check if any sub-buffers should be printed
            if composite_key in composite_to_subs:
                sub_keys = composite_to_subs[composite_key]
                # Check if composite or any sub-buffer should be printed
                keys_to_check = [composite_key, *sub_keys]

                def should_print(k):
                    table = table_lookup.get(k)
                    if not table:
                        return False
                    status = self.analysis_links.get(
                        f"{table['hlsl_data']['file'].lower()}:{table['hlsl_name'].lower()}", {}
                    ).get("status", "")
                    if only_matched:
                        return status != "Unmatched"
                    return True

                should_print_any = any(should_print(k) for k in keys_to_check)
                if should_print_any and composite_key not in printed_keys:
                    print(f"\n## Composite Buffer: {buffer_name}\n")
                    # Print composite buffer table if present
                    table = table_lookup.get(composite_key)
                    if table:
                        # Handle show_top_candidate for composite buffer
                        cpp_name = table["cpp_name"]
                        cpp_data = table["cpp_data"]
                        align_matches = table["align_matches"]
                        report = table["report"]

                        # If no accepted match but we want to show top candidate, extract from candidates
                        if show_top_candidate and not cpp_name and table["candidates"]:
                            top_candidate = table["candidates"][0]  # Best candidate
                            cpp_name = f"{top_candidate[0]} (top candidate - rejected)"
                            cpp_data = top_candidate[1]
                            # For regular candidate tuples, use empty alignment data
                            align_matches = []
                            report = table["report"]

                        print(
                            generate_comparison_table(
                                table["hlsl_name"],
                                cpp_name,
                                table["hlsl_data"],
                                cpp_data,
                                align_matches,
                                report,
                                table["candidates"],
                                status=self.analysis_links.get(
                                    f"{table['hlsl_data']['file'].lower()}:{table['hlsl_name'].lower()}",
                                    {},
                                ).get("status", ""),
                                show_top_candidate=show_top_candidate,
                            )
                        )
                        printed_keys.add(composite_key)
                    # Print sub-buffers
                    for sub_key in sub_keys:
                        if sub_key in printed_keys:
                            continue
                        table = table_lookup.get(sub_key)
                        if table and should_print(sub_key):
                            # Handle show_top_candidate for sub-buffers
                            cpp_name = table["cpp_name"]
                            cpp_data = table["cpp_data"]
                            align_matches = table["align_matches"]
                            report = table["report"]

                            # If no accepted match but we want to show top candidate, use the first candidate
                            if show_top_candidate and not cpp_name and table["candidates"]:
                                top_candidate = table["candidates"][0]  # First candidate should be best after sorting
                                cpp_name = f"{top_candidate[0]} (top candidate - rejected)"
                                cpp_data = top_candidate[1]
                                # Use the alignment data from the candidate
                                align_matches = top_candidate[3] if len(top_candidate) > 3 else []
                                report = top_candidate[4] if len(top_candidate) > 4 else table["report"]

                            print(
                                generate_comparison_table(
                                    table["hlsl_name"],
                                    cpp_name,
                                    table["hlsl_data"],
                                    cpp_data,
                                    align_matches,
                                    report,
                                    table["candidates"],
                                    status=self.analysis_links.get(
                                        f"{table['hlsl_data']['file'].lower()}:{table['hlsl_name'].lower()}",
                                        {},
                                    ).get("status", ""),
                                    depth=3,
                                )
                            )
                            printed_keys.add(sub_key)
            # Otherwise, print as a regular buffer if not already printed
            else:
                key_lookup = f"{buffer_name}:{file}"
                logging.debug(f"Looking for key_lookup: {key_lookup}")
                if key_lookup in printed_keys:
                    logging.debug(f"Already printed: {key_lookup}")
                    continue
                table = table_lookup.get(key_lookup)
                if table:
                    logging.debug(f"Found table for: {key_lookup}")
                    status = self.analysis_links.get(
                        f"{table['hlsl_data']['file'].lower()}:{table['hlsl_name'].lower()}", {}
                    ).get("status", "")
                    logging.debug(f"Status for {key_lookup}: {status}")
                    if only_matched and status == "Unmatched":
                        logging.debug(f"Skipping unmatched: {key_lookup}")
                        continue
                    # Handle show_top_candidate for regular buffers
                    cpp_name = table["cpp_name"]
                    cpp_data = table["cpp_data"]

                    # If no accepted match but we want to show top candidate, use the first candidate
                    if show_top_candidate and not cpp_name and table["candidates"]:
                        top_candidate = table["candidates"][0]  # First candidate should be best after sorting
                        cpp_name = f"{top_candidate[0]} (top candidate - rejected)"
                        cpp_data = top_candidate[1]
                        # Use empty alignment data for simple candidate tuples
                        align_matches = []
                        report = table["report"]
                    else:
                        align_matches = table["align_matches"]
                        report = table["report"]

                    print(
                        generate_comparison_table(
                            table["hlsl_name"],
                            cpp_name,
                            table["hlsl_data"],
                            cpp_data,
                            align_matches,
                            report,
                            table["candidates"],
                            status=status,
                        )
                    )
                    printed_keys.add(key_lookup)
                else:
                    logging.debug(f"No table found for: {key_lookup}")

    def update_result_map(self, result_map: dict[str, dict[str, Any]]) -> None:
        """Update the result map with stored analysis results.

        Args:
            result_map: Dictionary mapping keys to buffer metadata, to be updated with analysis links.
        """
        if not self.analysis_results:
            logging.debug("No analysis results available to update result_map")
            return

        # Create template type mapping
        template_types: dict[str, str] = {}
        for hlsl_name, struct_list in self.hlsl_structs.items():
            for struct in struct_list:
                if struct.get("is_template") and "template_type" in struct:
                    template_types[hlsl_name] = struct["template_type"]

        for key, analysis in self.analysis_results.items():
            if key in result_map:
                result_map[key]["Matching Struct Analysis"] = analysis["link"]
                logging.debug(f"Updated result_map for {key} with link: {analysis['link']}")
                continue

            try:
                file, buffer_name = key.split(":")
            except ValueError:
                logging.warning(f"Invalid key format in analysis_results: {key}")
                continue

            for result_key, entry in result_map.items():
                if entry.get("File Path", "").lower() == file and entry.get("Name", "").lower() == buffer_name:
                    result_map[result_key]["Matching Struct Analysis"] = analysis["link"]
                    logging.debug(f"Matched {key} to result_map key {result_key}")
                    break
                # Fallback: try file+name match (legacy)
                elif entry.get("File Path", "").lower() == file and entry.get("Name", "").lower() == buffer_name:
                    result_map[result_key]["Matching Struct Analysis"] = analysis["link"]
                    logging.debug(f"Fallback file+name match for {key} to result_map key {result_key}")
                    break
                # Fallback: try name-only match for user-defined types
                elif entry.get("Name", "").lower() == buffer_name and buffer_name not in BASE_TYPE_SIZES:
                    result_map[result_key]["Matching Struct Analysis"] = analysis["link"]
                    logging.debug(f"Fallback name-only match for {key} to result_map key {result_key}")
                    break
                else:
                    logging.warning(f"No buffer table entry found for struct {key}")
        for entry in result_map.values():
            if "Matching Struct Analysis" not in entry:
                entry["Matching Struct Analysis"] = "Unmatched"
                name = entry.get("Name", "unknown")
                file_path = entry.get("File Path", "unknown")
                logging.debug(f"Buffer {name} in {file_path} not matched to any struct")

    def get_nested_fields(self, struct_data: StructDict) -> list[FieldDict]:
        """Get all fields from a struct, including nested struct fields.

        Args:
            struct_data: Dictionary containing struct metadata.

        Returns:
            list[FieldDict]: List of field dictionaries, including nested fields.
        """
        fields = struct_data.get("fields", [])
        if not fields:
            return []

        processed_fields: list[FieldDict] = []

        # Determine which struct dictionary to use based on the struct's origin
        struct_file = struct_data.get("file", "")
        is_cpp_struct = struct_file.endswith((".cpp", ".h", ".hpp"))
        struct_dict = self.cpp_structs if is_cpp_struct else self.hlsl_structs

        for field in fields:
            field_type = field.get("type", "")
            if field_type in struct_dict:
                # Process each StructDict in the list for nested structs
                nested_structs = struct_dict[field_type]
                struct_file = struct_data.get("file", "")
                preferred_struct = None
                for nested_struct in nested_structs:
                    if not isinstance(nested_struct, dict):
                        logging.error(f"Invalid nested struct for {field_type}: {nested_struct}")
                        continue
                    if nested_struct.get("file", "") == struct_file:
                        preferred_struct = nested_struct
                        break
                if not preferred_struct:
                    preferred_struct = nested_structs[0]  # fallback to first if no file match
                nested_fields = self.get_nested_fields(preferred_struct)
                processed_fields.extend(nested_fields)
            else:
                processed_fields.append(field)

        return processed_fields

    def get_field_name(self, field: FieldDict) -> str:
        """Get the field name without array notation.

        Args:
            field: Field dictionary containing name information.

        Returns:
            str: Field name without array notation.

        Example:
            >>> analyzer.get_field_name({"name": "myArray[3]"})
            'myArray'
        """
        if not isinstance(field, dict) or "name" not in field:
            logging.warning(f"Invalid field dictionary: {field}")
            return ""
        return strip_array_notation(field["name"])

    def find_struct_candidates(
        self,
        hlsl_name: str,
        hlsl_data: StructDict,
        hlsl_fields: list[FieldDict],
        matched_cpp_structs: set[str],
    ) -> list[tuple[str, StructDict, float]]:
        """Find candidate C++ structs for an HLSL struct.

        Args:
            hlsl_name: Name of the HLSL struct.
            hlsl_data: HLSL struct metadata.
            hlsl_fields: List of HLSL fields.
            matched_cpp_structs: Set of already matched C++ struct names.

        Returns:
            list[tuple[str, StructDict, float]]: List of (cpp_name, cpp_data, similarity) tuples.
        """
        candidates: list[tuple[str, StructDict, float]] = []

        for cpp_name, cpp_struct_list in self.cpp_structs.items():
            for cpp_data in cpp_struct_list:
                # Test original version
                original_fields = cpp_data.get("fields", [])
                if original_fields:
                    temp_cpp_data = dict(cpp_data)
                    temp_cpp_data["fields"] = original_fields

                    result = align_structs(temp_cpp_data, hlsl_data, 0.5)
                    if result is not None:
                        alignment_score, _, _ = result
                        candidates.append((cpp_name, temp_cpp_data, alignment_score))

                        logging.debug(
                            f"\t\t Candidate {cpp_name} (original) from {cpp_data.get('file', 'unknown')}: "
                            f"alignment_score={alignment_score:.2f}, total_size={calculate_struct_size(original_fields)}, fields: {[(f['name'], f['type'], f['size']) for f in original_fields]}"
                        )

                # Test flattened version if different
                flattened_fields = self.get_nested_fields(cpp_data)
                if flattened_fields != original_fields:
                    temp_cpp_data = dict(cpp_data)
                    temp_cpp_data["fields"] = flattened_fields
                    temp_cpp_data["size"] = calculate_struct_size(flattened_fields)

                    result = align_structs(temp_cpp_data, hlsl_data, 0.5)
                    if result is not None:
                        alignment_score, _, _ = result
                        candidates.append((f"{cpp_name}", temp_cpp_data, alignment_score))

                        logging.debug(
                            f"\t\t Candidate {cpp_name} (flattened) from {cpp_data.get('file', 'unknown')}: "
                            f"alignment_score={alignment_score:.2f}, total_size={calculate_struct_size(flattened_fields)}, fields: {[(f['name'], f['type'], f['size']) for f in flattened_fields]}"
                        )
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    def _find_best_match_from_candidates(
        self,
        hlsl_name: str,
        hlsl_data: StructDict,
        hlsl_fields: list[FieldDict],
        candidates: list[tuple[str, StructDict, float]],
        apply_score_boosts: bool = True,
    ) -> tuple[StructMatch | None, list[StructCandidate]]:
        """Find the best match among candidates and return StructMatch with candidate list.

        Args:
            hlsl_name: Name of the HLSL struct
            hlsl_data: HLSL struct metadata
            hlsl_fields: List of HLSL fields
            candidates: List of (cpp_name, cpp_data, similarity) tuples
            apply_score_boosts: Whether to apply score boosts for exact name matches and outliers

        Returns:
            tuple: (StructMatch_or_None, list_of_StructCandidates)
        """
        if not candidates:
            return None, []

        evaluated_candidates = []
        best_score = -1
        best_match = None

        for idx, (cpp_name, cpp_data, similarity) in enumerate(candidates):
            if not isinstance(cpp_data, dict):
                logging.error(f"Invalid cpp_data for {cpp_name}: {cpp_data}")
                continue

            struct_name_weight = 0.7 if len(hlsl_fields) <= 3 else 0.5
            result = align_structs(cpp_data, hlsl_data, struct_name_weight)

            if result is not None:
                score, align_matches, report = result
                logging.debug(f"Evaluating {hlsl_name} vs {cpp_name}: score={score:.3f}")

                # Apply score boosts if requested
                if apply_score_boosts:
                    if cpp_name == hlsl_name:
                        score *= 1.05
                    if similarity > 0.85 and (
                        idx == 0 and (len(candidates) == 1 or similarity - candidates[1][2] > 0.2)
                    ):
                        score *= 1.05

                # Store evaluated candidate with full alignment info
                candidate = StructCandidate(
                    name=cpp_name, data=cpp_data, score=score, align_matches=align_matches, report=report
                )
                evaluated_candidates.append(candidate)

                if score > best_score:
                    logging.debug(f"New best match for {hlsl_name}: {cpp_name} (score={score:.3f})")
                    best_score = score
                    best_match = StructMatch(
                        hlsl_name=hlsl_name,
                        hlsl_file=hlsl_data["file"],
                        hlsl_line=hlsl_data["line"],
                        cpp_name=cpp_name,
                        cpp_file=cpp_data["file"],
                        cpp_line=cpp_data["line"],
                        score=score,
                        align_matches=align_matches,
                        report=report,
                        candidates=evaluated_candidates,  # Will be updated below
                    )

        # Re-sort candidates by actual evaluation scores
        evaluated_candidates.sort(key=lambda x: x.score, reverse=True)

        # Update the match to use sorted candidates
        if best_match:
            best_match.candidates = evaluated_candidates

        return best_match, evaluated_candidates


def main() -> None:
    """Main entry point for scanning HLSL shaders and generating a buffer table."""
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="Scan HLSL/C++ buffers and print struct analysis.")
    parser.add_argument(
        "--only-matched",
        action="store_true",
        help="Only print buffers with matched or mismatched struct analysis (default: print all discovered buffers)",
    )
    parser.add_argument(
        "--show-top-candidate",
        action="store_true",
        help="Show fields of the top candidate struct even if the match was rejected",
    )
    parser.add_argument(
        "--show-conflicts",
        action="store_true",
        help="Show register conflicts analysis (default: disabled)",
    )
    args = parser.parse_args()
    cwd = os.getcwd()

    scanner = FileScanner(cwd)
    defines_list = get_defines_list()
    shader_pattern = re.compile(
        r"""(?P<type>
                (?:cbuffer|ConstantBuffer<(?P<template_type>\w+)>) |
                (?:(?:RW)?(?:StructuredBuffer|Buffer|Texture1D|Texture2D|Texture3D|TextureCube|RWBuffer|RWTexture1D|RWTexture2D|RWTexture3D|RWTextureCube|SamplerState|SamplerComparisonState))
                (?:<(?P<template_name>\w+)>)?
            )
            \s+
            (?P<name>\w+)
            \s*:\s*register\s*\(
                (?P<buffer_type>[a-z])
                (?P<buffer_number>\d+)
            \)
            (?:\s*;\s*|$)
        """,
        re.MULTILINE | re.VERBOSE,
    )
    hlsl_types = {"b": "CBV", "t": "SRV", "u": "UAV", "s": "Sampler"}
    pattern = re.compile(r".*\.(hlsl|hlsli)$", re.IGNORECASE)
    feature_pattern = re.compile(r"features[/\\](?P<feature>[^/\\]+)")

    results, compilation_units = scanner.scan_for_buffers(
        pattern=pattern,
        feature_pattern=feature_pattern,
        shader_pattern=shader_pattern,
        hlsl_types=hlsl_types,
        defines_list=defines_list,
    )

    result_map = {f"{entry['File Path'].lower()}:{entry['Name'].lower()}": entry for entry in results}
    logging.debug(f"Result map contains {len(result_map)} entries: {list(result_map.keys())}")

    hlsl_structs, cpp_structs = scanner.scan_for_structs()
    analyzer = StructAnalyzer(hlsl_structs, cpp_structs)

    for entry in result_map.values():
        file = entry.get("File Path")
        buffer_name = entry["Name"]
        line = entry.get("Original Line")
        if file is not None and buffer_name is not None:
            analyzer.add_buffer_location(file, buffer_name, line)

    analyzer.compare_all_structs(result_map)
    analyzer.update_result_map(result_map)

    # Add all struct definitions to buffer_locations for printing
    for hlsl_name, hlsl_struct_list in analyzer.hlsl_structs.items():
        for hlsl_data in hlsl_struct_list:
            if isinstance(hlsl_data, dict):
                file = hlsl_data.get("file", "")
                line = hlsl_data.get("line", 0)
                if file:
                    analyzer.add_buffer_location(file, hlsl_name, line)
                    logging.debug(f"Added struct definition to buffer_locations: {hlsl_name} from {file}:{line}")
                else:
                    logging.debug(f"Skipping struct {hlsl_name} - no file information")
    print_buffers_and_conflicts(result_map, compilation_units, show_conflicts=args.show_conflicts)
    analyzer.print_comparison_tables(only_matched=args.only_matched, show_top_candidate=args.show_top_candidate)


if __name__ == "__main__":
    main()
