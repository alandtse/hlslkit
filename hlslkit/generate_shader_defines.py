"""Generate shader defines YAML from CommunityShaders log.

This script parses a `CommunityShaders.log` file to extract shader configurations, warnings, and errors.
It generates a `shader_defines.yaml` file summarizing shader entry points, preprocessor defines, and compilation issues.
The output is structured for use with `compile_shaders.py`.

Example:
    ```bash
    python generate_shader_defines.py --log CommunityShaders.log --output shader_defines.yaml
    ```

Dependencies:
    - `yaml`: For generating YAML output.
    - `tqdm`: For progress bars.
    - `gooey` (optional): For GUI support.
"""

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import yaml
from tqdm import tqdm

try:
    from gooey import Gooey, GooeyParser

    HAS_GOOEY = True
except ImportError:
    Gooey = lambda x: x
    HAS_GOOEY = False


@dataclass
class CompilationTask:
    """A shader compilation task extracted from the log.

    Attributes:
        process_id (str): Process ID of the compilation task.
        entry_point (str): Shader entry point (e.g., 'main:vertex:1234').
        file_path (str): Path to the shader file.
        defines (list[str]): Preprocessor defines used.
        start_time (datetime): Start time of the compilation.
        end_time (datetime | None): End time of the compilation, if completed.
    """

    process_id: str
    entry_point: str
    file_path: str
    defines: list[str]
    start_time: datetime
    end_time: datetime | None = None


def parse_timestamp(line: str) -> datetime:
    """Parse a timestamp from a log line.

    Args:
        line (str): The log line containing a timestamp.

    Returns:
        datetime: The parsed datetime object.
    """
    timestamp_str = line[1:13]
    return datetime.strptime(timestamp_str, "%H:%M:%S.%f")


def count_compiling_lines(log_file: str) -> int:
    """Count the number of compilation log lines in a file.

    Args:
        log_file (str): Path to the log file.

    Returns:
        int: Number of lines containing '[D] Compiling'.
    """
    with open(log_file, encoding="utf-8") as f:
        return sum(1 for line in f if "[D] Compiling" in line)


def count_log_blocks(log_file: str) -> int:
    """Count the number of log blocks (warnings, errors, completions) in a file.

    Args:
        log_file (str): Path to the log file.

    Returns:
        int: Number of log blocks.
    """
    with open(log_file, encoding="utf-8") as f:
        return sum(
            1
            for line in f
            if any(
                keyword in line
                for keyword in (
                    "[D] Shader logs:",
                    "[E] Failed to compile",
                    "[W] Shader compilation failed",
                    "[D] Adding Completed shader",
                )
            )
        )


def normalize_path(file_path: str) -> str:
    """Normalize a file path by standardizing separators and extracting relative path.    This function ensures paths are consistent across platforms, converting backslashes to forward slashes
    and attempting to extract the relative path from the `Shaders` directory (case-insensitive). If the
    `Shaders` directory is not found, the path is returned as-is with normalized separators.

    Args:
        file_path (str): The file path to normalize.

    Returns:
        str: The normalized file path, relative to the Shaders directory if present.
    """
    file_path = file_path.replace("\\", "/")
    file_path = re.sub(r"/+", "/", file_path)
    pattern = r"(?i).*?\bShaders[/](.*)"
    match = re.search(pattern, file_path)
    if match:
        norm_path = match.group(1).replace("\\", "/")
        logging.debug(f"Normalized path (Shaders found): {file_path} -> {norm_path}")
        return norm_path
    norm_path = file_path.replace("\\", "/")
    logging.debug(f"Normalized path (no Shaders, using as-is): {file_path} -> {norm_path}")
    return norm_path


def get_shader_type_from_entry(entry_point: str) -> str:
    """Determine the shader type from an entry point string.

    Args:
        entry_point (str): The entry point string (e.g., 'main:vertex:1234').

    Returns:
        str: The shader type (VSHADER, PSHADER, CSHADER, or UNKNOWN).
    """
    parts = entry_point.split(":")
    if len(parts) >= 3:
        shader_type = parts[1].lower()
        shader_types = {"vertex": "VSHADER", "pixel": "PSHADER", "compute": "CSHADER"}
        return shader_types.get(shader_type, "UNKNOWN")
    return "UNKNOWN"


def collect_tasks(lines: list[str]) -> list[CompilationTask]:
    """Collect compilation tasks from log lines.

    Args:
        lines (list[str]): List of log lines to parse.

    Returns:
        list[CompilationTask]: List of extracted compilation tasks.
    """
    tasks = []
    compile_regex = re.compile(
        r"\[(\d{2}:\d{2}:\d{2}\.\d{3})\] \[(\d+)\] \[D\] Compiling (.*?)\s+([^:]+:[^:]+:[0-9a-fA-F]+)\s+to\s+(.*)$"
    )
    compiled_shader_regex = re.compile(
        r"\[(\d{2}:\d{2}:\d{2}\.\d{3})\] \[(\d+)\] \[D\] Compiled shader ([^:]+:[^:]+:[0-9a-fA-F]+)"
    )
    completed_regex = re.compile(
        r"\[(\d{2}:\d{2}:\d{2}\.\d{3})\] \[(\d+)\] \[D\] Adding Completed shader to map: ([^:]+:[^:]+:[0-9a-fA-F]+)(?::.*)?$"
    )

    for line in lines:
        compile_match = compile_regex.match(line)
        if compile_match:
            timestamp, process_id, file_path, entry_point, compile_args = compile_match.groups()
            defines = re.findall(r"\S+=[\w\d]+|\S+", compile_args.strip())
            tasks.append(
                CompilationTask(
                    process_id=process_id,
                    entry_point=entry_point,
                    file_path=file_path,
                    defines=defines,
                    start_time=parse_timestamp(line),
                )
            )
            continue

        compiled_match = compiled_shader_regex.match(line)
        if compiled_match:
            timestamp, process_id, entry_point = compiled_match.groups()
            for task in reversed(tasks):
                if task.process_id == process_id and task.entry_point == entry_point and task.end_time is None:
                    task.end_time = parse_timestamp(line)
                    break
            continue

        completed_match = completed_regex.match(line)
        if completed_match:
            timestamp, process_id, entry_point = completed_match.groups()
            for task in reversed(tasks):
                if task.process_id == process_id and task.entry_point == entry_point and task.end_time is None:
                    task.end_time = parse_timestamp(line)
                    break
    return tasks


def populate_configs(tasks: list[CompilationTask], shader_configs: dict) -> dict:
    """Populate shader configurations from compilation tasks.

    Args:
        tasks (list[CompilationTask]): List of compilation tasks.
        shader_configs (dict): Dictionary to store shader configurations.

    Returns:
        dict: Updated shader configurations.
    """
    total_lines = len(tasks)
    with tqdm(total=total_lines, desc="Parsing compiling lines", unit="line") as pbar:
        for task in tasks:
            file_path = normalize_path(task.file_path)
            file_name = file_path
            entry_point = task.entry_point
            shader_type = get_shader_type_from_entry(entry_point)
            defines = task.defines

            if file_name not in shader_configs:
                shader_configs[file_name] = {
                    "PSHADER": [],
                    "VSHADER": [],
                    "CSHADER": [],
                }
            config = {"entry": entry_point, "defines": defines}
            existing_configs = [c for c in shader_configs[file_name][shader_type] if c["entry"] == config["entry"]]
            if existing_configs:
                existing = existing_configs[0]
                if set(existing["defines"]) != set(config["defines"]):
                    print(
                        f"Warning: Updating defines for {file_name} {config['entry']} ({shader_type}): {existing['defines']} -> {config['defines']}"
                    )
                    existing["defines"] = config["defines"]
            else:
                shader_configs[file_name][shader_type].append(config)
            pbar.update(1)
    return shader_configs


def collect_warnings_and_errors(
    lines: list[str],
    tasks: list[CompilationTask],
    warnings: dict,
    errors: dict,
    total_logs: int,
) -> tuple[dict, dict]:
    """Collect warnings and errors from log lines.

    Args:
        lines (list[str]): List of log lines.
        tasks (list[CompilationTask]): List of compilation tasks.
        warnings (dict): Dictionary to store warnings.
        errors (dict): Dictionary to store errors.
        total_logs (int): Total number of log blocks to process.

    Returns:
        tuple[dict, dict]: Updated warnings and errors dictionaries.
    """
    warning_entry_regex = re.compile(r"^(.*?)\((\d+(?:,\d+(?:-\d+)?|\:\d+)?)\): warning (\w+): (.+)$")
    error_e_regex = re.compile(
        r"\[\d{2}:\d{2}:\d{2}\.\d{3}\] \[(\d+)\] \[E\] Failed to compile Pixel shader ([^:]+::[0-9a-fA-F]+):\n(.*?)\((\d+(?:,\d+(?:-\d+)?))\): error (\w+): (.+)$",
        re.DOTALL,
    )
    error_w_regex = re.compile(
        r"\[\d{2}:\d{2}:\d{2}\.\d{3}\] \[(\d+)\] \[W\] Shader compilation failed:\n(.*?):(\d+(?::\d+))\: (\w+): (.+)$",
        re.DOTALL,
    )
    completed_regex = re.compile(
        r"\[(\d{2}:\d{2}:\d{2}\.\d{3})\] \[(\d+)\] \[D\] Adding Completed shader to map: ([^:]+:[^:]+:[0-9a-fA-F]+)(?::.*)?$"
    )

    with tqdm(total=total_logs, desc="Parsing logs (warnings/errors)", unit="block") as pbar:
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            shader_log_match = re.match(r"\[(\d{2}:\d{2}:\d{2}\.\d{3})\] \[(\d+)\] \[D\] Shader logs:", line)
            if shader_log_match:
                timestamp, current_process_id = shader_log_match.groups()
                current_time = parse_timestamp(line)
                current_warnings = []
                pbar.update(1)
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if not next_line or next_line.startswith("["):
                        break
                    warning_match = warning_entry_regex.match(next_line)
                    if warning_match:
                        file_path, line_info, warning_code, warning_msg = warning_match.groups()
                        norm_file_path = normalize_path(file_path)
                        location = f"{norm_file_path}:{line_info}"
                        current_warnings.append((warning_code, warning_msg, location))
                    i += 1

                for task in reversed(tasks):
                    if (
                        task.process_id == current_process_id
                        and task.start_time <= current_time
                        and (task.end_time is None or task.end_time >= current_time)
                    ):
                        entry_point = task.entry_point
                        for code, message, location in current_warnings:
                            warning_key = f"{code}:{message}".lower()
                            if warning_key not in warnings:
                                warnings[warning_key] = {
                                    "code": code,
                                    "message": message,
                                    "instances": {},
                                }
                            location_lower = location.lower()
                            if location_lower not in warnings[warning_key]["instances"]:
                                warnings[warning_key]["instances"][location_lower] = {"entries": []}
                            if entry_point not in warnings[warning_key]["instances"][location_lower]["entries"]:
                                warnings[warning_key]["instances"][location_lower]["entries"].append(entry_point)
                        break
                continue

            match = error_e_regex.search(line)
            if match:
                process_id = match.group(1)
                entry_point = match.group(2).replace("::", ":")
                file_path = match.group(3)
                line_info = match.group(4)
                error_code = match.group(5)
                error_msg = match.group(6)
                for task in tasks:
                    if task.process_id == process_id and task.entry_point == entry_point:
                        file_name = normalize_path(task.file_path)
                        key = f"{file_name}:{entry_point}"
                        if key not in errors:
                            errors[key] = []
                        norm_file_path = normalize_path(file_path)
                        error_text = f"{error_code}: {error_msg} ({norm_file_path}:{line_info})"
                        if error_text not in errors[key]:
                            errors[key].append(error_text)
                        break
                pbar.update(1)

            match = error_w_regex.search(line)
            if match:
                process_id = match.group(1)
                file_path = match.group(2)
                line_info = match.group(3)
                error_code = match.group(4)
                error_msg = match.group(5)
                for task in tasks:
                    if task.process_id == process_id:
                        file_name = normalize_path(task.file_path)
                        entry_point = task.entry_point
                        key = f"{file_name}:{entry_point}"
                        if key not in errors:
                            errors[key] = []
                        norm_file_path = normalize_path(file_path)
                        error_text = f"{error_code}: {error_msg} ({norm_file_path}:{line_info})"
                        if error_text not in errors[key]:
                            errors[key].append(error_text)
                        break
                pbar.update(1)

            match = completed_regex.search(line)
            if match:
                pbar.update(1)
            i += 1

    return warnings, errors


def parse_log(
    log_file: str,
    update_configs: dict | None = None,
    update_warnings: dict | None = None,
    update_errors: dict | None = None,
) -> tuple[dict, dict, dict]:
    """Parse a CommunityShaders log file to extract shader configurations, warnings, and errors.

    Args:
        log_file (str): Path to the log file.
        update_configs (dict | None): Existing shader configurations to update (optional).
        update_warnings (dict | None): Existing warnings to update (optional).
        update_errors (dict | None): Existing errors to update (optional).

    Returns:
        tuple[dict, dict, dict]: Shader configurations, warnings, and errors.
    """
    shader_configs = update_configs or {}
    warnings = update_warnings or {}
    errors = update_errors or {}

    with open(log_file, encoding="utf-8") as f:
        lines = f.readlines()

    tasks = collect_tasks(lines)
    shader_configs = populate_configs(tasks, shader_configs)
    total_logs = count_log_blocks(log_file)
    warnings, errors = collect_warnings_and_errors(lines, tasks, warnings, errors, total_logs)

    return shader_configs, warnings, errors


def compute_common_defines(shader_configs: dict) -> tuple[list, dict, dict]:
    """Compute common defines across shaders, types, and files.

    Args:
        shader_configs (dict): Shader configurations.
    Returns:
        tuple[list, dict, dict]: Global common defines, type-specific common defines, and file-type-specific common defines.
    """
    all_defines = [
        config["defines"]
        for file_configs in shader_configs.values()
        for configs in file_configs.values()
        for config in configs
    ]
    global_common = list(set.intersection(*[set(d) for d in all_defines])) if all_defines else []
    defines_by_type = {"PSHADER": [], "VSHADER": [], "CSHADER": []}
    for file_configs in shader_configs.values():
        for shader_type, configs in file_configs.items():
            for config in configs:
                defines_by_type[shader_type].append(config["defines"])
    type_common = {}
    for shader_type, defines_list in defines_by_type.items():
        type_common[shader_type] = list(set.intersection(*[set(d) for d in defines_list])) if defines_list else []
        type_common[shader_type] = [d for d in type_common[shader_type] if d not in global_common]
    file_type_common = {}
    for file_name, file_configs in shader_configs.items():
        file_type_common[file_name] = {"PSHADER": [], "VSHADER": [], "CSHADER": []}
        for shader_type, configs in file_configs.items():
            file_type_common[file_name][shader_type] = (
                list(set.intersection(*[set(c["defines"]) for c in configs])) if configs else []
            )
            file_type_common[file_name][shader_type] = [
                d
                for d in file_type_common[file_name][shader_type]
                if d not in global_common and d not in type_common[shader_type]
            ]
    return global_common, type_common, file_type_common


def generate_yaml_data(shader_configs: dict, warnings: dict, errors: dict) -> dict:
    """Generate YAML data structure from shader configurations, warnings, and errors.

    Args:
        shader_configs (dict): Shader configurations.
        warnings (dict): Compilation warnings.
        errors (dict): Compilation errors.
    Returns:
        dict: YAML-compatible data structure.
    """
    global_common, type_common, file_type_common = compute_common_defines(shader_configs)
    yaml_data = {
        "common_defines": global_common,
        "common_pshader_defines": type_common["PSHADER"],
        "common_vshader_defines": type_common["VSHADER"],
        "common_cshader_defines": type_common["CSHADER"],
        "file_common_defines": file_type_common,
        "warnings": warnings,
        "errors": errors,
        "shaders": [],
    }
    for file_name, file_configs in shader_configs.items():
        shader_entry = {"file": file_name, "configs": {}}
        for shader_type, configs in file_configs.items():
            if configs:
                common_defines = []
                for defines in [
                    global_common,
                    type_common[shader_type],
                    file_type_common[file_name][shader_type],
                ]:
                    if isinstance(defines, list):
                        common_defines.extend(defines)
                    elif isinstance(defines, (set, tuple)):
                        common_defines.extend(list(defines))
                shader_entry["configs"][shader_type] = {
                    "common_defines": common_defines,
                    "entries": [
                        {
                            "entry": config["entry"],
                            "defines": [
                                d
                                for d in config["defines"]
                                if d not in global_common
                                and d not in type_common[shader_type]
                                and d not in file_type_common[file_name][shader_type]
                            ],
                        }
                        for config in configs
                    ],
                }
        if shader_entry["configs"]:
            yaml_data["shaders"].append(shader_entry)
    return yaml_data


def make_hashable(obj):
    if isinstance(obj, list):
        return tuple(make_hashable(x) for x in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    else:
        return obj


def deduplicate_lists(obj, memo=None):
    """Recursively deduplicate all equal lists in a data structure, replacing them with a shared object."""
    if memo is None:
        memo = {}
    if isinstance(obj, list):
        key = make_hashable(obj)
        if key in memo:
            return memo[key]
        # Deduplicate elements recursively
        deduped = [deduplicate_lists(x, memo) if isinstance(x, (list, dict)) else x for x in obj]
        memo[key] = deduped
        return deduped
    elif isinstance(obj, dict):
        return {k: deduplicate_lists(v, memo) if isinstance(v, (list, dict)) else v for k, v in obj.items()}
    else:
        return obj


def save_yaml(yaml_data: dict, output_file: str) -> None:
    """Save the YAML data to a file, using anchors for repeated lists to reduce repetition.
    Automatically deduplicates all equal lists so anchors are maximally used.
    Args:
        yaml_data (dict): YAML data to save.
        output_file (str): Path to the output YAML file.
    """
    from collections import defaultdict

    class AnchorDumper(yaml.SafeDumper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._list_id_to_anchor = {}
            self._anchor_counts = defaultdict(int)

        def ignore_aliases(self, data):
            # Only use anchors for lists (not dicts or scalars)
            return not isinstance(data, list)

        def represent_sequence(self, tag, sequence, flow_style=None):
            # Use anchors for repeated lists
            list_id = id(sequence)
            if list_id in self._list_id_to_anchor:
                anchor = self._list_id_to_anchor[list_id]
            else:
                anchor = None
                # Only anchor if this list is referenced more than once
                for _k, v in self.represented_objects.items():
                    if v is sequence:
                        self._anchor_counts[list_id] += 1
                        if self._anchor_counts[list_id] == 2:
                            anchor = f"id{list_id}"
                            self._list_id_to_anchor[list_id] = anchor
                        break
            node = super().represent_sequence(tag, sequence, flow_style)
            if anchor:
                node.anchor = anchor  # type: ignore[attr-defined]
            return node

    deduped = deduplicate_lists(yaml_data)
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(
            deduped,
            f,
            Dumper=AnchorDumper,
            sort_keys=False,
            allow_unicode=True,
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the shader defines generator.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    is_gui_mode = HAS_GOOEY and ("--gui" in sys.argv or "-g" in sys.argv)
    parser_class = GooeyParser if is_gui_mode else argparse.ArgumentParser
    parser = parser_class(description="Generate shader defines from CommunityShaders log.")
    parser.add_argument(
        "--log",
        default="CommunityShaders.log",
        help="Path to CommunityShaders log file",
    )
    parser.add_argument(
        "--output",
        default="shader_defines.yaml",
        help="Output YAML file for shader defines",
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    if not is_gui_mode:
        parser.add_argument("-g", "--gui", action="store_true", help="Run with GUI")
    return parser.parse_args()


def main() -> int:
    """Main entry point for generating shader defines from a log file.

    Returns:
        int: Exit code (0 for success, 1 for errors).
    """
    args = parse_arguments()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    if not os.path.exists(args.log):
        logging.error(f"Log file not found: {args.log}")
        return 1

    start_time = time.time()
    shader_configs, warnings, errors = parse_log(args.log)
    yaml_data = generate_yaml_data(shader_configs, warnings, errors)
    save_yaml(yaml_data, args.output)

    total_variants = sum(len(configs) for file_configs in shader_configs.values() for configs in file_configs.values())
    logging.info(
        f"Generated {args.output} in {time.time() - start_time:.2f} seconds "
        f"with {len(yaml_data['shaders'])} shaders, {total_variants} variants, "
        f"{len(warnings)} unique warnings, and {sum(len(e) for e in errors.values())} errors"
    )
    return 0


if __name__ == "__main__":
    if HAS_GOOEY and ("--gui" in sys.argv or "-g" in sys.argv):

        @Gooey(
            progress_regex=r"(\d+)/(\d+)",
            progress_expr="x[0] / x[1] * 100",
            program_name="Shader Defines Generator",
            default_size=(800, 600),
        )
        def gooey_main():
            return main()

        sys.exit(gooey_main())
    else:
        sys.exit(main())
