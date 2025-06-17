"""Compile HLSL shaders using fxc.exe and generate a warning/error report.

This script compiles HLSL shaders specified in a YAML configuration file using Microsoft's `fxc.exe` compiler.
It supports parallel compilation, system-adaptive job management, and detailed logging of warnings and errors.
The output includes compiled shader objects and a `new_issues.log` file summarizing new warnings and errors.

Example:
    ```bash
    python compile_shaders.py --config shader_defines.yaml --shader-dir src --output-dir build
    ```

Dependencies:
    - `yaml`: For parsing configuration files.
    - `tqdm`: For progress bars.
    - `psutil` (optional): For system resource monitoring.
    - `gooey` (optional): For GUI support.
"""

import argparse
import concurrent.futures
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from types import FrameType
from typing import Any

import yaml
from tqdm import tqdm

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

running_processes = set()
running_processes_lock = threading.Lock()
stop_event = threading.Event()

WARNING_REGEX = r"^(.*?)\((\d+(?:,\d+(?:-\d+)?|\:\d+)?)\): warning (\w+): (.+)$"
ERROR_REGEX = r"^(.*?)\((\d+(?:,\d+(?:-\d+)?|\:\d+)?)\): error (\w+): (.+)$"

try:
    from gooey import Gooey, GooeyParser

    HAS_GOOEY = True
except ImportError:
    Gooey = lambda x: x
    HAS_GOOEY = False


def normalize_path(file_path: str) -> str:
    """Normalize a file path by standardizing separators and extracting relative path.

    Args:
        file_path (str): The file path to normalize.

    Returns:
        str: The normalized file path, relative to the Shaders directory if present.

    Example:
        >>> normalize_path("C:/Projects/Shaders/src/test.hlsl")
        'src/test.hlsl'
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


def flatten_defines(defines: list) -> list[str]:
    """Flatten a list of defines, removing duplicates while preserving order.

    Args:
        defines (list): List of defines, possibly nested.

    Returns:
        list[str]: Flattened list of unique defines.

    Example:
        >>> flatten_defines([["A=1", "B"], "C"])
        ['A=1', 'B', 'C']
    """
    flat = []
    for d in defines:
        if isinstance(d, list):
            flat.extend(flatten_defines(d))
        else:
            flat.append(d)
    seen = set()
    return [d for d in flat if not (d in seen or seen.add(d))]


def handle_termination(signum: int | None = None, frame: FrameType | None = None) -> None:
    """Handle termination signals by gracefully shutting down subprocesses.

    Args:
        signum (int | None): Signal number (e.g., SIGINT).
        frame (FrameType | None): Current stack frame.
    """
    logging.warning("Termination signal received. Shutting down gracefully...")
    stop_event.set()
    with running_processes_lock:
        for proc in list(running_processes):
            try:
                logging.info(f"Terminating subprocess: {proc.pid}")
                proc.terminate()
            except Exception:
                logging.exception(f"Failed to terminate process {proc.pid}")
    logging.info("Exiting...")


def validate_shader_inputs(
    fxc_path: str, shader_file: str, output_dir: str, defines: list[str], shader_dir: str
) -> str | None:
    """Validate inputs for shader compilation.

    Args:
        fxc_path (str): Path to fxc.exe.
        shader_file (str): Path to the shader file.
        output_dir (str): Output directory for compiled shaders.
        defines (list[str]): List of preprocessor defines.
        shader_dir (str): Directory containing shader files.

    Returns:
        str | None: Error message if validation fails, else None.

    Example:
        >>> validate_shader_inputs("fxc.exe", "test.hlsl", "build", [], "src")        None
    """
    fxc_executable = shutil.which(fxc_path)
    if not fxc_executable:
        return "fxc.exe not found in PATH or specified path"
    shader_file_path = os.path.join(shader_dir, os.path.basename(shader_file))
    if not os.path.isfile(shader_file_path) or not shader_file.endswith((".hlsl", ".hlsli")):
        return f"Invalid shader file: {shader_file}"
    abs_output_dir = os.path.abspath(output_dir)
    if not os.path.isdir(abs_output_dir):
        return f"Invalid output directory: {output_dir}"
    valid_define_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(?:=[\w\d]+)?$")
    invalid_defines = [d for d in defines if not valid_define_pattern.match(d)]
    if invalid_defines:
        return f"Invalid defines: {invalid_defines}"
    return None


def compile_shader(
    fxc_path: str,
    shader_file: str,
    shader_type: str,
    entry: str,
    defines: list[str],
    output_dir: str,
    shader_dir: str,
    debug: bool = False,
    strip_debug_defines: bool = False,
    optimization_level: str = "1",
    force_partial_precision: bool = False,
    debug_defines: set = None,
) -> dict[str, Any]:
    """Compile a shader using fxc.exe.

    Args:
        fxc_path (str): Path to fxc.exe.
        shader_file (str): Path to the shader file.
        shader_type (str): Type of shader (VSHADER, PSHADER, CSHADER).
        entry (str): Entry point for the shader.
        defines (list[str]): Preprocessor defines.
        output_dir (str): Output directory for compiled shaders.
        shader_dir (str): Directory containing shader files.
        debug (bool): Enable debug logging.
        strip_debug_defines (bool): Strip debug-related defines.
        optimization_level (str): Optimization level (0-3).
        force_partial_precision (bool): Force 16-bit precision.
        debug_defines (set): Set of debug defines to strip.

    Returns:
        dict[str, any]: Compilation result with file, entry, type, log, success, and command.

    Example:
        >>> compile_shader("fxc.exe", "test.hlsl", "PSHADER", "main:1234", [], "build", "src")
        {'file': 'test.hlsl', 'entry': 'main:1234', 'type': 'PSHADER', 'log': '...', 'success': True, 'cmd': [...]}
    """
    if stop_event.is_set():
        return {
            "file": shader_file,
            "entry": entry,
            "type": shader_type,
            "log": "Compilation aborted.",
            "success": False,
            "cmd": [],
        }

    validation_error = validate_shader_inputs(fxc_path, shader_file, output_dir, defines, shader_dir)
    if validation_error:
        logging.error(validation_error)
        return {
            "file": shader_file,
            "entry": entry,
            "type": shader_type,
            "log": validation_error,
            "success": False,
            "cmd": [],
        }

    entry_name = "main"
    shader_id = entry.split(":")[-1]
    shader_basename = os.path.basename(shader_file)
    shader_name_no_ext, _ = os.path.splitext(shader_basename)
    output_subdir = os.path.join(output_dir, shader_name_no_ext)

    ext_map = {"VSHADER": ".vso", "PSHADER": ".pso", "CSHADER": ".cso"}
    ext = ext_map.get(shader_type.upper())
    if ext is None:
        return {
            "file": shader_file,
            "entry": entry,
            "type": shader_type,
            "log": f"Unsupported shader type: {shader_type}",
            "success": False,
            "cmd": [],
        }

    os.makedirs(output_subdir, exist_ok=True)
    output_path = os.path.join(output_subdir, shader_id + ext)

    shader_model_map = {"VSHADER": "vs_5_0", "PSHADER": "ps_5_0", "CSHADER": "cs_5_0"}
    model = shader_model_map[shader_type.upper()]

    shader_file_path = os.path.join(shader_dir, shader_basename)
    if not os.path.exists(shader_file_path):
        error_msg = f"Shader file not found in {shader_dir}: {shader_basename}"
        logging.error(error_msg)
        return {
            "file": shader_file,
            "entry": entry,
            "type": shader_type,
            "log": error_msg,
            "success": False,
            "cmd": [],
        }

    if debug_defines is None:
        debug_defines = {"DEBUG", "_DEBUG", "D3D_DEBUG_INFO", "D3DCOMPILE_DEBUG", "D3DCOMPILE_SKIP_OPTIMIZATION"}
    if strip_debug_defines:
        defines = [d for d in defines if d.split("=")[0].upper() not in debug_defines]
        defines.append("D3DCOMPILE_AVOID_FLOW_CONTROL")
        logging.debug(
            f"Stripped debug defines and added D3DCOMPILE_AVOID_FLOW_CONTROL for {shader_file}:{entry}. Defines: {defines}"
        )

    cmd = [
        fxc_path,
        "/T",
        model,
        "/E",
        entry_name,
        "/Fo",
        output_path,
        "/O" + optimization_level,
        shader_basename,
    ]
    if force_partial_precision:
        cmd.append("/Gfp")
    for d in defines:
        cmd.extend(["/D", d])
    cmd.extend(["/I", os.path.abspath(shader_dir)])

    log = ""
    success = False
    process = None

    logging.debug(f"Executing command: {' '.join(cmd)}")
    # Defines are sanitized in validate_shader_inputs to prevent injection
    try:
        process = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.abspath(shader_dir),
        )
        with running_processes_lock:
            running_processes.add(process)
        stdout, stderr = process.communicate()
        log = stdout + stderr
        success = process.returncode == 0
    except Exception as e:
        log = str(e)
        success = False
    finally:
        with running_processes_lock:
            if process in running_processes:
                running_processes.remove(process)

    if debug:
        logging.debug(f"Command {'failed' if not success else 'succeeded'}: {' '.join(cmd)}")
        logging.debug(f"Output:\n{log}")

    return {
        "file": shader_file,
        "entry": entry,
        "type": shader_type,
        "log": log,
        "success": success,
        "cmd": cmd,
    }


def parse_shader_configs(config_file: str) -> list[tuple]:
    """Parse shader configurations from a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        list[tuple]: List of tuples containing (file_name, shader_type, entry_name, defines).

    Example:
        >>> parse_shader_configs("shader_defines.yaml")
        [('test.hlsl', 'PSHADER', 'main:1234', ['A=1'])]
    """
    with open(config_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "shaders" not in data:
        logging.error("Invalid shader configuration: missing 'shaders' section")
        return []

    tasks = []
    for shader in data["shaders"]:
        file_name = shader["file"]
        if "configs" not in shader:
            logging.warning(f"Skipping shader {file_name}: missing 'configs' section")
            continue
        for shader_type, config in shader["configs"].items():
            if "entries" not in config:
                logging.warning(f"Skipping {shader_type} in {file_name}: missing 'entries'")
                continue
            common_defines = config.get("common_defines", [])
            if not isinstance(common_defines, list):
                logging.warning(f"common_defines is not a list in shader config: {file_name}")
                common_defines = []
            for entry in config["entries"]:
                entry_name = entry["entry"]
                entry_defines = entry.get("defines", [])
                if not isinstance(entry_defines, list):
                    logging.warning(f"entry defines is not a list in shader config: {file_name}")
                    entry_defines = []
                defines = flatten_defines(common_defines + entry_defines)
                tasks.append((file_name, shader_type, entry_name, defines))

    return tasks


def load_baseline_warnings(config_file: str) -> dict:
    """Load baseline warnings from a YAML configuration file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Dictionary of baseline warnings.

    Example:
        >>> load_baseline_warnings("shader_defines.yaml")
        {'x3206:implicit truncation': {'code': 'X3206', 'message': 'implicit truncation', 'instances': {...}}}
    """
    baseline_warnings = {}
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
                if config_data and "warnings" in config_data:
                    baseline_warnings = {k.lower(): v for k, v in config_data["warnings"].items()}
        except Exception as e:
            logging.warning(f"Failed to load baseline warnings from {config_file}: {e}")
    return baseline_warnings


def build_defines_lookup(config_file: str) -> dict:
    """Build a lookup table for shader defines from a YAML configuration.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Lookup table mapping shader keys to (shader_type, defines).

    Example:
        >>> build_defines_lookup("shader_defines.yaml")
        {'test.hlsl:main:1234': ('PSHADER', ['A=1'])}
    """
    defines_lookup = {}
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
                if config_data and "shaders" in config_data:
                    for shader in config_data["shaders"]:
                        file_name = shader["file"]
                        for shader_type, config in shader.get("configs", {}).items():
                            for entry in config.get("entries", []):
                                entry_name = entry["entry"]
                                defines = flatten_defines(config.get("common_defines", []) + entry.get("defines", []))
                                defines_lookup[f"{file_name}:{entry_name}".lower()] = (
                                    shader_type,
                                    defines,
                                )
        except Exception as e:
            logging.warning(f"Failed to load shader configs from {config_file}: {e}")
    return defines_lookup


class IssueHandler:
    """Base class for handling compilation issues (warnings and errors)."""
    
    def __init__(self, result: dict):
        self.result = result
        self.file_name = os.path.basename(result['file'])
        self.shader_key = f"{self.file_name}:{result['entry']}"
        self.shader_key_lower = self.shader_key.lower()
        self.context = {
            "shader_type": result["type"],
            "entry_point": result["entry"]
        }

    def normalize_location(self, file_path: str, line_info: str) -> str:
        """Normalize file path and create location string."""
        norm_file_path = normalize_path(file_path)
        return f"{norm_file_path}:{line_info}"

    def create_issue_data(self, code: str, message: str, location: str) -> dict:
        """Create a standardized issue data structure."""
        return {
            "code": code,
            "message": message,
            "location": location,
            "context": self.context.copy()
        }

    def add_to_instances(self, instances: dict, location: str, issue_data: dict) -> None:
        """Add an issue to the instances dictionary."""
        if location not in instances:
            instances[location] = []
        if not any(i["code"] == issue_data["code"] and i["message"] == issue_data["message"] 
                  for i in instances[location]):
            instances[location].append(issue_data)


class WarningHandler(IssueHandler):
    """Handler for compilation warnings."""
    
    def process(self, line: str, baseline_warnings: dict, suppress_warnings: list[str],
                all_warnings: dict, new_warnings_dict: dict, suppressed_count: int) -> tuple[dict, dict, int]:
        """Process a warning line."""
        warning_match = re.match(WARNING_REGEX, line)
        if not warning_match:
            return all_warnings, new_warnings_dict, suppressed_count

        file_path, line_info, warning_code, warning_msg = warning_match.groups()
        location = self.normalize_location(file_path, line_info)
        warning_key = f"{warning_code}:{warning_msg}".lower()

        if warning_code.lower() in suppress_warnings:
            suppressed_count += 1
            logging.debug(f"Suppressed warning: {warning_code} at {location}")
            return all_warnings, new_warnings_dict, suppressed_count

        # Create context-aware warning key
        context_warning_key = f"{warning_key}:{self.context['shader_type']}:{self.context['entry_point']}"

        if warning_key not in all_warnings:
            all_warnings[warning_key] = {
                "code": warning_code,
                "message": warning_msg,
                "instances": {}
            }

        if location not in all_warnings[warning_key]["instances"]:
            all_warnings[warning_key]["instances"][location] = {"entries": []}
        if self.context["entry_point"] not in all_warnings[warning_key]["instances"][location]["entries"]:
            all_warnings[warning_key]["instances"][location]["entries"].append(self.context["entry_point"])

        # Check if this is a new warning
        is_new_warning = True
        if warning_key in baseline_warnings:
            baseline_data = baseline_warnings[warning_key]
            baseline_count = sum(
                1 for baseline_loc, baseline_instance in baseline_data["instances"].items()
                if self.context["entry_point"].lower() in [e.lower() for e in baseline_instance.get("entries", [])]
            )
            if baseline_count > 0:
                current_count = sum(
                    1 for loc, inst in all_warnings[warning_key]["instances"].items()
                    if self.context["entry_point"].lower() in [e.lower() for e in inst.get("entries", [])]
                )
                is_new_warning = current_count > baseline_count

        if is_new_warning:
            warning_str = f"{self.shader_key}:{warning_code}: {warning_msg} ({location})"
            warning_id = f"{context_warning_key}:{location.lower()}"
            if warning_id not in new_warnings_dict:
                new_warnings_dict[warning_id] = {
                    "warning_key": warning_key,
                    "location": location,
                    "code": warning_code,
                    "message": warning_msg,
                    "example": warning_str,
                    "entries": [],
                    "instances": {}
                }
            if self.shader_key not in new_warnings_dict[warning_id]["entries"]:
                new_warnings_dict[warning_id]["entries"].append(self.shader_key)

            if location not in new_warnings_dict[warning_id]["instances"]:
                new_warnings_dict[warning_id]["instances"][location] = {"entries": []}
            if self.shader_key not in new_warnings_dict[warning_id]["instances"][location]["entries"]:
                new_warnings_dict[warning_id]["instances"][location]["entries"].append(self.shader_key)

        return all_warnings, new_warnings_dict, suppressed_count


class ErrorHandler(IssueHandler):
    """Handler for compilation errors."""
    
    def process(self, line: str, errors: dict) -> dict:
        """Process an error line."""
        error_match = re.match(ERROR_REGEX, line)
        if not error_match:
            return errors

        file_path, line_info, error_code, error_msg = error_match.groups()
        location = self.normalize_location(file_path, line_info)

        if self.shader_key_lower not in errors:
            errors[self.shader_key_lower] = {
                "instances": {},
                "entries": [],
                "type": self.context["shader_type"]
            }

        error_data = self.create_issue_data(error_code, error_msg, location)
        self.add_to_instances(errors[self.shader_key_lower]["instances"], location, error_data)
        
        if self.context["entry_point"] not in errors[self.shader_key_lower]["entries"]:
            errors[self.shader_key_lower]["entries"].append(self.context["entry_point"])

        return errors


def process_single_warning(
    line: str,
    result: dict,
    baseline_warnings: dict,
    suppress_warnings: list[str],
    all_warnings: dict,
    new_warnings_dict: dict,
    suppressed_warnings_count: int,
) -> tuple[dict, dict, int]:
    """Process a single warning line from compilation output."""
    handler = WarningHandler(result)
    return handler.process(line, baseline_warnings, suppress_warnings, all_warnings, new_warnings_dict, suppressed_warnings_count)


def process_single_error(line: str, result: dict, errors: dict) -> dict:
    """Process a single error line from compilation output."""
    handler = ErrorHandler(result)
    return handler.process(line, errors)


def process_warnings_and_errors(
    results: list[dict],
    baseline_warnings: dict,
    suppress_warnings: list[str],
    defines_lookup: dict,
) -> tuple[list[dict], dict, dict, int]:
    """Process warnings and errors from shader compilation results.

    Args:
        results (list[dict]): List of compilation results.
        baseline_warnings (dict): Baseline warnings for comparison.
        suppress_warnings (list[str]): Warning codes to suppress.
        defines_lookup (dict): Lookup table for shader defines.

    Returns:
        tuple[list[dict], dict, dict, int]: New warnings, all warnings, errors, and suppressed warning count.
    """
    all_warnings = {}
    errors = {}
    new_warnings_dict = {}
    suppressed_warnings_count = 0
    suppress_warnings = [code.lower() for code in (suppress_warnings or [])]

    for result in results:
        if not result.get("log"):
            continue
        log_lines = result["log"].splitlines()
        for line in log_lines:
            all_warnings, new_warnings_dict, suppressed_warnings_count = process_single_warning(
                line,
                result,
                baseline_warnings,
                suppress_warnings,
                all_warnings,
                new_warnings_dict,
                suppressed_warnings_count,
            )
            errors = process_single_error(line, result, errors)

    return (
        list(new_warnings_dict.values()),
        all_warnings,
        errors,
        suppressed_warnings_count,
    )


def log_new_issues(
    new_warnings: list[dict], errors: dict, results: list[dict], output_dir: str, defines_lookup: dict
) -> None:
    """Log new warnings and errors to a unified file.

    Args:
        new_warnings (list[dict]): List of new warnings.
        errors (dict): Dictionary of compilation errors.
        results (list[dict]): Compilation results.
        output_dir (str): Directory to save the log file.
        defines_lookup (dict): Lookup table for shader defines.
    """

    issue_logger = logging.getLogger("new_issues")
    issue_logger.setLevel(logging.INFO)
    issue_handler = logging.FileHandler(os.path.join(output_dir, "new_issues.log"), mode="w", encoding="utf-8")
    issue_handler.setFormatter(logging.Formatter("%(message)s"))
    issue_logger.addHandler(issue_handler)
    issue_logger.propagate = False

    # Calculate totals
    total_warnings = sum(get_instance_count(w) for w in new_warnings)
    total_warning_entries = sum(
        sum(len(loc_data["entries"]) for loc_data in w["instances"].values()) for w in new_warnings
    )
    total_errors = sum(len(e) for e in errors.values())

    # Header with summary
    issue_logger.info("=" * 80)
    issue_logger.info("NEW SHADER COMPILATION ISSUES DETECTED")
    issue_logger.info("=" * 80)
    issue_logger.info(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    issue_logger.info(
        f"New warnings: {len(new_warnings)} types, {total_warnings} instances, {total_warning_entries} shader combinations"
    )
    issue_logger.info(f"Compilation errors: {total_errors} total errors")
    issue_logger.info("=" * 80)
    issue_logger.info("")

    # Log compilation errors first (they're more critical)
    if errors:
        issue_logger.info("COMPILATION ERRORS")
        issue_logger.info("=" * 60)
        issue_logger.info("")

        for shader_key, error_data in errors.items():
            shader_file = shader_key.split(":")[0]
            entry_point = ":".join(shader_key.split(":")[1:]) if ":" in shader_key else "unknown"
            shader_type = error_data.get("type", "unknown")

            issue_logger.info(f"ERROR in {shader_file} (entry: {entry_point}, type: {shader_type}):")
            issue_logger.info("-" * 40)

            # Group errors by location for better context
            for location, error_instances in error_data["instances"].items():
                issue_logger.info(f"  Location: {location}")
                for error in error_instances:
                    issue_logger.info(f"    Error Code: {error['code']}")
                    issue_logger.info(f"    Message: {error['message']}")
                    if error.get("context"):
                        issue_logger.info(f"    Context: {error['context']['shader_type']} - {error['context']['entry_point']}")
                    issue_logger.info("")

            issue_logger.info("=" * 40)
            issue_logger.info("")

    # Log new warnings
    if new_warnings:
        issue_logger.info("NEW WARNINGS")
        issue_logger.info("=" * 60)
        issue_logger.info("")

        # Sort warnings by total entry count (impact) - highest first
        sorted_warnings = sorted(
            new_warnings,
            key=lambda w: sum(len(loc_data["entries"]) for loc_data in w["instances"].values()),
            reverse=True,
        )

        for i, warning in enumerate(sorted_warnings, 1):
            # Calculate total entries for this warning
            warning_entry_count = sum(len(loc_data["entries"]) for loc_data in warning["instances"].values())

            issue_logger.info(f"WARNING #{i}: {warning['code']} - {warning['message']}")
            issue_logger.info(f"Affected shader combinations: {warning_entry_count}")
            issue_logger.info("-" * 60)

            # Show each location where this warning occurs
            for location, location_data in warning["instances"].items():
                entry_count = len(location_data["entries"])
                issue_logger.info(f"Location: {location} ({entry_count} combinations)")

                # Show the actual compilation output for this location
                for result in results:
                    shader_key = f"{os.path.basename(result['file'])}:{result['entry']}"
                    if shader_key in location_data["entries"]:
                        if result.get("log"):
                            log_lines = result["log"].splitlines()
                            warning_lines = [
                                line for line in log_lines if warning["code"] in line and location.split(":")[0] in line
                            ]
                            if warning_lines:
                                warning_line_index = log_lines.index(warning_lines[0])
                                context_start = max(0, warning_line_index - 2)
                                context_end = min(len(log_lines), warning_line_index + 3)
                                context = log_lines[context_start:context_end]
                                issue_logger.info("  Compiler output context:")
                                for ctx_line in context:
                                    issue_logger.info(f"    {ctx_line}")
                        break

                issue_logger.info("")

            issue_logger.info("=" * 60)
            issue_logger.info("")

    # Summary section
    issue_logger.info("SUMMARY")
    issue_logger.info("=" * 60)
    if total_errors > 0:
        issue_logger.info(f"ACTION REQUIRED: Fix {total_errors} compilation errors before proceeding.")
    if total_warnings > 0:
        issue_logger.info(
            f"RECOMMENDED: Address {total_warnings} new warnings across {total_warning_entries} shader combinations."
        )
    if total_errors == 0 and total_warnings == 0:
        issue_logger.info("No new issues detected - compilation is clean!")
    issue_logger.info("=" * 80)

    issue_handler.close()
    issue_logger.removeHandler(issue_handler)


def parse_args_for_defaults() -> dict[str, Any]:
    """Parse command-line arguments to extract default values.

    Returns:
        dict[str, any]: Dictionary of default argument values.
    """
    arg_dict = {}
    args = [arg for arg in sys.argv[1:] if arg != "--ignore-gooey"]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in [
            "--fxc",
            "--shader-dir",
            "--output-dir",
            "--config",
            "--suppress-warnings",
            "--optimization-level",
        ]:
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                arg_dict[arg.lstrip("-")] = args[i + 1]
                i += 2
            else:
                i += 1
        elif arg in ["--jobs", "--max-warnings"]:
            if i + 1 < len(args):
                try:
                    arg_dict[arg.lstrip("-")] = int(args[i + 1])
                    i += 2
                except ValueError:
                    i += 2
            else:
                i += 1
        elif arg in [
            "-d",
            "--debug",
            "-g",
            "--gui",
            "--strip-debug-defines",
            "--force-partial-precision",
        ]:
            arg_dict[arg.lstrip("-")] = True
            i += 1
        else:
            i += 1
    return arg_dict


def count_fxc_processes() -> int:
    """Count the number of running fxc.exe processes.

    Returns:
        int: Number of fxc.exe processes.
    """
    if not HAS_PSUTIL:
        return 0
    count = 0
    for proc in psutil.process_iter(["name"]):
        try:
            if proc.info["name"].lower() == "fxc.exe":
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return count


def get_system_adaptive_jobs(
    cpu_count: int, physical_cores: int | None, is_ci: bool, avg_cpu: float
) -> tuple[int, str]:
    """Determine the optimal number of parallel jobs based on system resources.

    Args:
        cpu_count (int): Total CPU cores.
        physical_cores (int | None): Physical CPU cores.
        is_ci (bool): Whether running in a CI environment.
        avg_cpu (float): Average CPU usage.

    Returns:
        tuple[int, str]: Number of jobs and reason for selection.
    """
    if is_ci:
        # In CI environments, be more aggressive with job allocation since runners are dedicated
        jobs = min(max(cpu_count - 1, 2), 4)
        reason = "auto-detected for CI environment (aggressive)"
        if HAS_PSUTIL:
            try:
                cpu_usage = psutil.cpu_percent(interval=0.5)
                mem_usage = psutil.virtual_memory().percent
                if cpu_usage > 90 or mem_usage > 95:
                    jobs = min(max(cpu_count // 2, 2), 4)
                    reason = (
                        f"auto-detected for CI, high CPU ({cpu_usage:.1f}%) or memory ({mem_usage:.1f}%) - conservative"
                    )
                # For low load, keep the aggressive setting
            except Exception as e:
                logging.debug(f"Failed to check system usage in CI: {e}")
        return jobs, reason

    max_jobs = min(physical_cores or 24, cpu_count - 2) if physical_cores else min(cpu_count - 2, 24)
    if HAS_PSUTIL:
        try:
            cpu_usage = psutil.cpu_percent(interval=0.5)
            mem_usage = psutil.virtual_memory().percent
            if cpu_usage < 70 and mem_usage < 90:
                jobs = max_jobs
                reason = f"auto-detected, low CPU ({cpu_usage:.1f}%), low memory ({mem_usage:.1f}%)"
            elif cpu_usage < 90 and mem_usage < 95:
                jobs = max(int(max_jobs * 0.85), 6)
                reason = f"auto-detected, moderate CPU ({cpu_usage:.1f}%), moderate memory ({mem_usage:.1f}%)"
            else:
                jobs = max(int(max_jobs * 0.6), 6)
                reason = f"auto-detected, high CPU ({cpu_usage:.1f}%) or memory ({mem_usage:.1f}%)"
            job_levels = [6, 8, 12, 16, 20, 24]
            jobs = min(job_levels, key=lambda x: abs(x - jobs))
        except Exception as e:
            logging.debug(f"Failed to check system usage: {e}")
            jobs = max_jobs
            reason = "auto-detected, system usage check failed"
    else:
        jobs = max_jobs
        reason = "auto-detected, no psutil available"

    return jobs, reason


def parse_arguments(default_jobs: int) -> argparse.Namespace:
    """Parse command-line arguments for the shader compiler.

    Args:
        default_jobs (int): Default number of parallel jobs.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    defaults = parse_args_for_defaults()
    is_gui_mode = HAS_GOOEY and ("--gui" in sys.argv or "-g" in sys.argv)
    parser_class = GooeyParser if is_gui_mode else argparse.ArgumentParser
    parser = parser_class(description="Compile shaders using fxc.exe.")
    parser.add_argument(
        "--fxc",
        default=defaults.get("fxc"),
        help="Path to fxc.exe (optional if it's in PATH)",
    )
    parser.add_argument(
        "--shader-dir",
        default=defaults.get("shader-dir", "build/aio/Shaders"),
        help="Directory containing shader files",
    )
    parser.add_argument(
        "--output-dir",
        default=defaults.get("output-dir", "build/ShaderCache"),
        help="Output directory for compiled shaders",
    )
    parser.add_argument(
        "--config",
        default=defaults.get("config", "shader_defines.yaml"),
        help="Shader defines YAML file",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=defaults.get("jobs", default_jobs),
        help="Number of parallel jobs (default: dynamic, based on system usage)",
    )
    parser.add_argument(
        "--max-warnings",
        type=int,
        default=defaults.get("max-warnings", 0),
        help="Warning control: positive=max new warnings, negative=must eliminate N baseline warnings, 0=no new warnings",
    )
    parser.add_argument(
        "--suppress-warnings",
        default=defaults.get("suppress-warnings", ""),
        help="Comma-separated list of warning codes to suppress (e.g., X1519,X3206)",
    )
    parser.add_argument(
        "--strip-debug-defines",
        action="store_true",
        default=defaults.get("strip-debug-defines", False),
        help="Strip debug defines for release shaders",
    )
    parser.add_argument(
        "--optimization-level",
        default=defaults.get(
            "optimization-level",
            "3" if defaults.get("strip-debug-defines", False) else "1",
        ),
        choices=["0", "1", "2", "3"],
        help="Optimization level (0=none, 1=default, 2=aggressive, 3=max)",
    )
    parser.add_argument(
        "--force-partial-precision",
        action="store_true",
        default=defaults.get("force-partial-precision", False),
        help="Force partial precision (16-bit floats) for performance",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=defaults.get("debug", False),
        help="Enable debug output",
    )
    parser.add_argument(
        "--debug-defines",
        default=defaults.get(
            "debug-defines",
            "DEBUG,_DEBUG,D3D_DEBUG_INFO,D3DCOMPILE_DEBUG,D3DCOMPILE_SKIP_OPTIMIZATION"
        ),
        help="Comma-separated list of defines to treat as debug (default: common debug defines)",
    )
    if not is_gui_mode:
        parser.add_argument("-g", "--gui", action="store_true", help="Run with GUI")
    return parser.parse_args()


def setup_environment(args: argparse.Namespace) -> tuple[int, int | None, bool]:
    """Set up the environment for shader compilation.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        tuple[int, int | None, bool]: CPU count, physical cores, and CI environment flag.
    """
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    signal.signal(signal.SIGINT, handle_termination)
    signal.signal(signal.SIGTERM, handle_termination)
    is_ci = os.getenv("GITHUB_ACTIONS") == "true"
    cpu_count = os.cpu_count() or 4
    physical_cores = (
        psutil.cpu_count(logical=False) if HAS_PSUTIL and psutil.cpu_count(logical=False) is not None else None
    )
    return cpu_count, physical_cores, is_ci


def adjust_target_jobs(
    target_jobs: int,
    cpu_count: int,
    physical_cores: int | None,
    is_ci: bool,
    cpu_usages: deque,
    completed_tasks: int,
) -> tuple[int, str]:
    """Adjust the number of parallel jobs based on system resource usage.

    Args:
        target_jobs (int): Current number of jobs.
        cpu_count (int): Total CPU cores.
        physical_cores (int | None): Physical CPU cores.
        is_ci (bool): Whether running in a CI environment.
        cpu_usages (deque): Recent CPU usage measurements.
        completed_tasks (int): Number of completed tasks.

    Returns:
        tuple[int, str]: Adjusted number of jobs and reason.
    """
    if completed_tasks < 20:
        max_jobs = min(physical_cores or 24, cpu_count - 2) if physical_cores else min(cpu_count - 2, 24)
        return max_jobs, "initial max jobs for first 20 tasks"

    try:
        cpu_usage = psutil.cpu_percent(interval=0.5)
        mem_usage = psutil.virtual_memory().percent
        cpu_usages.append(cpu_usage)
        avg_cpu = sum(cpu_usages) / len(cpu_usages) if cpu_usages else cpu_usage
        new_jobs, reason = get_system_adaptive_jobs(cpu_count, physical_cores, is_ci, avg_cpu)
        if abs(new_jobs - target_jobs) > 2:
            fxc_count = count_fxc_processes()
            logging.info(
                f"Adjusting active jobs to {new_jobs} ({reason}, avg CPU {avg_cpu:.1f}%, memory {mem_usage:.1f}%, fxc_processes={fxc_count})"
            )
            return new_jobs, reason
    except Exception as e:
        logging.debug(f"Failed to check system usage: {e}")
    return target_jobs, "no adjustment"


def initialize_compilation(
    args: argparse.Namespace, cpu_count: int, physical_cores: int | None, is_ci: bool
) -> tuple[int, int, str, list[tuple]]:
    """Initialize compilation settings and tasks.

    Args:
        args (argparse.Namespace): Command-line arguments.
        cpu_count (int): Total CPU cores.
        physical_cores (int | None): Physical CPU cores.
        is_ci (bool): Whether running in a CI environment.

    Returns:
        tuple[int, int, str, list[tuple]]: Max workers, target jobs, jobs reason, and tasks.
    """
    max_workers = min(physical_cores or 24, cpu_count - 2) if physical_cores else min(cpu_count - 2, 24)
    default_jobs = 4
    if args.jobs and args.jobs != default_jobs:
        target_jobs = args.jobs
        jobs_reason = "user-specified"
    else:
        target_jobs, jobs_reason = get_system_adaptive_jobs(cpu_count, physical_cores, is_ci, avg_cpu=0.0)

    if not args.fxc:
        args.fxc = shutil.which("fxc.exe")
        if not args.fxc:
            logging.error("fxc.exe not found in PATH. Please specify with --fxc.")
            return max_workers, target_jobs, jobs_reason, []

    if not os.path.exists(args.shader_dir):
        logging.error(f"Shader directory not found: {args.shader_dir}")
        return max_workers, target_jobs, jobs_reason, []
    if not os.path.exists(args.config):
        logging.error(f"Configuration file not found: {args.config}")
        return max_workers, target_jobs, jobs_reason, []

    tasks = []
    for file_name, shader_type, entry_name, defines in parse_shader_configs(args.config):
        shader_file = os.path.join(args.shader_dir, file_name)
        if not os.path.exists(shader_file):
            logging.error(f"Shader file not found: {file_name}")
            continue
        tasks.append((shader_file, shader_type, entry_name, defines))

    if not tasks:
        logging.error("No valid shader compilation tasks found")
        return max_workers, target_jobs, jobs_reason, []

    os.makedirs(args.output_dir, exist_ok=True)
    return max_workers, target_jobs, jobs_reason, tasks


def manage_jobs(
    target_jobs: int,
    cpu_count: int,
    physical_cores: int | None,
    is_ci: bool,
    cpu_usages: deque,
    completed_tasks: int,
    last_check: float,
    check_interval: float,
    jobs_reason: str,
) -> tuple[int, str, float]:
    """Adjust the number of parallel jobs based on system resources.

    Args:
        target_jobs (int): Current number of jobs.
        cpu_count (int): Total CPU cores.
        physical_cores (int | None): Physical CPU cores.
        is_ci (bool): Whether running in a CI environment.
        cpu_usages (deque): Recent CPU usage measurements.
        completed_tasks (int): Number of completed tasks.
        last_check (float): Time of last job adjustment.
        check_interval (float): Interval between adjustments.
        jobs_reason (str): Reason for current job count.

    Returns:
        tuple[int, str, float]: Adjusted jobs, reason, and last check time.
    """
    current_time = time.time()
    if HAS_PSUTIL and jobs_reason != "user-specified" and current_time - last_check >= check_interval:
        new_jobs, reason = adjust_target_jobs(
            target_jobs,
            cpu_count,
            physical_cores,
            is_ci,
            cpu_usages,
            completed_tasks,
        )
        if new_jobs != target_jobs:
            logging.info(f"Adjusted jobs to {new_jobs} ({reason})")
            return new_jobs, reason, current_time
    return target_jobs, jobs_reason, last_check


def submit_tasks(
    executor: concurrent.futures.ThreadPoolExecutor,
    task_iterator: Any,  # Iterator cannot be typed precisely without collections.abc
    active_tasks: int,
    target_jobs: int,
    args: argparse.Namespace,
    futures: dict,
    shader_dir: str,
) -> tuple[int, Any]:
    """Submit compilation tasks to the executor.

    Args:
        executor (concurrent.futures.ThreadPoolExecutor): Thread pool executor.
        task_iterator (any): Iterator over tasks.
        active_tasks (int): Number of active tasks.
        target_jobs (int): Target number of jobs.
        args (argparse.Namespace): Command-line arguments.
        futures (dict): Mapping of futures to tasks.
        shader_dir (str): Directory containing shader files.

    Returns:
        tuple[int, any]: Updated active tasks and task iterator.
    """
    while active_tasks < target_jobs and task_iterator:
        try:
            task = next(task_iterator)
            future = executor.submit(
                compile_shader,
                args.fxc,
                task[0],
                task[1],
                task[2],
                task[3],
                os.path.abspath(args.output_dir),
                shader_dir,
                args.debug,
                args.strip_debug_defines,
                args.optimization_level,
                args.force_partial_precision,
                args.debug_defines,
            )
            futures[future] = task
            active_tasks += 1
            logging.debug(f"Submitted task: active_tasks={active_tasks}, futures={len(futures)}")
        except StopIteration:
            task_iterator = None
            break
    return active_tasks, task_iterator


def process_completed_futures(
    completed_futures: list,
    futures: dict,
    results: list[dict],
    completion_times: deque,
    pbar: tqdm,
    target_jobs: int,
    jobs_reason: str,
    window_seconds: float,
) -> tuple[int, int]:
    """Process completed compilation tasks.

    Args:
        completed_futures (list): List of completed futures.
        futures (dict): Mapping of futures to tasks.
        results (list[dict]): List of compilation results.
        completion_times (deque): Recent completion times.
        pbar (tqdm): Progress bar.
        target_jobs (int): Target number of jobs.
        jobs_reason (str): Reason for current job count.
        window_seconds (float): Time window for rate calculation.

    Returns:
        tuple[int, int]: Updated active tasks and completed tasks count.
    """
    active_tasks = 0
    completed_tasks = 0
    for future in completed_futures:
        task = futures.pop(future)
        active_tasks -= 1
        completed_tasks += 1
        try:
            result = future.result()
            if result:
                results.append(result)
        except Exception:
            logging.exception(f"Error compiling {task[0]}:{task[2]}")

        current_time = time.time()
        completion_times.append((current_time, 1))
        while completion_times and current_time - completion_times[0][0] > window_seconds:
            completion_times.popleft()

        total_in_window = sum(count for _, count in completion_times)
        if len(completion_times) > 1:
            window_duration = current_time - completion_times[0][0]
            rate = total_in_window / window_duration if window_duration > 0 else 0
            postfix = {"rate": f"{rate:.2f} shaders/s"}
            if jobs_reason != "user-specified":
                postfix["jobs"] = target_jobs
            pbar.set_postfix(postfix)

        pbar.update(1)
        logging.debug(
            f"Completed task: active_tasks={active_tasks}, futures={len(futures)}, fxc_processes={count_fxc_processes()}"
        )
    return active_tasks, completed_tasks


def run_compilation(args: argparse.Namespace, cpu_count: int, physical_cores: int | None, is_ci: bool) -> list[dict]:
    """Run parallel shader compilation.

    Args:
        args (argparse.Namespace): Command-line arguments.
        cpu_count (int): Total CPU cores.
        physical_cores (int | None): Physical CPU cores.
        is_ci (bool): Whether running in a CI environment.

    Returns:
        list[dict]: List of compilation results.
    """
    max_workers, target_jobs, jobs_reason, tasks = initialize_compilation(args, cpu_count, physical_cores, is_ci)
    if not tasks:
        return []

    logging.info(
        f"Starting compilation of {len(tasks)} shader variants with {max_workers} max workers, {target_jobs} active jobs ({jobs_reason})"
    )

    results = []
    completion_times = deque()
    window_seconds = 10
    active_tasks = 0
    futures = {}
    task_iterator = iter(tasks)
    last_check = time.time()
    check_interval = 10
    cpu_usages = deque(maxlen=10)
    completed_tasks = 0

    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor,
        tqdm(total=len(tasks), desc="Compiling shaders", unit="shader") as pbar,
    ):
        while futures or task_iterator:
            if stop_event.is_set():
                break

            target_jobs, jobs_reason, last_check = manage_jobs(
                target_jobs,
                cpu_count,
                physical_cores,
                is_ci,
                cpu_usages,
                completed_tasks,
                last_check,
                check_interval,
                jobs_reason,
            )

            active_tasks, task_iterator = submit_tasks(
                executor, task_iterator, active_tasks, target_jobs, args, futures, args.shader_dir
            )

            completed_futures = [f for f in futures if f.done()]
            active_tasks, new_completed = process_completed_futures(
                completed_futures, futures, results, completion_times, pbar, target_jobs, jobs_reason, window_seconds
            )
            completed_tasks += new_completed

            if futures and active_tasks >= target_jobs:
                time.sleep(0.1)

    return results


def analyze_and_report_results(
    results: list[dict],
    config_file: str,
    output_dir: str,
    suppress_warnings: list[str],
    max_warnings: int,
) -> tuple[int, int, int]:
    """Analyze compilation results and report warnings and errors.

    Args:
        results (list[dict]): Compilation results.
        config_file (str): Path to the YAML configuration file.
        output_dir (str): Output directory for logs.
        suppress_warnings (list[str]): Warning codes to suppress.
        max_warnings (int): Maximum allowed new warnings.

    Returns:
        tuple[int, int, int]: Exit code, total new warnings, and error count.
    """
    baseline_warnings = load_baseline_warnings(config_file)
    defines_lookup = build_defines_lookup(config_file)
    new_warnings, all_warnings, errors, suppressed_warnings_count = process_warnings_and_errors(
        results, baseline_warnings, suppress_warnings, defines_lookup
    )
    log_new_issues(new_warnings, errors, results, output_dir, defines_lookup)

    total_new_warnings = sum(get_instance_count(w) for w in new_warnings)
    logging.info(
        f"Compilation complete: {total_new_warnings} new warnings, {suppressed_warnings_count} suppressed warnings, {len(errors)} errors"
    )
    if new_warnings:
        unique_warnings = sorted(new_warnings, key=lambda x: get_instance_count(x), reverse=True)
        max_to_show = min(10, len(unique_warnings))  # Show more warnings for better visibility
        logging.warning(f"*** NEW WARNINGS DETECTED ({len(unique_warnings)} unique, {total_new_warnings} total):")
        for i, warning in enumerate(unique_warnings[:max_to_show], 1):
            count = get_instance_count(warning)
            affected_variants = len(warning["entries"])
            percentage = (count / total_new_warnings * 100) if total_new_warnings > 0 else 0

            # Enhanced warning display with specific locations
            logging.warning(f"\n{i}. {warning['code']}: {warning['message']}")
            logging.warning(
                f"   Impact: {count} occurrences across {affected_variants} shader variants ({percentage:.1f}%)"
            )  # Show specific file locations and line numbers
            locations_shown = 0
            max_locations = 3  # Show up to 3 specific locations per warning

            # Handle both list and dictionary formats for instances
            if isinstance(warning["instances"], dict):
                # Dictionary format: {"location": {"entries": [...]}}
                for location, location_data in warning["instances"].items():
                    if locations_shown >= max_locations:
                        remaining_locations = len(warning["instances"]) - locations_shown
                        logging.warning(
                            f"   Location: ...and {remaining_locations} more locations (see new_issues.log)"
                        )
                        break

                    # Parse file and line info for better display
                    file_part = location.split(":")[0] if ":" in location else location
                    line_part = ":".join(location.split(":")[1:]) if ":" in location else "unknown"

                    # Show affected shader entries for this location
                    entries_list = list(location_data["entries"])[:2]  # Show first 2 entries
                    entries_str = ", ".join(entries_list)
                    if len(location_data["entries"]) > 2:
                        entries_str += f" (+{len(location_data['entries']) - 2} more)"

                    logging.warning(f"   Location: {file_part}:{line_part} (entries: {entries_str})")
                    locations_shown += 1
            else:
                # List format: ["location1", "location2", ...] (for backward compatibility)
                for location in warning["instances"][:max_locations]:
                    file_part = location.split(":")[0] if ":" in location else location
                    line_part = ":".join(location.split(":")[1:]) if ":" in location else "unknown"
                    logging.warning(f"   Location: {file_part}:{line_part}")
                    locations_shown += 1

                if len(warning["instances"]) > max_locations:
                    remaining_locations = len(warning["instances"]) - max_locations
                    logging.warning(f"   Location: ...and {remaining_locations} more locations (see new_issues.log)")

        if len(unique_warnings) > max_to_show:
            remaining_count = sum(get_instance_count(w) for w in unique_warnings[max_to_show:])
            logging.warning(
                f"\n...and {len(unique_warnings) - max_to_show} more unique warnings ({remaining_count} occurrences)"
            )
            logging.warning("See 'new_issues.log' for complete details and compilation context")

    error_count = sum(len(e) for e in errors.values())
    if errors:
        logging.error(f"*** COMPILATION ERRORS DETECTED ({error_count} total errors):")

        for shader_key, error_data in errors.items():
            shader_file = shader_key.split(":")[0]
            entry_point = ":".join(shader_key.split(":")[1:]) if ":" in shader_key else "unknown"
            shader_type = error_data.get("type", "unknown")

            logging.error(f"\nFile: {shader_file} (entry: {entry_point}, type: {shader_type}):")
            for error in error_data["instances"].values():
                logging.error(f"   ERROR {error['code']}: {error['message']}")
                if error.get("location"):
                    logging.error(f"      Location: {error['location']}")
                if error.get("context"):
                    logging.error(f"      Context: {error['context']['shader_type']} - {error['context']['entry_point']}")

        logging.error("\nACTION REQUIRED: Fix all compilation errors above before proceeding.")
        logging.error("Check the full compilation log for additional context and details.")
        return 1, total_new_warnings, error_count  # Enhanced warning threshold logic to support negative values
    if max_warnings < 0:
        # Negative max_warnings means user must eliminate existing warnings
        baseline_warning_count = sum(len(warning_data["instances"]) for warning_data in baseline_warnings.values())
        current_total_warnings = baseline_warning_count + total_new_warnings
        required_reduction = abs(max_warnings)
        # If required reduction exceeds baseline, target is 0 warnings (eliminate all)
        target_warning_count = max(0, baseline_warning_count - required_reduction)

        if current_total_warnings > target_warning_count:
            eliminated_warnings = baseline_warning_count - (current_total_warnings - total_new_warnings)
            if target_warning_count == 0:
                logging.error(
                    f"*** MUST ELIMINATE ALL WARNINGS: Required reduction {required_reduction} exceeds "
                    f"baseline count {baseline_warning_count}. Current total: {current_total_warnings}, "
                    f"target: {target_warning_count} warnings."
                )
                logging.error("ACTION REQUIRED: Eliminate ALL remaining warnings to pass this check.")
            else:
                needed_elimination = required_reduction - eliminated_warnings
                logging.error(
                    f"*** INSUFFICIENT WARNING REDUCTION: Need to eliminate {required_reduction} warnings, "
                    f"but only eliminated {eliminated_warnings}. Need {needed_elimination} more eliminations."
                )
                logging.error(f"ACTION REQUIRED: Eliminate {needed_elimination} more warnings to pass this check.")

            # Provide actionable guidance
            if new_warnings:
                logging.error(
                    f"TIP: Focus on eliminating existing warnings rather than adding {total_new_warnings} new ones."
                )
                logging.error("Check 'new_issues.log' for detailed locations and suggested fixes.")
            else:
                logging.error("TIP: Eliminate more existing baseline warnings to meet the reduction requirement.")

            logging.info(
                f"SUMMARY: Baseline: {baseline_warning_count} | New: {total_new_warnings} | "
                f"Target: ≤{target_warning_count} | Current: {current_total_warnings}"
            )
            # Show top existing warnings to help user prioritize elimination efforts
            if baseline_warnings:
                logging.error("\n*** TOP EXISTING WARNINGS TO ELIMINATE (by impact):")

                # Calculate entry counts for baseline warnings and sort by impact
                baseline_with_counts = []
                for _warning_key, warning_data in baseline_warnings.items():
                    total_entries = sum(
                        len(location_data.get("entries", [])) for location_data in warning_data["instances"].values()
                    )
                    baseline_with_counts.append({
                        "code": warning_data.get("code", "Unknown"),
                        "message": warning_data.get("message", "Unknown message"),
                        "total_entries": total_entries,
                        "instance_count": len(warning_data["instances"]),
                        "locations": list(warning_data["instances"].keys())[:3],  # Show first 3 locations
                    })

                # Sort by total entries (impact) descending
                top_existing = sorted(baseline_with_counts, key=lambda x: x["total_entries"], reverse=True)[:5]

                # Calculate total entries across all baseline warnings for percentage calculation
                total_baseline_entries = sum(w["total_entries"] for w in baseline_with_counts)

                for i, warning in enumerate(top_existing, 1):
                    impact_percentage = (
                        (warning["total_entries"] / total_baseline_entries * 100) if total_baseline_entries > 0 else 0
                    )
                    logging.error(f"\n{i}. {warning['code']}: {warning['message']}")
                    logging.error(
                        f"   Impact: {warning['total_entries']} shader combinations across {warning['instance_count']} locations ({impact_percentage:.1f}% of baseline)"
                    )
                    logging.error(f"   Sample locations: {', '.join(warning['locations'])}")

                logging.error(
                    f"\nEliminating these top {len(top_existing)} warning types would reduce {sum(w['total_entries'] for w in top_existing)} shader combinations."
                )
                logging.error("Focus on the highest impact warnings first for maximum progress toward the target.")

            return 1, total_new_warnings, error_count

        else:
            eliminated_warnings = baseline_warning_count - (current_total_warnings - total_new_warnings)
            if target_warning_count == 0:
                logging.info(
                    f"All warnings eliminated: required reduction {required_reduction} exceeded "
                    f"baseline count {baseline_warning_count}. Achieved zero warnings goal."
                )

            else:
                logging.info(
                    f"Warning reduction goal met: eliminated {eliminated_warnings} warnings "
                    f"(required: {required_reduction}), {total_new_warnings} new warnings"
                )
    elif total_new_warnings > max_warnings:
        # Positive max_warnings means limit on new warnings (original behavior)
        excess_warnings = total_new_warnings - max_warnings
        logging.error(f"*** TOO MANY NEW WARNINGS: {total_new_warnings} new warnings exceed limit of {max_warnings}")
        logging.error(f"ACTION REQUIRED: Eliminate {excess_warnings} new warnings to pass this check.")

        if new_warnings:
            logging.error("Check 'new_issues.log' for detailed locations and suggested fixes.")
            # Show the most common warnings first for prioritization
            unique_warnings = sorted(new_warnings, key=lambda x: len(x["instances"]), reverse=True)
            logging.error(f"TIP: Focus on the top {min(3, len(unique_warnings))} most frequent warning types:")
            for i, warning in enumerate(unique_warnings[:3], 1):
                count = len(warning["instances"])
                logging.error(f"   {i}. {warning['code']}: {warning['message']} ({count} occurrences)")

        return 1, total_new_warnings, error_count

    return 0, total_new_warnings, error_count


def get_instance_count(warning: dict) -> int:
    """Get the count of instances, handling both list and dict formats."""
    instances = warning.get("instances", [])
    if isinstance(instances, dict):
        return len(instances)
    else:
        return len(instances)


def main() -> int:
    """Main entry point for the shader compilation script.

    Returns:
        int: Exit code (0 for success, 1 for errors or excessive warnings).
    """
    default_jobs = 4
    args = parse_arguments(default_jobs)
    cpu_count, physical_cores, is_ci = setup_environment(args)
    try:
        results = run_compilation(args, cpu_count, physical_cores, is_ci)
    except KeyboardInterrupt:
        logging.warning("Keyboard interrupt received")
        handle_termination()
        results = []

    if stop_event.is_set() and results:
        suppress_warnings = [code.strip() for code in args.suppress_warnings.split(",") if code.strip()]
        exit_code, total_new_warnings, error_count = analyze_and_report_results(
            results, args.config, args.output_dir, suppress_warnings, args.max_warnings
        )
        logging.warning("Compilation was interrupted")
        return exit_code

    suppress_warnings = [code.strip() for code in args.suppress_warnings.split(",") if code.strip()]
    exit_code, total_new_warnings, error_count = analyze_and_report_results(
        results, args.config, args.output_dir, suppress_warnings, args.max_warnings
    )
    return exit_code


if __name__ == "__main__":
    if HAS_GOOEY and ("--gui" in sys.argv or "-g" in sys.argv):

        @Gooey(
            progress_regex=r"(\d+)/(\d+)",
            progress_expr="x[0] / x[1] * 100",
            program_name="Shader Compiler",
            default_size=(800, 600),
        )
        def gooey_main():
            return main()

        sys.exit(gooey_main())
    else:
        sys.exit(main())
