# hlslkit

[![Release](https://img.shields.io/github/v/release/alandtse/hlslkit)](https://img.shields.io/github/v/release/alandtse/hlslkit)
[![Build status](https://img.shields.io/github/actions/workflow/status/alandtse/hlslkit/main.yml?branch=main)](https://github.com/alandtse/hlslkit/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/alandtse/hlslkit/branch/main/graph/badge.svg)](https://codecov.io/gh/alandtse/hlslkit)
[![Commit activity](https://img.shields.io/github/commit-activity/m/alandtse/hlslkit)](https://img.shields.io/github/commit-activity/m/alandtse/hlslkit)
[![License](https://img.shields.io/github/license/alandtse/hlslkit)](https://img.shields.io/github/license/alandtse/hlslkit)

Tools for automating HLSL shader compilation, diagnostics, and define management, designed for projects like Skyrim Community Shaders.

-   **GitHub repository**: <https://github.com/alandtse/hlslkit/>
-   **Documentation**: <https://alandtse.github.io/hlslkit/>

## Overview

`hlslkit` provides Python scripts to streamline HLSL shader workflows:

-   **`compile_shaders.py`**: Compiles shaders using `fxc.exe`, supports parallel compilation with dynamic job adjustment, and processes warnings/errors from a YAML configuration.
-   **`generate_shader_defines.py`**: Generates `shader_defines.yaml` from `CommunityShaders.log`, defining shader files, types, entries, and preprocessor defines.
-   **`buffer_scan.py`**: Scans HLSL files for buffer definitions, generates a markdown table of register usage, and detects conflicts across features.

Key features:

-   Robust path normalization (handles forward and backward slashes).
-   Parallel compilation with CPU/memory-aware job scaling (requires `psutil`).
-   Warning/error parsing and suppression for diagnostics.
-   GitHub-integrated buffer reports with conflict detection.

## Prerequisites

-   **Python 3.10+**
-   **Poetry**: For dependency management and virtual environment setup.
-   **Dependencies**: Defined in `pyproject.toml`:
    -   **Required**: `pyyaml`, `tqdm`, `py-markdown-table`, `psutil`, `pcpp`, `jellyfish`.
    -   **Optional**: Install with `poetry install -E gui`:
        -   `gui`: `gooey` (GUI interface, Windows recommended)
-   **fxc.exe**: DirectX shader compiler (included in Windows SDK or DirectX SDK).
-   **CommunityShaders.log**: Log file from Skyrim Community Shaders for `generate_shader_defines.py`.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/alandtse/hlslkit.git
    cd hlslkit
    ```

2. Set up the environment using the Makefile:

    ```bash
    make install
    ```

    This installs Poetry dependencies, sets up the virtual environment, and configures pre-commit hooks.

3. Ensure `fxc.exe` is in your PATH or specify its path with `--fxc`.

## Usage

### Workflow

1. **Generate Shader Defines**:
   Use `generate_shader_defines.py` to parse `CommunityShaders.log` and create `shader_defines.yaml`. This YAML defines shader configurations (files, types, entries, defines) for `compile_shaders.py`.

2. **Compile Shaders**:
   Use `compile_shaders.py` to compile shaders based on `shader_defines.yaml`, outputting compiled shaders to a specified directory.

3. **Scan Buffers**:
   Use `buffer_scan.py` to analyze HLSL files for buffer register usage and detect conflicts, generating a markdown report.

### Example Calls

#### 1. Generate `shader_defines.yaml`

Parse `CommunityShaders.log` to create the YAML configuration:

```bash
python generate_shader_defines.py --log "E:\Documents\my games\Skyrim Special Edition\SKSE\CommunityShaders.log" --output shader_defines.yaml
```

-   `--log`: Path to the log file (required).
-   `--output`: Output YAML file (default: `shader_defines.yaml`).
-   `--update-log`: Optional additional log (e.g., VR log) to merge configs.
-   `-d/--debug`: Enable debug output.
-   `-g/--gui`: Run with GUI (requires `gooey`).

This generates `shader_defines.yaml`, e.g.:

```yaml
common_defines: []
common_pshader_defines: []
common_vshader_defines: []
common_cshader_defines: []
file_common_defines: {}
warnings: {}
errors: {}
shaders:
    - file: RunGrass.hlsl
      configs:
          VSHADER:
              common_defines: []
              entries:
                  - entry: Grass:Vertex:4
                    defines: [D3DCOMPILE_DEBUG]
```

**Note**: Ensure the log contains valid compilation lines (e.g., `[D] Compiling ...`). Untagged errors (e.g., `RunGrass.hlsl(10): error X1000: syntax error`) are not parsed.

#### 2. Compile Shaders

Compile shaders using the generated YAML:

```bash
python compile_shaders.py --shader-dir build\ALL-WITH-AUTO-DEPLOYMENT\aio\Shaders --output-dir build\ShaderCache --config shader_defines.yaml --max-warnings 0
```

Or with custom options:

```bash
python compile_shaders.py --shader-dir build\ALL-WITH-AUTO-DEPLOYMENT\aio\Shaders --output-dir build\ShaderCache --config shader_defines.yaml --jobs 4 --max-warnings 0 --suppress-warnings X1519
```

-   `--shader-dir`: Directory with HLSL files (default: `build/aio/Shaders`).
-   `--output-dir`: Output directory for compiled shaders (default: `build/ShaderCache`).
-   `--config`: Path to `shader_defines.yaml` (default: `shader_defines.yaml`).
-   `--jobs`: Number of parallel jobs (default: dynamic based on CPU).
-   `--max-warnings`: Maximum allowed new warnings (default: 0).
-   `--suppress-warnings`: Comma-separated warning codes to suppress (e.g., `X1519,X3206`).
-   `--fxc`: Path to `fxc.exe` (optional if in PATH).
-   `--strip-debug-defines`: Remove debug defines (e.g., `D3DCOMPILE_DEBUG`).
-   `--optimization-level`: Optimization level (0-3, default: 1 or 3 if stripping debug defines).
-   `--force-partial-precision`: Use 16-bit floats for performance.
-   `-d/--debug`: Enable debug output.
-   `-g/--gui`: Run with GUI (requires `gooey`).

**Note**: Validate `shader_defines.yaml` syntax, as malformed YAML raises uncaught errors.

#### 3. Scan Buffer Usage

Generate a markdown table of buffer register usage:

```bash
python buffer_scan.py
```

-   Run in a directory with HLSL files.
-   Outputs a markdown table to stdout, e.g.:

    ```markdown
    ## Table generated on 2025-05-24 20:09:23

    | Register | Feature | Type   | Name     | File                       | Register Type | Buffer Type | Number | PSHADER | VSHADER | VR    |
    | -------- | ------- | ------ | -------- | -------------------------- | ------------- | ----------- | ------ | ------- | ------- | ----- |
    | t0       | Grass   | Buffer | myBuffer | [src/RunGrass.hlsl:5](...) | SRV           | t           | 0      | False   | True    | False |
    ```

-   Detects conflicts (e.g., same register used by multiple features).

### Workflow Example

To compile shaders for Skyrim Community Shaders:

1. Generate the YAML:
    ```bash
    python generate_shader_defines.py --log "E:\Documents\my games\Skyrim Special Edition\SKSE\CommunityShaders.log"
    ```
2. Compile shaders:
    ```bash
    python compile_shaders.py --shader-dir build\ALL-WITH-AUTO-DEPLOYMENT\aio\Shaders --output-dir build\ShaderCache --config shader_defines.yaml --jobs 4 --max-warnings 0
    ```
3. Check buffer conflicts:
    ```bash
    python buffer_scan.py
    ```

## Development

To contribute to `hlslkit`, clone the repository and set up the development environment:

1. Clone the repository:

    ```bash
    git clone https://github.com/alandtse/hlslkit.git
    cd hlslkit
    ```

2. Set up the environment:
    ```bash
    make install
    ```

### Testing

Run tests with coverage:

```bash
make test
```

This executes `pytest` with coverage, generating an XML report. View the HTML report:

```bash
poetry run pytest --cov --cov-config=pyproject.toml --cov-report=html
open htmlcov/index.html
```

Tests cover:

-   Path normalization (forward/backward slashes).
-   Shader compilation (success, missing files, warnings, timeouts).
-   YAML parsing and define flattening.
-   Log parsing (configs, warnings, tagged errors).
-   Buffer scanning (register usage, conflicts, `#line` directives).

**Note**: Untagged errors (e.g., `error X1000: syntax error`) in logs are not parsed by `generate_shader_defines.py`.

### Code Quality

Run linting, type checking, and dependency checks:

```bash
make check
```

This runs:

-   Poetry lock file consistency check.
-   Pre-commit hooks (linting).
-   Pyright for static type checking.
-   Deptry for obsolete dependencies.

### Documentation

Build and serve documentation with MkDocs:

```bash
make docs
```

Test documentation builds:

```bash
make docs-test
```

### Building

Build a wheel file:

```bash
make build
```

### Contributing

-   Submit issues or pull requests on [GitHub](https://github.com/alandtse/hlslkit).
-   Run `make check` and `make test` before submitting changes.
-   Follow the test suite and coverage guidelines.
-   See [CONTRIBUTING.md](CONTRIBUTING.md) for branching, testing, code quality checks, and submitting pull requests.

### CI/CD

-   **Triggers**: Pull requests, merges to `main`, or new releases.
-   **Publishing**: Configure PyPI/Artifactory per [cookiecutter-poetry](https://fpgmaas.github.io/cookiecutter-poetry/features/publishing/#set-up-for-pypi).
-   **Documentation**: Enable MkDocs per [cookiecutter-poetry](https://fpgmaas.github.io/cookiecutter-poetry/features/mkdocs/#enabling-the-documentation-on-github).
-   **Codecov**: Enable coverage reports per [cookiecutter-poetry](https://fpgmaas.github.io/cookiecutter-poetry/features/codecov/).

## Limitations

-   **generate_shader_defines.py**: Only parses `[E]` or `[W]` tagged errors in logs. Untagged errors (e.g., `RunGrass.hlsl(10): error X1000: syntax error`) are ignored.
-   **compile_shaders.py**: Malformed YAML in `shader_defines.yaml` raises uncaught errors. Validate YAML before running.
-   **buffer_scan.py**: Requires `pcpp` for preprocessing and assumes HLSL files are in the project directory or subdirectories.

## License

[GPL-3.0-or-later](COPYING).

---

Repository initiated with [fpgmaas/cookiecutter-poetry](https://github.com/fpgmaas/cookiecutter-poetry).
