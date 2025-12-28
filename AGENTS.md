# AGENTS.md - AI Agent Guide for hlslkit

This document provides guidance for AI agents working with the hlslkit repository.

## Repository Overview

**hlslkit** is a Python toolkit for automating HLSL shader compilation, diagnostics, and define management. It's primarily designed for projects like Skyrim Community Shaders but can be used for any HLSL shader workflow.

### Core Components

1. **`hlslkit/compile_shaders.py`** - Main shader compilation script
2. **`hlslkit/generate_shader_defines.py`** - Generates shader configuration from logs
3. **`hlslkit/buffer_scan.py`** - Analyzes buffer register usage and detects conflicts

### Entry Points

The project defines three CLI commands in `pyproject.toml`:

-   `hlslkit-compile` → `hlslkit.compile_shaders:main`
-   `hlslkit-generate` → `hlslkit.generate_shader_defines:main`
-   `hlslkit-buffer-scan` → `hlslkit.buffer_scan:main`

## Development Setup

### Prerequisites

-   Python 3.10+
-   Poetry (dependency management)
-   fxc.exe (DirectX shader compiler) - required for shader compilation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/alandtse/hlslkit.git
cd hlslkit

# Install dependencies and setup environment
make install

# Activate Poetry shell
poetry shell
```

### Available Make Commands

-   `make install` - Install dependencies, setup venv, configure pre-commit hooks
-   `make check` - Run all quality checks (poetry check, pre-commit, pyright, deptry)
-   `make test` - Run pytest with coverage
-   `make build` - Build wheel file
-   `make docs` - Build and serve documentation with MkDocs
-   `make docs-test` - Test documentation builds

## Project Structure

```
hlslkit/
├── hlslkit/                    # Source code
│   ├── __init__.py
│   ├── compile_shaders.py      # Shader compilation
│   ├── generate_shader_defines.py  # Config generation
│   └── buffer_scan.py          # Buffer analysis
├── tests/                      # Test suite
│   ├── test_compile_shaders.py
│   ├── test_generate_shader_defines.py
│   └── test_buffer_scan.py
├── docs/                       # MkDocs documentation
│   ├── index.md               # Main documentation (synced from README.md)
│   └── modules.md             # API documentation
├── pyproject.toml             # Project configuration & dependencies
├── README.md                  # Main documentation
├── CONTRIBUTING.md            # Contribution guidelines
└── Makefile                   # Development commands
```

## Common Workflows

### 1. Generate Shader Configuration

Parse a CommunityShaders log to create `shader_defines.yaml`:

```bash
python -m hlslkit.generate_shader_defines \
  --log "path/to/CommunityShaders.log" \
  --output shader_defines.yaml
```

**Key Parameters:**

-   `--log`: Path to the log file (default: `CommunityShaders.log`)
-   `--output`: Output YAML file (default: `shader_defines.yaml`)
-   `--log-level`: Logging level (default: INFO, choices: DEBUG, INFO, WARNING, ERROR, CRITICAL)
-   `-d/--debug`: Enable debug output
-   `-g/--gui`: Run with GUI (requires gooey)

### 2. Compile Shaders

Compile shaders using the generated YAML configuration:

```bash
python -m hlslkit.compile_shaders \
  --shader-dir build/Shaders \
  --output-dir build/ShaderCache \
  --config shader_defines.yaml \
  --max-warnings 0
```

**Key Parameters:**

-   `--shader-dir`: Directory or single HLSL file to compile
-   `--output-dir`: Output directory for compiled shaders
-   `--config`: Path to shader_defines.yaml
-   `--jobs`: Number of parallel jobs (default: auto-detected)
-   `--max-warnings`: Warning control (0=no new warnings, positive=max new, negative=must eliminate N warnings)
-   `--suppress-warnings`: Comma-separated warning codes (e.g., X1519,X3206)
-   `--fxc`: Path to fxc.exe (if not in PATH)
-   `--strip-debug-defines`: Remove debug defines
-   `--optimization-level`: Optimization level (0-3)
-   `--extra-includes`: Additional include directories

### 3. Scan Buffer Usage

Analyze HLSL files for buffer register usage:

```bash
python -m hlslkit.buffer_scan
```

Outputs a markdown table to stdout showing buffer register usage and conflicts.

## Testing

### Run All Tests

```bash
make test
# or
poetry run pytest --cov --cov-config=pyproject.toml --cov-report=xml
```

### Run Specific Tests

```bash
poetry run pytest tests/test_compile_shaders.py -v
poetry run pytest tests/test_generate_shader_defines.py::test_specific_function -v
```

### View Coverage Report

```bash
poetry run pytest --cov --cov-config=pyproject.toml --cov-report=html
open htmlcov/index.html
```

### Test Coverage Expectations

-   Target: ~85% line coverage
-   Tests cover: path normalization, shader compilation, YAML parsing, log parsing, buffer scanning

## Code Quality

### Run All Quality Checks

```bash
make check
```

This runs:

1. `poetry check --lock` - Verify poetry.lock consistency
2. `pre-commit run --all-files` - Run linting/formatting hooks
3. `pyright` - Static type checking
4. `deptry .` - Check for obsolete dependencies

### Linting and Formatting

The project uses:

-   **Ruff** - Linter and formatter (configured in `pyproject.toml`)
-   **Pyright** - Type checker
-   **Pre-commit** - Git hooks for automation

Configuration in `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py39"
line-length = 120
fix = true
```

## Important Files and Configurations

### pyproject.toml

Main configuration file containing:

-   Project metadata (name, version, description)
-   Dependencies (required and optional)
-   CLI entry points
-   Tool configurations (ruff, pyright, pytest, coverage)

### shader_defines.yaml (Generated)

Configuration file for shader compilation containing:

-   Common defines for all shaders
-   Per-shader-type defines (PSHADER, VSHADER, CSHADER)
-   Per-file configurations
-   Warning and error baselines
-   Shader entries with preprocessor defines

Example structure:

```yaml
common_defines: []
common_pshader_defines: []
common_vshader_defines: []
common_cshader_defines: []
file_common_defines: {}
warnings: {}
errors: {}
shaders:
    - file: Example.hlsl
      configs:
          VSHADER:
              common_defines: []
              entries:
                  - entry: main:vertex:1234
                    defines: [DEBUG=1]
```

## Key Implementation Details

### Warning Management System

The `--max-warnings` parameter supports sophisticated warning control:

1. **Positive values** (e.g., `5`): Allow up to N new warnings
2. **Negative values** (e.g., `-3`): Require elimination of N existing baseline warnings
3. **Zero** (`0`): No new warnings allowed (default)

**How it works:**

-   Baseline warnings are stored in `shader_defines.yaml`
-   New warnings are detected by comparing current compilation against baseline
-   Exit code 1 if warning threshold exceeded

### Path Normalization

The codebase handles both forward and backward slashes for cross-platform compatibility:

-   Windows paths: `build\Shaders\file.hlsl`
-   Unix paths: `build/Shaders/file.hlsl`

### Parallel Compilation

`compile_shaders.py` uses CPU/memory-aware job scaling:

-   Auto-detects optimal job count
-   Requires `psutil` for dynamic adjustment
-   Can be overridden with `--jobs` parameter

## Limitations and Known Issues

1. **generate_shader_defines.py**: Only parses `[E]` or `[W]` tagged errors in logs. Untagged errors (e.g., `file.hlsl(10): error X1000: syntax error`) are ignored.

2. **compile_shaders.py**: Malformed YAML in `shader_defines.yaml` raises uncaught errors. Validate YAML before running.

3. **buffer_scan.py**: Requires `pcpp` for preprocessing and assumes HLSL files are in the project directory or subdirectories.

## Contributing

When making changes:

1. Create a feature branch: `git checkout -b feature-name`
2. Make changes and add tests
3. Run quality checks: `make check`
4. Run tests: `make test`
5. Update documentation if needed
6. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Setup Python
  uses: actions/setup-python@v5
  with:
      python-version: "3.11"

- name: Install hlslkit
  run: pip install git+https://github.com/alandtse/hlslkit.git

- name: Compile shaders
  run: |
      hlslkit-compile \
        --shader-dir build/Shaders \
        --output-dir build/ShaderCache \
        --config shader_defines.yaml \
        --max-warnings 0
```

### Docker Example

```bash
docker build -t hlslkit:latest .
docker run --rm -v $(pwd):/workspace -w /workspace \
  hlslkit:latest hlslkit-compile --help
```

## Resources

-   **GitHub Repository**: <https://github.com/alandtse/hlslkit/>
-   **Documentation**: <https://alandtse.github.io/hlslkit/>
-   **Issue Tracker**: <https://github.com/alandtse/hlslkit/issues>
-   **License**: GPL-3.0-or-later

## Tips for AI Agents

1. **Always run `make check` and `make test`** before considering a task complete
2. **Use absolute paths** when working with shader files to avoid path resolution issues
3. **Check for both forward and backward slashes** in path-related code
4. **Review existing tests** in `tests/` directory to understand expected behavior
5. **Keep line length ≤ 120 characters** per Ruff configuration
6. **Add type hints** where appropriate (pyright is configured for basic type checking)
7. **Update documentation** in both README.md and docs/index.md when adding features
8. **Test with actual HLSL files** when possible to ensure compatibility with fxc.exe
