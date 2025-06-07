# Contributing to `hlslkit`

Contributions to `hlslkit` are welcome and greatly appreciated! Whether you're fixing bugs, adding features, improving documentation, or reporting issues, your efforts help enhance our HLSL shader compilation and diagnostics tools for projects like Skyrim Community Shaders.

## Types of Contributions

### Report Bugs

File bug reports at <https://github.com/alandtse/hlslkit/issues>. Include:

-   Operating system name and version (e.g., Windows 10).
-   Local setup details (e.g., Python version, `fxc.exe` availability, Poetry version).
-   Steps to reproduce the bug, including logs (e.g., `CommunityShaders.log` snippets or `new_warnings.log` from `compile_shaders.py`).

### Fix Bugs

Check <https://github.com/alandtse/hlslkit/issues> for issues tagged "bug" and "help wanted". These are open for contributors to fix. Example bugs:

-   Uncaught `yaml.YAMLError` in `compile_shaders.py`.
-   Untagged error parsing in `generate_shader_defines.py`.

### Implement Features

Look for issues tagged "enhancement" and "help wanted" at <https://github.com/alandtse/hlslkit/issues>. Propose new features via issues, explaining:

-   How the feature works (e.g., new shader type support, enhanced warning suppression).
-   A narrow scope for easier implementation.
-   Relevance to HLSL workflows (e.g., compilation, buffer analysis, log parsing).

### Write Documentation

Improve `hlslkit`’s documentation:

-   Update docstrings in `compile_shaders.py`, `generate_shader_defines.py`, or `buffer_scan.py`.
-   Enhance `README.md` or MkDocs pages (<https://alandtse.github.io/hlslkit/>).
-   Write tutorials on shader automation (e.g., integrating with Skyrim Community Shaders).

### Submit Feedback

Share feedback or feature proposals at <https://github.com/alandtse/hlslkit/issues>. Keep suggestions focused on HLSL shader automation.

## Get Started

Follow these steps to set up `hlslkit` for local development. This guide assumes you have **Poetry**, **Git**, and **Python 3.8+** installed.

### 1. Fork and Clone the Repository

Fork `hlslkit` on GitHub, then clone your fork:

```bash
cd <your-working-directory>
git clone git@github.com:YOUR_USERNAME/hlslkit.git
cd hlslkit
```

### 2. Set Up the Environment

Install dependencies and configure the Poetry environment:

```bash
make install
```

This:

-   Installs dependencies from `pyproject.toml` (e.g., `pyyaml`, `tqdm`, optional `psutil`, `gooey`, `py-markdown-table`, `pcpp`).
-   Sets up the virtual environment.
-   Installs pre-commit hooks for linting/formatting.

Activate the Poetry shell:

```bash
poetry shell
```

**Note**: Ensure `fxc.exe` (DirectX shader compiler) is in your PATH or specify its path with `--fxc` in `compile_shaders.py`. For `buffer_scan.py`, HLSL files should be in the project directory or subdirectories.

### 3. Create a Branch

Create a branch for your changes:

```bash
git checkout -b name-of-your-bugfix-or-feature
```

### 4. Make Changes

Modify scripts, tests, or documentation. Key scripts:

-   **`compile_shaders.py`**: Compiles HLSL shaders using `fxc.exe`, supports parallel compilation, and processes warnings/errors from `shader_defines.yaml`.
-   **`generate_shader_defines.py`**: Generates `shader_defines.yaml` from `CommunityShaders.log`, defining shader configs.
-   **`buffer_scan.py`**: Scans HLSL files for buffer registers, generates markdown tables, and detects conflicts.

**Example Usage** (for reference):

-   Generate `shader_defines.yaml`:
    ```bash
    python generate_shader_defines.py --log "E:\Documents\my games\Skyrim Special Edition\SKSE\CommunityShaders.log" --output shader_defines.yaml
    ```
-   Compile shaders:
    ```bash
    python compile_shaders.py --shader-dir build\ALL-WITH-AUTO-DEPLOYMENT\aio\Shaders --output-dir build\ShaderCache --config shader_defines.yaml --jobs 4 --max-warnings 0
    ```
-   Scan buffers:
    ```bash
    python buffer_scan.py
    ```

**Limitations to Note**:

-   `compile_shaders.py`: Malformed YAML in `shader_defines.yaml` raises uncaught errors. Validate YAML syntax.
-   `generate_shader_defines.py`: Only parses `[E]` or `[W]` tagged errors in logs. Untagged errors (e.g., `RunGrass.hlsl(10): error X1000: syntax error`) are ignored.
-   `buffer_scan.py`: Requires `pcpp` and assumes HLSL files are accessible.

### 5. Add Tests

Add test cases in the `tests` directory. Tests cover:

-   Path normalization (forward/backward slashes).
-   Shader compilation (success, missing files, warnings, timeouts).
-   YAML parsing and define flattening.
-   Log parsing (configs, tagged `[E]`/`[W]` errors).
-   Buffer scanning (register usage, `#line` directives, conflicts).

Run tests:

```bash
make test
```

View the HTML coverage report:

```bash
poetry run pytest --cov --cov-config=pyproject.toml --cov-report=html
open htmlcov/index.html
```

### 6. Check Code Quality

Run linting, type checking, and dependency checks:

```bash
make check
```

This executes:

-   `poetry check --lock` (lock file consistency).
-   Pre-commit hooks (linting/formatting).
-   Pyright (static type checking).
-   Deptry (obsolete dependencies).

### 7. Update Documentation

For new functionality:

-   Update docstrings in modified scripts.
-   Add features to `README.md`’s feature list.
-   Enhance MkDocs pages:
    ```bash
    make docs
    ```
    Test documentation builds:
    ```bash
    make docs-test
    ```

### 8. Commit and Push

Commit changes:

```bash
git add .
git commit -m "Your detailed description of changes."
git push origin name-of-your-bugfix-or-feature
```

### 9. Optional: Run Tox

Test across multiple Python versions:

```bash
poetry run tox
```

This requires multiple Python versions installed. Alternatively, rely on the CI/CD pipeline, which runs `tox` automatically.

### 10. Submit a Pull Request

Submit a pull request (PR) via GitHub. Ensure your PR:

-   Includes tests for new functionality.
-   Updates documentation (docstrings, `README.md`, MkDocs).
-   Passes `make check` and `make test`.

## Pull Request Guidelines

-   Include tests for new functionality or bug fixes.
-   Update documentation for new features (docstrings, `README.md`, MkDocs).
-   Ensure `make check` passes (linting, type checking).
-   Verify `make test` passes with adequate coverage.
-   Provide a clear PR description, referencing related issues.

## Additional Notes

-   **CI/CD**: The pipeline runs `tox`, `make check`, and `make test` on PRs, merges to `main`, and releases.
-   **Coverage**: Aim for ~85% line coverage (check with `make test` or HTML report).
-   **Debugging**: Use `-d/--debug` flags in scripts for verbose logs. Check `new_warnings.log` for `compile_shaders.py` issues.

Thank you for contributing to `hlslkit`!
