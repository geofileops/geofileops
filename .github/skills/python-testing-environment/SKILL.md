---
name: python-testing-environment
description: "Use when running Python scripts, pytest, pre-commit, or other Python/GDAL-dependent commands in geofileops. Use the Python environment selected for the workspace in the IDE and preserve native GDAL and SpatiaLite DLL paths, especially on Windows."
---

# Python Environment and Tests

Use this workflow for Python commands in this repository, including test runs, scripts, formatting, lint, and type checks.

## Select the Environment

1. Get the Python environment selected for the geofileops workspace from the VS Code Python environment provider. Do not infer it from the shell's `python`, `where python`, or `conda` commands; shell `PATH` may not contain the selected environment.
2. Use that selected environment regardless of its name. Check the repository's `AGENTS.md` for required dependencies. If the selected environment lacks a required dependency, report that rather than silently switching environments.
3. On Windows, for a Conda-selected environment, do not invoke `pytest.exe` or `python.exe` directly for tests. Native GDAL and SpatiaLite DLL directories may not be available until Conda activates the environment.
4. If `conda` is not on `PATH`, derive the Conda executable from the selected interpreter. For a standard Conda layout such as `<root>\\envs\\<environment-name>\\python.exe`, the environment prefix is the interpreter's parent directory and Conda is `<root>\\Scripts\\conda.exe`. Prefer `conda run -p <environment-prefix>` so a duplicate environment name cannot select a different installation. For a non-Conda selected environment, use its selected interpreter or environment manager rather than substituting another Python installation.

In PowerShell, with `$pythonExe` set to the selected interpreter path:

```powershell
$envPrefix = Split-Path $pythonExe -Parent
$condaRoot = Split-Path (Split-Path $envPrefix -Parent) -Parent
$condaExe = Join-Path $condaRoot 'Scripts\conda.exe'
```

Check that `$condaExe` exists before using it. If the selected environment does not use this standard layout, locate its Conda executable through the selected environment's metadata or active VS Code environment details; do not silently switch to another Python installation.

## Run Commands

For a Conda-selected environment, run tests through its prefix, using `python -m pytest`:

```powershell
& $condaExe run -p $envPrefix --no-capture-output python -m pytest tests/util/test_io_util.py -q
```

Run an individual Python script the same way, replacing the script and arguments as needed. For a non-Conda selected environment, invoke its selected interpreter or environment manager:

```powershell
& $condaExe run -p $envPrefix --no-capture-output python path/to/script.py --option value
```

For configured hooks, pass the changed files explicitly:

```powershell
& $condaExe run -p $envPrefix --no-capture-output pre-commit run --files path/to/changed_file.py
```

Prefer the narrowest test command that validates the change. Use a broader suite only when the change's scope requires it.

## Handle Failures Honestly

- If Conda cannot be found, derive its path from the selected interpreter before concluding that the environment is unavailable.
- If collection fails because a native DLL cannot load, verify that the command used `conda run` with the selected environment prefix; do not report the tests as run.
- Do not install packages or modify the environment as part of troubleshooting unless requested.
- Report the exact command and distinguish a test failure from a test command that failed before collection.
