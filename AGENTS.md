# Repository Instructions

## Project Structure
- `geofileops/__init__.py` is the public API re-export surface. File and layer
  operations live in `fileops.py`; geometry operations and orchestration live in
  `geoops.py`.
- Put shared runtime options and orchestration helpers in `helpers/`; keep
  lower-level I/O, GDAL/OGR, SQLite, and geometry utilities in `util/`.
  SQL-specific operations live in `geoops_sql/`.
- In Python modules, keep public functions above private helpers.
- When changing a public option or API, update its existing documentation and
  public exports as appropriate. See [API reference](docs/reference.rst) and
  [user guide](docs/user_guide.rst).

## Environment and Validation
- Use the Python environment selected for this workspace in the IDE. It must
  include the development dependencies described in `environment-dev.yml`,
  including native GDAL and SpatiaLite. Setup details are in [development
  docs](docs/development.rst) and [installation docs](docs/installation.rst).
- Before running tests, `pre-commit`, or other Python/GDAL-dependent commands,
  use the IDE-selected environment. The selected interpreter doesn't guarantee
  shell activation; on Windows with Conda, use
  `conda run -p <selected-environment-prefix> --no-capture-output <command>`.
- Run Python tests in the selected environment. On Windows with Conda, use an
  activated Conda prompt or `conda run -p <selected-environment-prefix>
  --no-capture-output python -m pytest <test-path>` so native GIS and SpatiaLite
  DLLs are available; invoking `pytest.exe` directly can fail during extension
  loading.
- Prefer focused tests, for example
  `python -m pytest tests/util/test_io_util.py`. Run the configured hooks
  with `pre-commit run --files <changed-files>`; they apply Ruff formatting,
  Ruff lint, and mypy. Hook configuration is in `.pre-commit-config.yaml`.

## Test Conventions
- Follow nearby tests: use `tmp_path`, monkeypatch external commands and
  environment variables, and skip platform-specific behavior on the wrong OS.
- When renaming a pytest parameter, update both the test function and the string
  names in `@pytest.mark.parametrize`; symbol rename tools do not update those
  strings.

## Project Documentation
- See [contribution guidance](CONTRIBUTING.md), [FAQ](docs/faq.rst), and
  [documentation index](docs/index.rst) rather than duplicating their details
  here.