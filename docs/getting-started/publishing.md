# Publishing Guide: PyPI & TestPyPI 📦

This guide walks you through building, validating, and publishing **PolyRAG** to **TestPyPI** and the official **PyPI** registry.

---

## 1. Prerequisites

Ensure packaging tools are installed:

```bash
pip install build twine
```

Both tools are already configured in `pyproject.toml` under `[project.optional-dependencies.dev]`.

---

## 2. Step 1: Build the Package

Clean up any old build artifacts and generate the source distribution (`.tar.gz`) and wheel (`.whl`):

```bash
# Clean previous builds (optional)
rm -rf dist/ build/ *.egg-info

# Build modern wheel and sdist
python -m build
```

This generates two artifacts in `dist/`:
- `dist/polyrag-0.1.0-py3-none-any.whl` (Binary wheel)
- `dist/polyrag-0.1.0.tar.gz` (Source distribution)

---

## 3. Step 2: Validate Package Metadata

Before uploading, verify the package metadata, license, and README formatting:

```bash
twine check dist/*
```

Expected output:
```text
Checking dist/polyrag-0.1.0-py3-none-any.whl: PASSED
Checking dist/polyrag-0.1.0.tar.gz: PASSED
```

---

## 4. Step 3: Test Upload to TestPyPI (Recommended)

[TestPyPI](https://test.pypi.org/) is a sandbox environment where you can practice uploading without affecting the public registry.

1. Create an account at [test.pypi.org](https://test.pypi.org/account/register/).
2. Generate an API token under **Account Settings** $\rightarrow$ **API Tokens** (set scope to "Entire account" or "Project: polyrag").
3. Upload using `twine`:

```bash
twine upload --repository testpypi dist/*
```

When prompted:
- **Username**: `__token__`
- **Password**: `pypi-your-testpypi-token`

4. Test installing from TestPyPI in an isolated virtual environment:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ polyrag
```

---

## 5. Step 4: Production Release to Official PyPI

Once verified on TestPyPI, publish to the production [PyPI](https://pypi.org) registry:

1. Create an account at [pypi.org](https://pypi.org/account/register/).
2. Create an API token under **Account Settings** $\rightarrow$ **API Tokens**.
3. Upload:

```bash
twine upload dist/*
```

When prompted:
- **Username**: `__token__`
- **Password**: `pypi-your-production-token`

Once uploaded, developers worldwide can install your package with:
```bash
pip install polyrag
pip install "polyrag[all]"
```

---

## 6. Storing Credentials (`.pypirc`)

To avoid entering your token every time, create a `~/.pypirc` file:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-PRODUCTION-TOKEN

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TESTPYPI-TOKEN
```

Secure the file permissions:
```bash
chmod 600 ~/.pypirc
```

---

## 7. Versioning & Subsequent Releases

PolyRAG follows [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`):
- `PATCH` (e.g., `0.1.1`): Bug fixes, minor internal optimizations.
- `MINOR` (e.g., `0.2.0`): New pipelines (e.g., GraphRAG), new vector stores, non-breaking features.
- `MAJOR` (e.g., `1.0.0`): Stable production API or breaking interface changes.

### Release Workflow:
1. Update `version = "0.2.0"` in `pyproject.toml` and `polyrag/__init__.py`.
2. Commit and tag:
   ```bash
   git commit -am "chore(release): bump version to 0.2.0"
   git tag v0.2.0
   git push origin main --tags
   ```
3. Re-build and upload:
   ```bash
   rm -rf dist/
   python -m build
   twine upload dist/*
   ```
