# CatVision Documentation

This directory contains the Sphinx documentation for CatVision.

## Building Documentation Locally

To build the documentation locally:

1. Install the documentation dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Build the HTML documentation:
   ```bash
   # On Windows
   make.bat html
   
   # On Linux/Mac
   make html
   ```

3. Open `_build/html/index.html` in your browser to view the documentation.

## Read the Docs

The documentation is automatically built and hosted on Read the Docs when you push to your repository.

Configuration is in `.readthedocs.yaml` at the project root.

## Documentation Structure

- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation page
- `installation.rst` - Installation instructions
- `quickstart.rst` - Quick start guide
- `api.rst` - API reference (auto-generated from docstrings)
- `examples.rst` - Usage examples
- `contributing.rst` - Contributing guidelines
- `requirements.txt` - Documentation build dependencies

## Adding New Pages

1. Create a new `.rst` file in this directory
2. Add it to the `toctree` in `index.rst`
3. Build the documentation to verify

## Updating API Documentation

The API documentation is automatically generated from docstrings in the source code. To update:

1. Update docstrings in the source code
2. Rebuild the documentation
3. The changes will appear in the API Reference section
