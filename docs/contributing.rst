Contributing
============

We welcome contributions to CatVision! This document provides guidelines for contributing.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bugfix
4. Make your changes
5. Run tests to ensure everything works
6. Submit a pull request

Development Setup
-----------------

Install the package in development mode with all dependencies:

.. code-block:: bash

   git clone https://github.com/aryashah2k/catvision.git
   cd catvision
   pip install -e ".[dev]"

Running Tests
-------------

Run the test suite using pytest:

.. code-block:: bash

   pytest

Run tests with coverage:

.. code-block:: bash

   pytest --cov=catvision --cov-report=html

Code Style
----------

We use Black for code formatting and flake8 for linting.

Format your code:

.. code-block:: bash

   black src/catvision tests

Check for linting issues:

.. code-block:: bash

   flake8 src/catvision tests

Type Checking
-------------

We use mypy for type checking:

.. code-block:: bash

   mypy src/catvision

Pull Request Guidelines
------------------------

* Write clear, descriptive commit messages
* Include tests for new features
* Update documentation as needed
* Ensure all tests pass
* Follow the existing code style
* Keep pull requests focused on a single feature or fix

Reporting Issues
----------------

If you find a bug or have a feature request:

1. Check if the issue already exists
2. If not, create a new issue on GitHub
3. Provide a clear description and reproduction steps
4. Include relevant code snippets or error messages

Documentation
-------------

Documentation is built using Sphinx. To build the docs locally:

.. code-block:: bash

   cd docs
   make html

The built documentation will be in ``docs/_build/html``.

License
-------

By contributing, you agree that your contributions will be licensed under the MIT License.
