Installation
============

Requirements
------------

CatVision requires Python 3.8 or higher.

Install from PyPI
-----------------

The easiest way to install CatVision is using pip:

.. code-block:: bash

   pip install catvision

Install from Source
-------------------

To install from source:

.. code-block:: bash

   git clone https://github.com/aryashah2k/catvision.git
   cd catvision
   pip install -e .

Development Installation
------------------------

For development, install with the development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

This will install additional packages for testing and development:

* pytest
* pytest-cov
* black
* flake8
* mypy

Dependencies
------------

CatVision depends on the following packages:

* opencv-python-headless >= 4.8.0
* numpy >= 1.24.0
* scipy >= 1.10.0
* matplotlib >= 3.7.0
* Pillow >= 10.0.0

These will be automatically installed when you install CatVision.
