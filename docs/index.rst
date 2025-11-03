.. CatVision documentation master file

Welcome to CatVision's documentation!
======================================

CatVision is a biologically accurate cat vision filter for neuroscience and visual perception research.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
   contributing

Overview
--------

CatVision provides tools to simulate how cats perceive the visual world, based on the latest research in feline vision and neuroscience. This package is designed for researchers, educators, and enthusiasts interested in understanding animal vision.

Key Features
------------

* **Biologically Accurate**: Based on peer-reviewed research on feline vision
* **Easy to Use**: Simple API for processing images and videos
* **Comprehensive**: Covers spectral, spatial, temporal, and motion processing
* **Well-Documented**: Extensive documentation and examples
* **Research-Grade**: Suitable for scientific publications and educational purposes

Quick Example
-------------

.. code-block:: python

   from catvision import CatVision
   from PIL import Image

   # Load an image
   image = Image.open("input.jpg")

   # Create a CatVision instance
   cat = CatVision()

   # Apply cat vision filter
   result = cat.apply_cat_vision(image)

   # Save the result
   result.save("output.jpg")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
