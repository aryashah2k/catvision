Quick Start Guide
=================

This guide will help you get started with CatVision quickly.

Basic Usage
-----------

The simplest way to use CatVision is with the main ``CatVision`` class:

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

Processing Options
------------------

You can customize the processing with various options:

.. code-block:: python

   from catvision import CatVision

   # Create instance with custom settings
   cat = CatVision(
       apply_spectral=True,
       apply_spatial=True,
       apply_temporal=True,
       apply_motion=True
   )

Working with Videos
-------------------

CatVision can also process video files:

.. code-block:: python

   from catvision import CatVision

   cat = CatVision()
   
   # Process a video file
   cat.process_video("input.mp4", "output.mp4")

Advanced Features
-----------------

For more control over the processing pipeline, you can use individual modules:

.. code-block:: python

   from catvision.spectral import apply_spectral_sensitivity
   from catvision.spatial import apply_spatial_processing
   import numpy as np

   # Your image as numpy array
   image = np.array(...)

   # Apply specific processing steps
   spectral_result = apply_spectral_sensitivity(image)
   spatial_result = apply_spatial_processing(spectral_result)

Next Steps
----------

* Check out the :doc:`api` for detailed API documentation
* See :doc:`examples` for more usage examples
* Read about :doc:`contributing` if you want to contribute to the project
