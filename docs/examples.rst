Examples
========

This page contains various examples of using CatVision.

Basic Image Processing
----------------------

Process a single image with default settings:

.. code-block:: python

   from catvision import CatVision
   from PIL import Image

   # Load image
   img = Image.open("cat.jpg")

   # Apply cat vision
   cat = CatVision()
   result = cat.apply_cat_vision(img)

   # Save result
   result.save("cat_vision.jpg")

Custom Processing Pipeline
---------------------------

Create a custom processing pipeline:

.. code-block:: python

   from catvision import CatVision

   # Custom settings
   cat = CatVision(
       apply_spectral=True,
       apply_spatial=True,
       apply_temporal=False,
       apply_motion=False
   )

   # Process image
   result = cat.apply_cat_vision(img)

Batch Processing
----------------

Process multiple images:

.. code-block:: python

   from catvision import CatVision
   from PIL import Image
   import os

   cat = CatVision()

   input_dir = "input_images"
   output_dir = "output_images"

   for filename in os.listdir(input_dir):
       if filename.endswith(('.jpg', '.png')):
           img = Image.open(os.path.join(input_dir, filename))
           result = cat.apply_cat_vision(img)
           result.save(os.path.join(output_dir, filename))

Video Processing
----------------

Process a video file:

.. code-block:: python

   from catvision import CatVision

   cat = CatVision()
   cat.process_video("input.mp4", "output.mp4", fps=30)

Low-light Enhancement
---------------------

Apply low-light vision enhancement:

.. code-block:: python

   from catvision import CatVision
   from PIL import Image

   img = Image.open("dark_scene.jpg")
   
   cat = CatVision()
   # Cats have excellent low-light vision
   result = cat.apply_cat_vision(img, enhance_lowlight=True)
   
   result.save("enhanced_dark_scene.jpg")

Visualization
-------------

Create comparison visualizations:

.. code-block:: python

   from catvision import CatVision
   from catvision.visualization import create_comparison
   from PIL import Image

   img = Image.open("scene.jpg")
   cat = CatVision()
   result = cat.apply_cat_vision(img)

   # Create side-by-side comparison
   comparison = create_comparison(img, result)
   comparison.save("comparison.jpg")
