To perform manual labeling :

**1. Import files**

Add all the images you want to an `to_label/` folder placed in the `labeling/` folder.
Chose the resizing ratio (on `label.py:9`, edit `fx=` and `fy=`)
Run `label.py`

**2. Manually perform labeling**

Follow instructions here to label with OpenLabeling : https://github.com/Cartucho/OpenLabeling

**3. Extract you images for future use**

Run `get_images.py` to keep only labeled images. They are now stored in `labeled_img`, along with their labels (txt files).