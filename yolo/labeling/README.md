To perform manual labeling :

**1. Import files**

Add all the images you want to the `nnd/` folder
Chose the resizing ratio (on `nnd.py:9`, edit `fx=` and `fy=`)
Run `nnd.py`

**2. Manually perform labeling**

Follow instructions here to label with OpenLabeling : https://github.com/Cartucho/OpenLabeling

**3. Extract you images for future use**

Run `bbox.py` to keep only labeled images. They are now stored in `labeled_img`, along with their labels (txt files).