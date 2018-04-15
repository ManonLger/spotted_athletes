**Yolo** : https://pjreddie.com/darknet/yolo/
First of all, you need to download pre-trained weights and to build the network :
```
cd ../darknet
make
cd ../yolo
wget https://pjreddie.com/media/files/yolov3.weights
```

If you want to make it work manually with a list of images :
```
cd ../darknet
./darknet detector txt cfg/coco.data cfg/yolov3.cfg yolov3.weights ../samples/samples.txt
```
Box coordinates are saved to a `*_boxes.txt` file, with each line like: `<original_file_name> <person_nb> <left> <right> <top> <bottom>`.

Or, if you're lazy and you want to use our python script to perform detection + crop images and save them in a folder :
```
python run.py samples.txt
```
Cropped images are saved to a new `samples.txt_results` directory.
