To train the model with our bib data! :)

Get the pre-trained weights here:
```
wget https://pjreddie.com/media/files/darknet53.conv.74
```

Generate the sets and train darknet:
```
python train.py
```

The weights should be saved at `/yolo/training/backup/*`.