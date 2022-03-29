# TreeRecognition


## Evironment
For convenience, anaconda is recommended.
```bash
conda install --yes --file requirements.txt
```


## Dataset
The data is unavailable for public.


## Segmentation
Using HRNet to segment the tree part from image.

### Configure
All the configure infomation is under the `TreeSegmentation/configs/*.json`.

### Train
Using 4 GTX 1080Ti trains about 2 hours.
```bash
cd TreeSegmentation
bash scripts/train.sh
```

### Synthesize
```bash
cd TreeSegmentation
bash scripts/synthesize.sh
```


## Recognize Tree Structure
Detect keypoints of the tree from the result of segmentation, and then recognize the structure of the tree in the form of string.

### Recognize
```bash
cd PostProcess
bash scripts/recognize.sh
```

### Test
```bash
cd PostProcess
# output the result about skeletonize on the tree
bash scripts/skeleton.sh
# output the result of detected keypoints
bash scripts/keypoint.sh
```
