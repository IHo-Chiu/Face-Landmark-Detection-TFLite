
# Face Landmark Detection

base on: https://github.com/ZhenglinZhou/STAR

## Source Code Structure
|-- FaceLandmarkDetection
  |-- Datasets
    |-- annotations
      |-- ivslab
        | test.tsv
        | train.tsv
    |-- images
      |-- ivslab
        |-- 300W
        |-- afw
        |-- helen
        |-- ibug
        |-- IFPW
  |-- Inference
    | run_model.py
    | star.tflite
    | yolo.tflite
  |-- Models // several trained STAR models
  |-- STAR // github repo
  

## Environment
### Docker
``` bash
docker build -t your_account/face_landmark:1.0.0 .
docker run -it -rm -d --user root --gpus all --shm-size 8G \
    -v output_folder:output_folder \
    -v image_list_folder:image_list_folder \
    --name face-landmark \
    your_account/face_landmark:1.0.0 python3 run_model.py image_list_folder/image_list.txt output_folder
```
### Conda
``` bash
conda create -n STAR python==3.7.3
conda activate STAR
cd FaceLandmarkDetection/STAR/
pip install -r requirements.txt
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio

pip uninstall opencv-python
sudo apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
sudo apt-get install libgl1  -y
pip install numpy opencv-python imutils

conda install dlib
pip install gradio

pip install torchsummary
```

## Functions
### Inference
``` bash
python3 run_model.py image_list.txt output_folder --visualize
```
### Train STAR Model
``` bash
cd STAR
python main.py --mode=train --device_ids=0 \
               --image_dir=../Datasets/images \
               --annot_dir=../Datasets/annotations \
               --data_definition=ivslab \
               --batch_size 128 \
               --train_num_workers 4 \
               --val_batch_size 128 \
               --val_num_workers 4
```
### Train STAR Model With Distilation
`FaceLandmarkDetection/STAR/conf/alignment.py`
* self.distill = True

