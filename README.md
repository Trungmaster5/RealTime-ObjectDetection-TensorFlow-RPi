# RealTime-ObjectDetection-TensorFlow-RPi
Object detection on Raspberry Pi 3 with TensorFlow

This project adapts the work from: https://www.theta.co.nz/news-blogs/tech-blog/object-detection-on-a-raspberry-pi/.
## Installation:
### Install dependencies
```
sudo apt-get install libatlas-base-dev
sudo apt-get install lubjasper-dev

sudo pip install nympy h5py pillow
sudo pip install lxml
sudo pip install jupyter
sudo pip install matplotlib

sudo apt-get install protobuf-compiler
```
Compile the Object Detection API using Protobuf. 
```
sudo protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PythONPATH:`pwd`:`pwd`/slim
```
### Install tensorflow (1.7.0)
```
sudo pip install tensorflow-1.7.0-cp27-none-any.whl
```
Check installation
```
python
> import tensorflow as tf
> tf.__version__
```
If tensorflow version is 1.7.0 --> Installation successful 

### Install openCV for both python2 and 3 ( in fact only python2 is required)
```
pip3 install opencv-python
pip install opencv-python
```
Check installation (python 2)
```
python
>import cv2
```
Check installation (python 3)
```
python3
>import cv2
```

## Start detection
```
python start.py
```
