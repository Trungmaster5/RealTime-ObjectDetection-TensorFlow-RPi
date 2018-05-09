import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

import cv2

from picamera.array import PiRGBArray

import picamera

from collections import defaultdict
from io import StringIO
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils


########################################
# DOWNLOAD MODEL
########################################

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017' #fast 
 #MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017' #medium speed 
MODEL_FILE = MODEL_NAME + '.tar.gz' 
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/' 
 
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb' 
 
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt') 
 
NUM_CLASSES = 90 
 
IMAGE_SIZE = (12, 8) 
  
fileAlreadyExists = os.path.isfile(PATH_TO_CKPT) 
 
if not fileAlreadyExists: 
    print('Downloading frozen inference graph') 
    opener = urllib.request.URLopener() 
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE) 
    tar_file = tarfile.open(MODEL_FILE) 
    for file in tar_file.getmembers(): 
        file_name = os.path.basename(file.name) 
        if 'frozen_inference_graph.pb' in file_name: 
            tar_file.extract(file, os.getcwd()) 

#####################################################
# LOAD GRAPH
#####################################################

detection_graph = tf.Graph() 
with detection_graph.as_default(): 
    od_graph_def = tf.GraphDef() 
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid: 
        serialized_graph = fid.read() 
        od_graph_def.ParseFromString(serialized_graph) 
        tf.import_graph_def(od_graph_def, name='')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS) 
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True) 
category_index = label_map_util.create_category_index(categories)

#####################################################
# SETUP PI CAMERA
#####################################################

camera = picamera.PiCamera()

camera.resolution = (1280, 960)
camera.vflip = True
camera.framerate = 30
rawCapture = PiRGBArray(camera, size = (1280,960))

#####################################################
# MAIN LOOP
#####################################################

with detection_graph.as_default(): 
    with tf.Session(graph=detection_graph) as sess: 
        for frame in camera.capture_continuous(rawCapture, format="bgr"): 
                              
            image_np = np.array(frame.array) 
             
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3] 
            image_np_expanded = np.expand_dims(image_np, axis=0) 
             
            # Definite input and output Tensors for detection_graph 
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0') 
             
            # Each box represents a part of the image where a particular object was detected. 
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0') 
             
            # Each score represent how level of confidence for each of the objects. 
            # Score is shown on the result image, together with the class label. 
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0') 
             
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0') 
             
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            print('Running detection..') 
            (boxes, scores, classes, num) = sess.run( 
                [detection_boxes, detection_scores, detection_classes, num_detections], 
                feed_dict={image_tensor: image_np_expanded}) 
     
            print('Done.  Visualizing..') 
            vis_utils.visualize_boxes_and_labels_on_image_array( 
                    image_np, 
                    np.squeeze(boxes), 
                    np.squeeze(classes).astype(np.int32), 
                    np.squeeze(scores), 
                    category_index, 
                    use_normalized_coordinates=True, 
                    line_thickness=8) 
                 
            cv2.imshow('object detection', cv2.resize(image_np, (1280, 960))) 
            rawCapture.truncate(0) 
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                cv2.destroyAllWindows() 
                break 
     
        print('exiting') 
        cap.release() 
        cv2.destroyAllWindows() 
