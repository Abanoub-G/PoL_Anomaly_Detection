import math
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import cv2
import time
import flowiz as fz

from object_detection import yolo_extract_objects_in_image
from optical_flow import reduce_optical_flow_resolution

# === Define paths for image, flownet file and other variables
image_path = "180.jpg"
optical_flow_file_path = "180.flo"
res_x, res_y = 8, 8

# === Get objects detected
image, boxes = yolo_extract_objects_in_image(image_path)
cv2.imshow("Image", image)
cv2.waitKey(0)

# === Read in flownet and convert it to u and v velocity matrices
files = glob.glob(optical_flow_file_path)
floArray = fz.read_flow(files[0])
uv = fz.convert_from_flow(floArray, mode='UV')

# === Extract objects detected areas from optical flow plot

# === Find the background optical flow value 

# === Find which objects have an optical flow value below the background 

# === Calculate the optical flow value for the background

# === Offset the optical flow values of the objects detected which are below that of the background value so that .  

# === Calculate the mean speed of each object and its direction (angle)









# def extract_optical_flow_section_of_detected_object(image, boxes, optical_flow_resultant):







# image, boxes = yolo_extract_objects_in_image("180.jpg")
# image, boxes = yolo_extract_objects_in_image(image_path)
# optical_flow_resultant = reduce_optical_flow_resolution(optical_flow_file_path, res_x, res_y)
# reduce_optical_flow_resolution('180.flo', 10, 10)

# extract_optical_flow_section_of_detected_object(image, boxes, optical_flow_resultant)

# reduce_optical_flow_resolution('180.flo', 10, 10)





# class feature_vector():
#   def __init__(self,w1,w2,w3,optical_flow_resultant, box, confidence):
#     self.velocity_mean_w1       =  * w1
#     self.velocity_variance_w1   =  * w1
#     self.velocity_skewness_w1   =  * w1
#     self.velocity_kurtosis_w1   =  * w1
#     self.C_x_w2                 =  * w2
#     self.C_y_w2                 =  * w2
#     self.area_w2                =  * w2

#     self.classes_confidences_w3 = confidence * w3