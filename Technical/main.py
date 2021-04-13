import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import os
import cv2
import time
import flowiz as fz

from object_detection import yolo_extract_objects_in_image
from optical_flow import reduce_optical_flow_resolution

# ======
# === Define paths for image, flownet file and other variables
# ======
image_path = "180.jpg"
optical_flow_file_path = "180.flo"
res_x, res_y = 8, 8

# ======
# === Get objects detected
# ======
image, boxes = yolo_extract_objects_in_image(image_path)
# plt.imshow(image)
# plt.show()
# cv2.imshow("Image", image)
# cv2.waitKey(0)


# ======
# === Read in flownet and convert it to u and v velocity matrices
# ======
files = glob.glob(optical_flow_file_path)
floArray = fz.read_flow(files[0])
uv = fz.convert_from_flow(floArray, mode='UV')

num_of_rows_in_original = len(uv[:]) # unit is pixel
num_of_columns_in_original = len(uv[:][:][0]) #unit is pixel
uv_summary = np.zeros((num_of_rows_in_original,num_of_columns_in_original,4)) # u|v|uv_resultant|angle from x axis

for i in range(num_of_rows_in_original):
  for j in range(num_of_columns_in_original):
    temp_u = uv[i,j,0]
    temp_v = uv[i,j,1]
    # temp_angle  = np.arctan2(temp_v,temp_u)
    uv_summary[i,j,0] = temp_u
    uv_summary[i,j,1] = temp_v
    # uv_summary[i,j,2] = np.sqrt(temp_u**2 + temp_v**2)
    # uv_summary[i,j,3] = temp_angle
plt.imshow(uv_summary[...,0])
plt.colorbar()
plt.show()
# ======
# === Find the background optical flow value     
# ======
#TODO Need to think of how to cater for overlapoing objects when counting pixels in objects to calculate the background velocity value
all_pixels_num = num_of_rows_in_original*num_of_columns_in_original
all_u_sum = uv_summary[...,0].sum()
all_v_sum = uv_summary[...,1].sum()
obj_pixels_num = 0 
obj_u_sum = 0
obj_v_sum = 0


for box in boxes:
  # print(box) 
  (x, y) = (box[0], box[1])
  (w, h) = (box[2], box[3])
  obj_u  = uv_summary[y:y+h,x:x+w,0]
  obj_v  = uv_summary[y:y+h,x:x+w,1]
  obj_u_sum = obj_u_sum + obj_u.sum()
  obj_v_sum = obj_v_sum + obj_v.sum()
  obj_pixels_num = obj_pixels_num + len(obj_u)*len(obj_u[0])

background_u = (all_u_sum - obj_u_sum)/(all_pixels_num - obj_pixels_num)
background_v = (all_v_sum - obj_v_sum)/(all_pixels_num - obj_pixels_num)

# ======
# === Offset optical flow values so that background becomes the minimum     
# ======
uv_summary[...,0] = uv_summary[...,0] - background_u
uv_summary[...,1] = uv_summary[...,1] - background_v

for i in range(num_of_rows_in_original):
  for j in range(num_of_columns_in_original):
    temp_u      = uv_summary[i,j,0]
    temp_v      = uv_summary[i,j,1]  
    temp_angle  = np.arctan2(temp_v,temp_u)
    uv_summary[i,j,2] = np.sqrt(temp_u**2 + temp_v**2)
    uv_summary[i,j,3] = temp_angle

plt.imshow(uv_summary[...,0])
plt.colorbar()
plt.show()

plt.imshow(uv_summary[...,1])
plt.colorbar()
plt.show()

plt.imshow(uv_summary[...,2])
plt.colorbar()
plt.show()

# ======
# === Populate feature vector     
# ======
# TODO: Output confidences from yolo function
# TODO: Normalise all values being inputed
print(len(boxes))
obj_feature_vectors = []
for box in boxes:
  # print(box) 
  (x, y) = (box[0], box[1])
  (w, h) = (box[2], box[3])

  speed    = uv_summary[y:y+h,x:x+w,2].mean()
  # variance = 
  # skewness = 
  # kurtosis = 
  C_x      = x + w/2
  C_y      = y + h/2
  area     = w * h


# class feature_vector():
#   def __init__(self,w1,w2,w3,optical_flow_resultant, box, confidence):
#     self.velocity_mean_w1       =  * w1
#     self.velocity_variance_w1   =  * w1
#     self.velocity_skewness_w1   =  * w1
#     self.velocity_kurtosis_w1   =  * w1
#     self.C_x_w2                 =  * w2
#     self.C_y_w2                 =  * w2
#     self.area_w2                =  * w2























# (x, y) = (boxes[0][0], boxes[0][1])
# (w, h) = (boxes[0][2], boxes[0][3])
# obj_uv_resultant = uv_summary[y:y+h,x:x+w,2]
# img = uv_summary[...,2]
# print(x)
# print(y)

# fig, ax = plt.subplots()
# ax.imshow(uv_summary[...,2])
# plt.colorbar()
# rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
# PCM = ax.get_children()[2]
# plt.colorbar(PCM,ax=ax)
# ax.add_patch(rect)
# plt.show()

# plt.imshow(uv_summary[y:y+h,x:x+w,2])
# plt.show()


# cv2.imshow('i',img)
# cv2.rectangle(image, (x, y), (x + w, y + h), [255,255,255], 2)
# cv2.imshow("Image", image)
# cv2.waitKey(0)

# img = np.random.randint(222, size=(100, 100,3))
# gen = np.array(img ,dtype=np.uint8)
# cv2.imshow('i',img)
# cv2.waitKey(0)
# cv2.destroyWindow('i')
# print()










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