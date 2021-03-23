import math
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import cv2
import time
import flowiz as fz





# def convert_image_to_flownet2_image(image):
#   pass
# return flownet_image

def convert_optical_flow_to_reduced_veolcity_plot(flownet_image_file_name, res_x, res_y):
  files = glob.glob(flownet_image_file_name)
  floArray = fz.read_flow(files[0])
  uv = fz.convert_from_flow(floArray, mode='UV')

  num_of_rows_in_original = len(uv[:]) # unit is pixel
  num_of_columns_in_original = len(uv[:][:][0]) #unit is pixel

  num_of_rows_per_block = res_x # unit is pixel
  num_of_column_per_block = res_y # unit is pixel

  if num_of_rows_in_original%num_of_rows_per_block:
    return print("ERROR: Optical flow image number of pixels in x direction must be divisible by res_x")
  if num_of_columns_in_original%num_of_column_per_block:
    return print("ERROR: Optical flow image number of pixels in y direction must be divisible by res_y")

  num_of_blocks_per_row_orinigal = int(num_of_rows_in_original/num_of_rows_per_block) # unit is block
  num_of_blocks_per_column_orinigal = int(num_of_columns_in_original/num_of_column_per_block) # unit is block

  # Creating matrix of blocks velocity 
  velocity_of_block_matrix = np.zeros((num_of_blocks_per_row_orinigal,num_of_blocks_per_column_orinigal,3))

  for i in range(num_of_blocks_per_row_orinigal):
    
    for j in range(num_of_blocks_per_column_orinigal):

      temp_u = uv[i*num_of_rows_per_block:(i+1)*num_of_rows_per_block,j*num_of_column_per_block:(j+1)*num_of_column_per_block,0]
      temp_v = uv[i*num_of_rows_per_block:(i+1)*num_of_rows_per_block,j*num_of_column_per_block:(j+1)*num_of_column_per_block,1]
      temp_u = temp_u.mean()
      temp_v = temp_v.mean()
      velocity_of_block_matrix[i,j,0] = temp_u
      velocity_of_block_matrix[i,j,1] = temp_v
      velocity_of_block_matrix[i,j,2] = np.sqrt(temp_u**2 + temp_v**2)

  
  # Normalising the blocks velocity matrix 
  velocity_of_block_matrix_normalised = np.zeros((num_of_blocks_per_row_orinigal,num_of_blocks_per_column_orinigal,3))

  u_max = velocity_of_block_matrix[...,0].max()
  u_min = velocity_of_block_matrix[...,0].min()

  v_max = velocity_of_block_matrix[...,1].max()
  v_min = velocity_of_block_matrix[...,1].min()

  uv_resultant_max = velocity_of_block_matrix[...,2].max()
  uv_resultant_min = velocity_of_block_matrix[...,2].min()


  for i in range(num_of_blocks_per_row_orinigal):
  
    for j in range(num_of_blocks_per_column_orinigal):

      velocity_of_block_matrix_normalised[i,j,0] = (velocity_of_block_matrix[i,j,0] - u_min)/(u_max-u_min) # normalising the u values
      velocity_of_block_matrix_normalised[i,j,1] = (velocity_of_block_matrix[i,j,1] - v_min)/(v_max-v_min) # normalising the v values
      velocity_of_block_matrix_normalised[i,j,2] = (velocity_of_block_matrix[i,j,2] - uv_resultant_min)/(uv_resultant_max-uv_resultant_min) # normalising the rsultant of uv values

  plt.imshow(velocity_of_block_matrix_normalised[...,2])
  plt.colorbar()
  plt.show()



  # rows = np.arange(0,num_of_blocks_per_row_orinigal,1)
  # cols = np.arange(0,num_of_blocks_per_column_orinigal,1)
  # fig, ax = plt.subplots()
  # q = ax.quiver(rows,cols,u,v,scale=30)
  # plt.show()

  # Returns the resultant of u and v for the optical flow image.
  return velocity_of_block_matrix_normalised[...,2] 


def yolo_extract_objects_in_image(image_path):

  # load the COCO class labels our YOLO model was trained on
  yolo_path = "ObjectDetection/yolo-coco"
  labelsPath = os.path.sep.join([yolo_path, "coco.names"])
  LABELS = open(labelsPath).read().strip().split("\n")

  # initialize a list of colors to represent each possible class label
  np.random.seed(42)
  COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

  # derive the paths to the YOLO weights and model configuration
  weightsPath = os.path.sep.join([yolo_path, "yolov3.weights"])
  configPath = os.path.sep.join([yolo_path, "yolov3.cfg"])

  # load our YOLO object detector trained on COCO dataset (80 classes)
  # and determine only the *output* layer names that we need from YOLO
  print("[INFO] loading YOLO from disk...")
  net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
  ln = net.getLayerNames()
  ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

  # load our input image and grab its spatial dimensions
  image = cv2.imread(image_path)
  (H, W) = image.shape[:2]

  # construct a blob from the input image and then perform a forward
  # pass of the YOLO object detector, giving us our bounding boxes and
  # associated probabilities
  blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
  net.setInput(blob)
  start = time.time()
  layerOutputs = net.forward(ln)
  end = time.time()
  # show timing information on YOLO
  print("[INFO] YOLO took {:.6f} seconds".format(end - start))

  # initialize our lists of detected bounding boxes, confidences, and
  # class IDs, respectively
  boxes = []
  confidences = []
  classIDs = []
  confidence_default = 0.5
  threshold_default  = 0.3 

  # loop over each of the layer outputs
  for output in layerOutputs:
    # loop over each of the detections
    for detection in output:
      # extract the class ID and confidence (i.e., probability) of
      # the current object detection
      scores = detection[5:]
      classID = np.argmax(scores)
      confidence = scores[classID]
      # filter out weak predictions by ensuring the detected
      # probability is greater than the minimum probability
      if confidence > confidence_default:
        # scale the bounding box coordinates back relative to the
        # size of the image, keeping in mind that YOLO actually
        # returns the center (x, y)-coordinates of the bounding
        # box followed by the boxes' width and height
        box = detection[0:4] * np.array([W, H, W, H])
        (centerX, centerY, width, height) = box.astype("int")
        # use the center (x, y)-coordinates to derive the top and
        # and left corner of the bounding box
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))
        # update our list of bounding box coordinates, confidences,
        # and class IDs
        boxes.append([x, y, int(width), int(height)])
        confidences.append(float(confidence))
        classIDs.append(classID)


  # apply non-maxima suppression to suppress weak, overlapping bounding
  # boxes
  idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_default, threshold_default)
  print(boxes)

  # ensure at least one detection exists
  if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
      # extract the bounding box coordinates
      (x, y) = (boxes[i][0], boxes[i][1])
      (w, h) = (boxes[i][2], boxes[i][3])
      # draw a bounding box rectangle and label on the image
      color = [int(c) for c in COLORS[classIDs[i]]]
      cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
      cv2.circle(image, (x, y), radius=1, color=(0, 0, 255), thickness=-1)
      text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
      cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, color, 2)

  # show the output image
  cv2.imshow("Image", image)
  cv2.waitKey(0)

  # Outputs: 
  # i) array of boxes for objects detected in the form of [x, y, width, height], 
  #    where x and y are the top corner of the square detected. Image and values 
  #    must be NORMALISED
  # ii) Confidences of each of the 80 classes  
  return boxes


class feature_vector():
  def __init__(self,w1,w2,w3,optical_flow_resultant, box, confidences):
    self.velocity_mean_w1       =  * w1
    self.velocity_variance_w1   =  * w1
    self.velocity_skewness_w1   =  * w1
    self.velocity_kurtosis_w1   =  * w1
    self.C_x_w2                 =  * w2
    self.C_y_w2                 =  * w2
    self.area_w2                =  * w2

    self.classes_confidences_w3 = confidences * w3





    

def produce_feature_vector(optical_flow_image, boxes_of_detected_objects, ):


# def map_yolo_to optical_flow():
# yolo_extract_objects_in_image("180.jpg")

convert_optical_flow_to_reduced_veolcity_plot('180.flo', 10, 10)