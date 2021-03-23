import math
import numpy as np
import matplotlib.pyplot as plt
import glob
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



convert_optical_flow_to_reduced_veolcity_plot('flow1.flo', 10, 10)


