import math
import numpy as np
# from sklearn import preprocessing
import matplotlib.pyplot as plt
import glob
import flowiz as fz

files = glob.glob('flow1.flo')
floArray = fz.read_flow(files[0])

uv = fz.convert_from_flow(floArray, mode='UV')

# np.savetxt("u_vector.txt", uv[...,0], fmt="%s")
# np.savetxt("v_vector.txt", uv[...,1], fmt="%s")

# uv = uv - uv.mean()
# u_max = uv[...,0].max()
# u_min = uv[...,0].min()
# u_normalized = (uv[...,0] - u_min) / (u_max - u_min)
# u_normalized = np.array(u_normalized)
# plt.imshow(u_normalized)
# plt.show()

# v_max = uv[...,1].max()
# v_min = uv[...,1].min()
# v_normalized = (uv[...,1] - v_min) / (v_max - v_min)


num_of_rows_in_original = len(uv[:]) # unit is pixel
num_of_columns_in_original = len(uv[:][:][0]) #unit is pixel


# print("num_rows_original",num_rows_original)
# print("num_pixels_columns",num_columns_original)
# print("uv shape is",uv.shape)

num_of_rows_per_block = 10 # unit is pixel
num_of_column_per_block = 10 # unit is pixel

num_of_blocks_for_per_row_orinigal = num_of_rows_in_original/num_of_rows_per_block # unit is block
num_of_blocks_for_per_column_orinigal = num_of_columns_in_original/num_of_column_per_block # unit is block
# print("num_of_blocks_for_per_row_orinigal",num_of_blocks_for_per_row_orinigal)
# print("num_of_blocks_for_per_column_orinigal",num_of_blocks_for_per_column_orinigal)
velocity_of_block_matrix = np.zeros((num_of_blocks_for_per_row_orinigal,num_of_blocks_for_per_column_orinigal,2))
#print("velocity_of_block_matrix shape is",velocity_of_block_matrix.shape)

for i in range(num_of_blocks_for_per_row_orinigal):
  
  for j in range(num_of_blocks_for_per_column_orinigal):

  	temp_u = uv[i*num_of_rows_per_block:(i+1)*num_of_rows_per_block,j*num_of_column_per_block:(j+1)*num_of_column_per_block,0]
  	temp_v = uv[i*num_of_rows_per_block:(i+1)*num_of_rows_per_block,j*num_of_column_per_block:(j+1)*num_of_column_per_block,1]
  	temp_u = temp_u.mean()
  	temp_v = temp_v.mean()
  	velocity_of_block_matrix[i,j,0] = temp_u
  	velocity_of_block_matrix[i,j,1] = temp_v

  	# print("temp_u",temp_u)
  	# print("temp_v",temp_v)
# print("velocity_of_block_matrix",velocity_of_block_matrix)

plt.imshow(velocity_of_block_matrix[...,0])
plt.show()

plt.imshow(velocity_of_block_matrix[...,1])
plt.show()

x_max = velocity_of_block_matrix[...,0].max()
x_min = velocity_of_block_matrix[...,1].min()

y_max = velocity_of_block_matrix[...,0].max()
y_min = velocity_of_block_matrix[...,1].min()


x = np.arange(0,100,1)
y = np.arange(0,100,1)
u = np.zeros((100,100))
v = np.zeros((100,100))

for i in range(num_of_blocks_for_per_row_orinigal):
  
  for j in range(num_of_blocks_for_per_column_orinigal):

    u[i,j] = (velocity_of_block_matrix[i,j,0] - x_min)/(x_max-x_min)
    v[i,j] = (velocity_of_block_matrix[i,j,1] - y_min)/(y_max-y_min)
# u[5,5] = 1   

plt.imshow(u)
plt.show()

plt.imshow(v)
plt.show()

fig, ax = plt.subplots()
q = ax.quiver(x,y,u,v,scale=30)
plt.show()

