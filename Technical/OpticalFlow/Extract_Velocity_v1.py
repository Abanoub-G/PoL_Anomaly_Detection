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
u_max = uv[...,0].max()
u_min = uv[...,0].min()
u_normalized = (uv[...,0] - u_min) / (u_max - u_min)

v_max = uv[...,1].max()
v_min = uv[...,1].min()
v_normalized = (uv[...,1] - v_min) / (v_max - v_min)

print("max uv  = ",u_max)
print("min uv  = ",u_min)
print("max u  = ",u_normalized.max())
print("max v  = ",v_normalized.max())
# print(uv_normalized)
# print(uv)
# uv_normalized = preprocessing.normalize(uv)
# uv = uv_normalized
no_pixels_rows = len(uv)
no_pixels_columns = len(uv[...,1][0])

no_blocks_rows = int(100)
no_blocks_columns = int(100)

block_pixels_no_rows    = no_pixels_rows/no_blocks_rows
block_pixels_no_columns = no_pixels_columns/no_blocks_columns

# Creating matrix of blocks velocity 
vel_matrix_blocks = np.zeros((no_blocks_rows,no_blocks_columns))

# Making the matrix have tuples so each tuple will have (u,v) velocities for each block
vel_matrix_blocks =  np.array(list(zip(vel_matrix_blocks.ravel(),vel_matrix_blocks.ravel())), dtype=('i4,i4')).reshape(vel_matrix_blocks.shape)

rows = np.arange(0, no_blocks_rows, 1)
coloumns = np.arange(0, no_blocks_columns, 1)
U, V = np.meshgrid(rows, coloumns)

for i in range(no_blocks_rows):
	rows_index_start = int(i * block_pixels_no_rows)
	rows_index_end   = int(rows_index_start + block_pixels_no_rows)

	for j in range(no_blocks_columns):
		columns_index_start = int(i * block_pixels_no_columns)
		columns_index_end   = int(columns_index_start + block_pixels_no_columns)


		# u_ij = uv[0:2,0,0]#uv[rows_index_start:rows_index_end][columns_index_start:columns_index_end]
		u_ij = u_normalized[rows_index_start:rows_index_end,columns_index_start:columns_index_end]
		u_ij = u_ij.mean()

		v_ij = v_normalized[rows_index_start:rows_index_end,columns_index_start:columns_index_end]
		v_ij = v_ij.mean()
		print(u_ij)
		print(v_ij)
		print("=======================")

		vel_matrix_blocks[i,j] = (u_ij,v_ij) 
		U[i][j] = u_ij
		V[i][j] = v_ij
		# print(u_ij)


# print(U)
# print(V)

fig, ax = plt.subplots()
q = ax.quiver(rows, coloumns, U, V)
ax.quiverkey(q, X=100, Y=100, U=1,
             label='Quiver key, length = 10', labelpos='E')

plt.show()





# rows = np.arange(0, no_blocks_rows, 1)
# coloumns = np.arange(0, no_blocks_columns, 1)
# U, V = np.meshgrid(rows, coloumns)


# sss = np.meshgrid(rows, coloumns)
# print(sss)
# print(U)
# print(V)
# fig, ax = plt.subplots()
# q = ax.quiver(rows, coloumns, U, V)
# ax.quiverkey(q, X=0.3, Y=1.1, U=10,
#              label='Quiver key, length = 10', labelpos='E')

# plt.show()





# 	block_start_limit_index = int(i * block_x_pixels_no)
# 	block_end_limit_index   = int(block_start_limit_index + block_x_pixels_no - 1)

# 	print("block_start_limit_index = ",block_start_limit_index)
# 	print("block_end_limit_index = ",block_end_limit_index)

# 	print("First row = ",uv[0])
# 	print("First pixel = ",uv[0][0])
# 	print("x_vel for First pixel = ",uv[0][0][0])
# 	print("y_vel for First pixel = ",uv[0][0][1])





# print("xy velocity for each pixel = ",uv)
# print("x vel for all pixels = uv[...,0] = ",uv[...,0])
# print("y vel for all pixels = uv[...,1] = ",uv[...,1])
# print("number of rows = len(uv) = ",len(uv))
# print("number of coloumns = len(uv[...,1][0]) = ",len(uv[...,1][0]))

# Calculate average velocity






# X = np.arange(-10, 10, 1)
# Y = np.arange(-10, 10, 1)
# U, V = np.meshgrid(X, Y)

# fig, ax = plt.subplots()
# q = ax.quiver(X, Y, U, V)
# ax.quiverkey(q, X=0.3, Y=1.1, U=10,
#              label='Quiver key, length = 10', labelpos='E')

# plt.show()