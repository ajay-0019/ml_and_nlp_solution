import numpy as np
import random

def numpy_boundary_transversal(matrix):
    result = []
    rows = len(matrix)
    cols = len(matrix[0])

    for j in range(cols):
        result.append(int(matrix[0][j]))

    for i in range(1, rows - 1):
        result.append(int(matrix[i][cols - 1]))

    if rows > 1:
        for j in range(cols - 1, -1, -1):   
            result.append(int(matrix[rows - 1][j]))

    if cols > 1:
        for i in range(rows - 2, 0, -1):
            result.append(int(matrix[i][0]))

    return result


Matrix =np.empty((5,4), dtype=int)
rows=5
cols=4
sum=0

for i in range(rows):
    each_row_max=0
    for j in range(cols):
        Matrix[i][j]=random.randint(1,50)
        sum+=Matrix[i][j]



print(Matrix)
anti_diag = np.fliplr(Matrix).diagonal()
print(f"Anti-Diagonal element are : {anti_diag}")

mean_val=np.mean(Matrix)
print(f"Mean value of Matrix: {mean_val}")

row_max = np.max(Matrix, axis=1)  # max of each row
print("Max of each row:", row_max)

Matrix2 = np.random.uniform(0, mean_val, size=Matrix.shape)

print(Matrix2)
print(f"Boundary element list: {numpy_boundary_transversal(Matrix)}")             
