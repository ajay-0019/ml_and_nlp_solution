import numpy as np
import random


def numpy_alternate_sort(arr):
    sorted_arr = np.sort(arr)
    result = []
    left = 0
    right = len(sorted_arr) - 1

    for i in range(len(sorted_arr)):
        if i % 2 == 0:
            result.append(sorted_arr[left])
            left += 1
        else:
            result.append(sorted_arr[right])
            right -= 1
    return np.array(result)
        
           


size=20
arr = np.random.uniform(0, 10, size)
rounded_arr = np.round(arr, 2)
print(arr)
print(f"All element after round to 2 decimal place: {rounded_arr}")

median=np.std(rounded_arr)
min=np.min(rounded_arr)
max=np.max(rounded_arr)


print(f"Median of the rounded array: {median}")
print(f"Max of the rounded array: {max}")
print(f"Min of the rounded array: {min}")


for i in range(size):
    if rounded_arr[i]<5:
        rounded_arr[i]=rounded_arr[i]**2

print(f"after squaring the elements less than 5: {rounded_arr}")        

print(f" alternate sort array in term of of max min: {numpy_alternate_sort(rounded_arr)}")
