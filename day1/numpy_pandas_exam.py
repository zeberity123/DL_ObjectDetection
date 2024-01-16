import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

arr = np.array([[2,3,4],
          [5,6,7]])

print(arr)
print(type(arr))
print(arr.shape , arr.ndim)

arr = arr.reshape((3,2))
print(arr.shape , arr.ndim)

# mydf = pd.DataFrame(arr, index=['one','two'],
#                     columns=list('ABC'))
# print(mydf)
# print(mydf.index)
# print(mydf.columns)
# print(mydf.values)
# print(mydf.loc['two':'two'])

