import numpy as np
import os


path = 'C:\Windows'
dirs = os.listdir(path)

listPath = path
for file in dirs:
    if 'addins' in file:
        listPath = listPath + '\\addins'

for file in os.listdir(listPath):
    print(file)
print(listPath)
# R = np.array([[123,123],
#               ['123','123'],
#               [123.32,123.23],
#               [True,False]])
#
# print(np.shape(R))
# print(R[0][1])
# print(len(np.shape(R)))
