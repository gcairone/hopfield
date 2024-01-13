import hopfield
import numpy as np
import preprocessing


list_patterns = []
# height and widht of image (pixels)
# number of parameters is O(n^4), avoid n greater than 70
n = 80
# create cat pattern
pattern1 = preprocessing.ImageMatrix("cat.PNG", n)
list_patterns.append(pattern1.matrix.flatten())
# create panda pattern
pattern2 = preprocessing.ImageMatrix("panda.PNG", n)
list_patterns.append(pattern2.matrix.flatten())
# create cow pattern
pattern3 = preprocessing.ImageMatrix("cow.jpg", n)
list_patterns.append(pattern3.matrix.flatten())
# create tree pattern
pattern4 = preprocessing.ImageMatrix("tree.jpg", n)
list_patterns.append(pattern4.matrix.flatten())

# create weight matrix that stores the two patterns
W = hopfield.create_matrix_patterns(list_patterns)
h = hopfield.HopfieldNetwork(W)

# initial state
x0 = pattern2.matrix
for i in range(int(2*n/3), n):
    for j in range(n):
        x0[i][j] = 1
# show initial state
# start animation
h.animate(x0)

