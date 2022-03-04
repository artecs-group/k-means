# Synthetic_Data_Generator.py
# generate random points in a circular distribution

# import necessary pacakages
import random
import math
import numpy
import matplotlib

# when running on Linux
matplotlib.use('Agg')

# initialize the radii of clusters
cluster_r = [9, 9, 9, 9]

# initialize the centers of clusters    
cluster_dim1 = [40, 40, 60, 60]
cluster_dim2 = [40, 60, 40, 60]
# cluster_dim3 = [60, 60, 40, 40]
# cluster_dim4 = [60, 40, 60, 40]

# initialize the nb of points within each cluster and sum up the total nb of points
# nb_points = [12500000, 12500000, 12500000, 12500000]
nb_points = [100000, 100000]
sum_np = sum(nb_points)

# initialize arrays dedicated to store the coordinates and the cluster label of points
dim1 = [None] * sum_np
dim2 = [None] * sum_np
# dim3 = [None] * sum_np
# dim4 = [None] * sum_np
label = [None] * sum_np

# generate random points from one cluster to another
for i in range(len(nb_points)):
    # calculate the index offset of arrays when jumping from one cluster to another
    offset = sum(nb_points[:i])
    for j in range(nb_points[i]):
        # generate a random angle
        alpha = 2 * math.pi * random.random()
        # generate a random radius
        r = cluster_r[i] * math.sqrt(random.random())
        # calculate the coordinates and store them into coordinate arrays
        dim1[offset + j] = numpy.float32(r * math.cos(alpha) + cluster_dim1[i])
        dim2[offset + j] = numpy.float32(r * math.sin(alpha) + cluster_dim2[i])
        # generate a new random angle
        # alpha = 2 * math.pi * random.random()
        # # generate a new random radius
        # r = cluster_r[i] * math.sqrt(random.random())
        # dim3[offset + j] = numpy.float32(r * math.cos(alpha) + cluster_dim3[i])
        # dim4[offset + j] = numpy.float32(r * math.sin(alpha) + cluster_dim4[i])
        # generate a label for each point according to the cluster it belongs to
        label[offset + j] = i

# write the points into a text file
file = open("SyntheticDataset.txt", "w")
for index in range(sum_np):
    # file.write(str(dim1[index]) + "\t" + str(dim2[index]) + "\t" + str(dim3[index]) + "\t" + str(dim4[index]) + "\n")
    file.write(str(dim1[index]) + "\t" + str(dim2[index]) + "\n")
file.close()

# write the labels into a text file
file = open("Labels.txt", "w")
for index in range(sum_np):
    file.write(str(label[index]) + "\n")
file.close()
