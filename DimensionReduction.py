import numpy


def x_to_z_projection_pca(WT, x_data, mu_array):

    z_array = list()

    for row in x_data:
        c_x = row-mu_array
        z_array.append(numpy.dot(WT, c_x))

    Z = numpy.array(z_array, dtype=numpy.float)

    return Z