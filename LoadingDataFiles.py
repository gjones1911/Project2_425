from Create_Data_files import *
from ML_Visualizations import *


def x_to_z_projection_pca(WT, x_data, mu_array):

    ret_list = list()

    #wt_np = numpy.array(Wt, dtype=numpy.float)
    #x_np = numpy.array(x_data, dtype=numpy.float)
    #mu_np = numpy.array(mu_array, dtype=numpy.float)


    centered_x = x_data[0] - mu_array

    print('Centered x')
    print(centered_x.shape)
    print(centered_x.tolist())
    print('')

    print('Shape of W')
    print(WT.shape)
    print(WT.tolist())
    print('')

    print('Shape of x')
    print(x_data.shape)
    print('')

    print('Mu array')
    print(mu_array.shape)
    print('')

    z_array = list()

    for row in x_data:
        c_x = row-mu_array
        z_array.append(numpy.dot(WT, c_x))

    Z = numpy.array(z_array, dtype=numpy.float)
    print('Z array')
    print(Z.shape)
    print(Z.tolist())

    return Z


utk_data_file = 'UTK-peers_data.dt'

data_list = load_data_files()

'''
ret_list.append(utk_labels)
ret_list.append(utk_data)
ret_list.append(np_utk_data)
ret_list.append(imputated_d)
ret_list.append(s_name)
ret_list.append(headers)
ret_list.append(attribs_l)
ret_list.append(basic_stats)
'''

utk_label = data_list[0]
utk_data = data_list[1]
np_utk_data = data_list[2]
imputated_data = data_list[3]
s_name = data_list[4]
head_l = data_list[5]
attribs = data_list[6]
stats = data_list[7]

num_obs = len(s_name)

'''
print('Labels:')
print(utk_label)
print('')
print('data')
for i in range(len(utk_data)):
    print(utk_data[i])
print('')
print('numpy data:')
for i in range(len(np_utk_data.tolist())):
    print(np_utk_data.tolist()[i])
print('')
print('imputated data:')
for i in range(len(imputated_data)):
    print(imputated_data[i])
print('')
print('school names')
print(s_name)
print('')
'''


u, s, vh = numpy.linalg.svd(np_utk_data, full_matrices=True, compute_uv=True)

v = numpy.transpose(vh)

vx = v[:, 0]
vy = v[:, 1]

mu_a = stats[0]
std_a = stats[1]
min_a = stats[2]
max_a = stats[3]

#make_scree_plot_usv(s, num_obs)

k = make_prop_o_var_plot(s, num_obs)

print('The value of K should be {:d}'.format(k))

W = v[:, 0:k]

WT = numpy.transpose(W)

z_array = x_to_z_projection_pca(WT, np_utk_data, numpy.array(mu_a, dtype=numpy.float))

#basic_scatter_plot(vx, vy, 'w1_1', 'w2_2', 'w1 vs. w2', 'w1 vs. w2')

z_scatter_plot(z_array, s_name)
