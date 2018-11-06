from Create_Data_files import *
# from LoadingDataFiles import x_to_z_projection_pca
import numpy
from ML_Visualizations import *
from DimensionReduction import x_to_z_projection_pca
from Process_CSV_to_Json import get_basic_stats


def make_rand_m(data_array, k):

    row_col = data_array.shape

    rows = row_col[0]
    cols = row_col[1]

    rand_chc = numpy.random.choice(rows, k)

    ret_l = list()

    for inst in rand_chc:
        ret_l.append(data_array[inst])

    mk = numpy.stack(ret_l)

    return rand_chc, mk


def calculate_bi(x, m):

    r_c = x.shape
    mr_mc = m.shape

    #print(r_c)
    #print(mr_mc)

    rows = r_c[0]

    bi_list = list()

    for i in range(0, rows):
        bi = [0]*mr_mc[0]
        min_l = list()
        save_j = 0
        for j in range(0, mr_mc[0]):
            #print('x at row {:d}'.format(i))
            #print(x[i].tolist())
            #print('m at row {:d}'.format(j))
            #print(m[j].tolist())
            dif = x[i]-m[j]
            #print('dif is {:d} vs {:d}'.format(i,j))
            #print(dif.tolist())
            norm = numpy.linalg.norm(dif)
            min_l.append(norm)
            #print('norm is {:f}'.format(norm))
            #print('')

        minimum = min(min_l)

        for id in range(len(min_l)):
            #print(id)
            if min_l[id] == minimum:
                bi[id] = 1
        bi_list.append(bi)
        #print('The min is {:f}'.format(minimum))
        #print('')
        #print('')

    return numpy.array(bi_list, dtype=numpy.float)


def get_new_m(x, m, bi):

    new_m = m.tolist()

    for i in range(len(m)):
        l = [0]*len(x[0])
        sm = numpy.array(l, dtype=numpy.float64)
        bs = 0
        for row in range(len(x)):

            # print('bval is {:f}'.format(bi[row][i]))
            bval = bi[row][i]
            if bval == 1:
                sm += x[row]
            bs += bval

        new_m[i] = sm/bs

    np_new_m = numpy.array(new_m)

    dif_m = np_new_m - m
    return np_new_m, dif_m


def k_means_clustering(x, k):
    r_c, mk = make_rand_m(x, k)

    print('')
    print('')
    print('')
    print('The random choice array is:')
    print(r_c.tolist())
    print('')
    print('')
    print('')
    avg_dif = 10000

    iter = 0

    while abs(avg_dif) > 0:
        bi_list = calculate_bi(x, mk)
        mk, dif_m = get_new_m(np_utk_data, mk[:, :], bi_list)
        avg_dif = numpy.mean(dif_m)
        iter += 1

    print('The average dif is now {:.2f}, iter {:d}'.format(avg_dif, iter))

    return mk, iter, bi_list



data_list = load_data_files()

utk_label = data_list[0]
utk_data = data_list[1]
np_utk_data = data_list[2]
imputated_data = data_list[3]
s_name = data_list[4]
head_l = data_list[5]
attribs = data_list[6]
stats = data_list[7]

num_obs = len(s_name)

u, s, vh = numpy.linalg.svd(np_utk_data, full_matrices=True, compute_uv=True)

v = numpy.transpose(vh)

vx = v[:, 0]
vy = v[:, 1]

mu_a = stats[0]
std_a = stats[1]
min_a = stats[2]
max_a = stats[3]

k = make_prop_o_var_plot(s, num_obs, show_it=False)


W = v[:, 0:k]

WT = numpy.transpose(W)

z_array = x_to_z_projection_pca(WT, np_utk_data, numpy.array(mu_a, dtype=numpy.float))


#basic_scatter_plot(vx, vy, 'w1_1', 'w2_2', 'w1 vs. w2', 'w1 vs. w2')

#z_scatter_plot(z_array, s_name,)

#r_c, mk = make_rand_m(np_utk_data, 4)

#print(r_c)


'''
for row in r_c:
    print(mk[i].tolist())
    print("")
    print(utk_data[row])
    print('')
    print('')
    i += 1
'''

km = 4

'''
bi_l has dimentsion num_obs x num_groupse
'''

# print(numpy.linalg.norm([2,4,3]))
found = True

while found:
    try:
        end_mk,  iter, bi_l = k_means_clustering(np_utk_data, km)
        um, sm, vhm = numpy.linalg.svd(end_mk, full_matrices=True, compute_uv=True)
        found = False
    except numpy.linalg.LinAlgError:
        found = True
        print('we have and error')

print('Shape of bi_list:')
print(bi_l.shape)
print('Shape of end mi:')
print(end_mk.shape)


vm = numpy.transpose(vhm)

vmx = vm[:, 0]
vmy = vm[:, 1]

k = make_prop_o_var_plot(sm, km, show_it=False)

print('The value of K should be {:d}'.format(k))

Wm = vm[:, 0:k]

WTm = numpy.transpose(Wm)

m_stats = get_basic_stats(end_mk)

mid_points = x_to_z_projection_pca(WTm, end_mk, numpy.array(m_stats[0], dtype=numpy.float))

print('Shape of end mid points:')
print(mid_points.shape)

colors_a = [[1, 0, 0],      # 0
            [0, 1, 0],      # 1
            [1, 1, 0],      # 2
            [0, 0, 1],      # 3
            [1, 0, 1],      # 4
            [0, 1, 1],      # 5
            [1, 1, 1],      # 6
            [0, 0, 0],      # 7
            [.5, 0, 1],     # 8
            [0, .5, 1],     # 9
            [.5, .5, 1],    # 10
            [.8, .3, .1],   # 11
            [.1, .3, .8],   # 12
            [.8, .5, 0],    # 13
            [.5, 0, .8],    # 14
            [.2, .7, .4],      # 15
            [.4, .7, 1],      # 16
            [.234, .541, .333],      # 17
            [.126, .9, .1],      # 18
            [.8, .7, .2]]   # 19


groups = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
          'V', 'W', 'X', 'Y', 'Z', 'aa', 'bb', 'cc', 'dd', 'ee']


k_cluster_title = 'K means clustering with {:d} groups for UTK peers data'.format(km)

'''
legend_titles = [['Group 1'],
                 ['Group 2'],
                 ['Group 3'],
                 ['Group 4']]
'''


legend_titles = ['Group 1',
                 'Group 2',
                 'Group 3',
                 'Group 4']

z_scatter_plot(mid_points, groups, show_it=False)
k_cluster_scatter_plot(z_array, s_name, mid_points, groups, colors=colors_a, b_list=bi_l, show_center=False,
                       title=k_cluster_title, legend=legend_titles)
#for row in end_mk.tolist():
#    print(row)



'''
bi_list = calculate_bi(np_utk_data, mk)

cnt = 0
for bi in bi_list:
    for c in range(len(mk.tolist())):
        print(cnt, bi[c])
    cnt += 1

print('mk')
print(mk.shape)
print(mk.tolist())
new_m, dif_m = get_new_m(np_utk_data, mk[:,:], bi_list)


print(dif_m)
print(mk.tolist())
print(new_m.tolist())
print(mk.shape)
'''

