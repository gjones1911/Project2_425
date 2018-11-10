from ML_Visualizations import *
from Process_CSV_to_Json import get_basic_stats


# -----------------------------------------Utility functions(make things easy---------------------------------------
# cheap print function
def p(str):
    print(str)
    return


# used to make sure we got all the groups we wanted
def check_grouping(groups):
    cnt = 0
    for group in groups:
        if len(group) == 0:
            p('Wrong number of groups found at {:d}'.format(cnt))
            return False
        cnt += 1
    return True


# makes a list where each row is a group and the entries in the list
# at this row are what observations belong to that group
def create_group_l(b, k):
    group_list = list()
    for i in range(k):
        group_list.append(list())
    for obs in range(len(b)):
        # look in current b row and find what group this observation
        # belongs to
        for n in range(len(b[obs])):
            if b[obs][n] == 1:
                #gnum = b[obs].index(1)
                group_list[n].append(obs)
        #p('the group number is {:d}'.format(gnum))
        #group_list[gnum].append(obs)
    #p('')
    #p('')
    #p('')
    #p('')
    #p('')
    for i in range(len(group_list)):
        if len(group_list[i]) == 0:
            p('No one in group {:d}'.format(i))

    return group_list


# random number generagor
def rng(start, stop, step):
    return numpy.random.choice(range(start, stop+step, step), 1)

# ------------------------------------------------------------------------------------------------------------------


# -----------------------------------------Reference Vector Functions-----------------------------------------------
# makes a random mk vector to be used for initial clustering
def make_rand_m(data_array, k):

    row_col = data_array.shape

    rows = row_col[0]
    cols = row_col[1]

    rand_chc = numpy.random.choice(rows, k, replace=False)

    #rand_chc = numpy.array([55,2,42,25,8,22,21,52,12])

    ret_l = list()

    for inst in rand_chc:
        ret_l.append(data_array[inst])

    #mk = numpy.stack(ret_l)
    mk = numpy.array(ret_l, dtype=numpy.float)

    return rand_chc, mk


# uses r_c to pick out vectors in x to make a set of reference vectors
def make_given_m(x, r_c):
    ret_l = list()
    for inst in r_c:
        ret_l.append(x[inst])
        # mk = numpy.stack(ret_l)
    mk = numpy.array(ret_l, dtype=numpy.float)
    return mk


# makes  a mean based set of referecne vectors
def make_mean_mod_m(x, k, mu_a, min_a, max_a):
    mu = numpy.array(mu_a, dtype=float)
    mn = numpy.array(min_a, dtype=float)
    mx = numpy.array(max_a, dtype=float)

    '''
    p(mu.tolist())

    p(min_a)
    p((mn/rng(50,75,5)).tolist())

    p(max_a)
    p((mx/rng(50,75,5)).tolist())
    '''
    ret_l = list()

    div = list([60])

    for i in range(k):
        div.append(div[-1]+5)

    for i in range(k):
        #ret_l.append(mu + mn/rng(50, 75, 5))
        #ret_l.append(mu + mn/div[i])
        ret_l.append(mu + mn/rng(5, 75, 5))

    return numpy.stack(ret_l)


def get_mid_p(glist, x):
    r_c = x.shape
    c = r_c[1]
    #xl = numpy.array([0]*c, dtype=numpy.float)
    yl = 0
    xlist = list()
    for i in glist:
        xlist.append(list(x[i].tolist()))
    xl = numpy.stack(xlist)
    x_stat = get_basic_stats(xl)
    return x_stat[0]


def make_g_m(x):

    g1 = [22]
    g2 = [3]
    g3 = [25,26]
    g4 =[19,14,27,12]
    g5 =[5,17,24,4,23,18,2]
    g6 = [44,6,38,40,7,46,1,11,12]
    g7 = [34,29,0,10,33,53,37,47,9,39,36,31,28,41,45,20,55]
    g8 = [32,56,43,49,51,35,21,48]
    g9 = [15,16,52,8,30,54,13,50]

    size = len(g1) + len(g2) + len(g3) + len(g4) + len(g5) + len(g6) + len(g7) + len(g8) + len(g9)
    print('the size is {:d}'.format(size))
    gtot = [g1, g2, g3, g4, g5, g6, g7, g8, g9]

    m_all = list()

    for g in gtot:
        m_all.append(get_mid_p(g, x))

    return numpy.array(m_all, dtype=numpy.float)

# -------------------------------------k means functions------------------------------------------------------


# calculate the minimum intercluster distance
def min_intercluster_distance(x, gs):

    mid = 100**5

    # get a group
    for g in range(len(gs)-1):

        # for every entry in this group
        # each entry is a point in the group
        for entry in gs[g]:
            #get a entry in group i
            x1 = x[entry]
            # look at every other group and get the distance between the entryies in that
            # group and group g and save the min
            for g2 in range(g+1, len(gs)):
                for entryj in gs[g2]:
                    x2 = x[entryj]
                    dis = numpy.linalg.norm(x1-x2)
                    if dis < mid:
                        mid = dis
    return mid


# returns maximun intracluster distance e.g.
# the largest distance between points in different clusters
def max_intracluster_distance(x, gs):

    maxid = 0

    for row in gs:
        for i in range(len(row)-1):
            for j in range(i+1, len(row)):
                dis = numpy.linalg.norm(x[i] - x[j])
                if dis > maxid:
                    maxid = dis
    return maxid


# calculate a label matrix
def calculate_bi(x, m):

    # N x d
    r_c = x.shape

    # K x d
    mr_mc = m.shape

    rows = r_c[0]

    bi_list = list()

    # go through every observation
    for i in range(0, rows):
        bi = [0]*mr_mc[0]
        min_l = list()
        # go through every reverence vector
        # calculating the difference between the
        # current observation vector and the jth
        # reverence vector. store the minimun length
        #for j in range(0, mr_mc[0]):
        for m_row in m:
            #dif = x[i]-m[j]
            dif = x[i]-m_row
            norm = numpy.linalg.norm(dif)
            min_l.append(norm)

        minimum = min(min_l)
        #print('the length of min_l is {:d}'.format(len(min_l)))
        #print('the length of bi is {:d}'.format(len(bi)))
        #p('The min was {:f}'.format(minimum))
        # add a 1 to the column to signify where this observation (row)
        # is in group id
        found = False
        for id in range(len(min_l)):
            if min_l[id] == minimum:
                bi[id] = 1
                found = True
        if not found:
            p('')
            p('')
            p('I DID NOT FIND THE MIN')
            p('')
            p('')

        print(bi)
        bi_list.append(bi)
    create_group_l(bi_list, len(bi))

    return numpy.array(bi_list, dtype=numpy.float)


# create new reference vectors
def get_new_m(x, m, bi):

    new_m = m.tolist()

    for i in range(len(m)):
        l = [0]*len(x[0])
        sm = numpy.array(l, dtype=numpy.float64)
        bs = 0
        in_g = 0
        for row in range(len(x)):
            # print('bval is {:f}'.format(bi[row][i]))
            bval = bi[row][i]
            if bval == 1:
                in_g += 1
                sm += x[row]
            bs += bval
        # print('------------------------------------------------------------------sm is now {:d}'.format(in_g))
        if in_g == 0:
            new_m[i] = m[i]
        else:
            new_m[i] = sm/bs

    np_new_m = numpy.array(new_m, dtype=numpy.float)

    # save how much the new and old differ
    dif_m = abs(np_new_m - m)
    return np_new_m, dif_m




# def k_means_clustering(x, k, mu_a, min_a, max_a):
def k_means_clustering(x, k, init_m=[], m_init_type=0, mu_a=numpy.array([]),
                       min_a=numpy.array([]), max_a=numpy.array([])):
    if len(init_m) ==0:
        if m_init_type == 0:
            p('initialize m to random k elements of x')
            r_c, mk = make_rand_m(x, k)
        elif m_init_type == 1:
            r_c = []
            mk = make_g_m(x)
        elif m_init_type == 2:
            p('initialize m to modified mean k elements of x')
            r_c = []
            mk = make_mean_mod_m(x,k,mu_a, min_a, max_a)
    else:
        mk = init_m

    '''
    rc = mk.shape
    print('----------------------------------------->', mk.tolist())
    print('---------------------------------------->{:d}'.format(rc[0]))
    print('---------------------------------------->{:d}'.format(rc[1]))
    '''
    #print('')
    #print('')
    #print('The random choice array is:')
    #print(r_c.tolist())
    #print('')
    #print('')

    avg_dif = 10000

    iter = 0

    #while abs(avg_dif) > 0:
    while abs(avg_dif) > 1:
        bi_list = calculate_bi(x, mk)
        mk, dif_m = get_new_m(x, mk[:, :], bi_list)
        #avg_dif = numpy.mean(dif_m)
        avg_dif = numpy.sum(dif_m)
        iter += 1
        #print('The average dif is now {:.2f}, iter {:d}'.format(avg_dif, iter))
        #print('The dif is :')
        #print(dif_m.tolist())

    print('The average dif is now {:.2f}, iter {:d}'.format(avg_dif, iter))
    #create_group_l(list(bi_list.tolist()), k)

    return mk, iter, bi_list


def reduce_x(W, z, mu):

    l = []
    cnt = 0
    x_array = list()

    for row in z:
        res = numpy.dot(W, row)
        x_array.append(res + mu)

    x = numpy.array(x_array, dtype=numpy.float)

    return x


def get_si(x, m, h):

    s = list()
    S = numpy.array([0]*len(m), dtype=numpy.float)
    for i in range(len(m)):
        st = 0
        sb = 0
        for t in range(len(x)):
            st += h[t][i]*(numpy.dot((x[t]-m[i]), numpy.transpose(x[t]-m[i])))
            sb += h[t][i]
        if sb == 0:
            S[i] = 0
        else:
            S[i] = st/sb

    return numpy.array(S, dtype=numpy.float)


def calculate_pi_i(h, N):
    pi_i = list()

    r_c = h.shape

    for i in range(r_c[1]):
        sm = 0
        for t in range(N):
            sm += h[t][i]
        pi_i.append(sm/N)
    return numpy.array(pi_i, dtype=numpy.float)


def e_step(S, x, m, pi_i):
    ht = list()
    for t in range(len(x)):
        sb = 0
        hi = list()
        for j in range(len(m)):
            dif = x[t] - m[j]
            s = S[j]
            if s == 0:
                s = .00001
            a1 = -.5 * numpy.transpose(dif)
            b1 = (1 / s) * dif
            sb += pi_i[j] * (abs(s)**(-.5)) * numpy.exp( numpy.dot(a1, b1) )
        for i in range(len(m)):
            s = S[i]
            if s == 0:
                s = 1
            dif = x[t] - m[i]
            a1 = -.5*numpy.transpose(dif)
            b1 = (1/s) * dif
            val = pi_i[i] * (abs(s)**(-.5)) * numpy.exp(numpy.dot(a1, b1) )
            hi.append(val/sb)
        ht.append(hi)

    return numpy.array(ht, dtype=numpy.float)


def calculate_mi(h, x):
    m = list()

    for i in range(len(h[0])):

        sm = numpy.array([], dtype=numpy.float)

        tp = numpy.array([0] * len(x[0]), dtype=numpy.float)

        bt = 0

        for t in range(len(x)):

            #print(h[t][i])
            tp += h[t][i] * x[t]
            bt += h[t][i]

        m.append(tp / bt)

    return numpy.array(m, dtype=numpy.float)


def expectation_maximization(x, m, h):
    # estimate S
    S = get_si(x, m, h)

    #calculate pi_i
    pi_i = calculate_pi_i(h, len(x))

    hi = e_step(S, x, m, pi_i)

    mi = calculate_mi(hi, x)
    dif = 0
    ret_d = 100000



    while abs(ret_d) > 1:
        mold = numpy.array(list(mi.tolist()))

        dif_old = dif

        s = get_si(x, mi, hi)

        # calculate pi_i
        pi_i = calculate_pi_i(hi, len(x))

        hi = e_step(s, x, mi, pi_i)

        mi = calculate_mi(hi, x)

        dif = numpy.mean(mold - mi, dtype=numpy.float)

        ret_d = numpy.around(abs(dif_old - dif), 1)
        #print('The dif is now {:f}'.format(ret_d))

    return mi, hi


def x_reduced(u, s, vt):

    us = numpy.dot(u,s)

    return numpy.dot(us, vt)

def get_EM_grouping(h, k):

    hl = h.tolist()

    g = list()
    #for i in range(k):
    #    l = list([0]*k)
    #    g.append(l)

    for row in range(len(hl)):
        mx = max(hl[row])
        l = list([0]*k)
        g.append(l)

        g[-1][hl[row].index(mx)] = 1

    return numpy.array(g, dtype=numpy.float)


def show_grouping(grouping):

    r = 1
    for row in grouping:
        print('Group {:d}: '.format(r))
        p(row)
        p('')
        r += 1
    return

def make_label_list(size):
    groups = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'k', 'L', 'M', 'N', 'O', 'P',
               'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    ret_l = [''] * size

    r = list

    for i in range(size):

        ret_l[i] = ret_l[i%len(groups)] + groups[i%len(groups)]

    return ret_l

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------







'''
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

k = make_prop_o_var_plot(s, num_obs, show_it=False, last_plot=False)
p('The original k:')
p('The best k for original data is {:d}'.format(k))
ko = k
W = v[:, 0:k]
WT = numpy.transpose(W)

W2 = v[:, 0:2]
WT2 = numpy.transpose(W2)


z_array = x_to_z_projection_pca(WT, np_utk_data, numpy.array(mu_a, dtype=numpy.float))
z_array2 = x_to_z_projection_pca(WT2, np_utk_data, numpy.array(mu_a, dtype=numpy.float))

x_red = reduce_x(W, z_array, numpy.array(mu_a, dtype=numpy.float))
#red_x = x_reduced(u, s, WT)

#xr_r_c = red_x.shape

#p('The rows are {:d} and cols {:d}'.format(xr_r_c[0], xr_r_c[1]))

p('The transformed x into z is:')
print(z_array.shape)


p('The reduced x is:')
print(x_red.shape)

#basic_scatter_plot(vx, vy, 'w1_1', 'w2_2', 'w1 vs. w2', 'w1 vs. w2')

z_title = 'The first 2 PC for the transformed data into {:d} dimension'.format(k)

z_scatter_plot(z_array, s_name,title=z_title, last=False, show_it=False)

#r_c, mk = make_rand_m(np_utk_data, 4)

# --------------------------------------------------------------------------------------------------------------
mg = make_g_m(np_utk_data)
print("find me")
print(mg.shape)
#print(r_c)


for row in r_c:
    print(mk[i].tolist())
    print("")
    print(utk_data[row])
    print('')
    print('')
    i += 1
#----------------------------------------------KKKKKKKKKKKKKKK--------------------------------------------------
km = 9
#----------------------------------------------KKKKKKKKKKKKKKK--------------------------------------------------

bi_l has dimentsion num_obs x num_groupse




# print(numpy.linalg.norm([2,4,3]))
found = False

while not found:
    try:
        #rcc, init_mk = make_rand_m(np_utk_data, km)
        #end_mk,  iter, bi_l = k_means_clustering(np_utk_data, km, init_m=init_mk)
        end_mk,  iter, bi_l = k_means_clustering(np_utk_data, km)
        grps = create_group_l(list(bi_l.tolist()), km)
        # end_mk,  iter, bi_l = k_means_clustering(np_utk_data, km, mu_a, min_a, max_a)
        um, sm, vhm = numpy.linalg.svd(end_mk, full_matrices=True, compute_uv=True)
        found = check_grouping(grps)
        mid = min_intercluster_distance(np_utk_data, grps)
        mxid = max_intracluster_distance(np_utk_data, grps)
        p('The min intercluster distance is : {:f}'.format(mid))
        p('The max intracluster distance is : {:f}'.format(mxid))
        p('the dun index is: {:f}'.format(mid/mxid))
    except numpy.linalg.LinAlgError:
        found = False
        print('we have and error')



print('__________________________________________HERE--------------------------------------------------------------')
print('__________________________________________HERE--------------------------------------------------------------')
print('__________________________________________HERE--------------------------------------------------------------')
S = get_si(np_utk_data, end_mk, bi_l)
print(S.shape)
print(S)
print('__________________________________________HERE--------------------------------------------------------------')
print('__________________________________________HERE--------------------------------------------------------------')
print('__________________________________________HERE--------------------------------------------------------------')
print('')
print('__________________________________________HERE2--------------------------------------------------------------')
print('__________________________________________HERE2--------------------------------------------------------------')
print('__________________________________________HERE2--------------------------------------------------------------')

print('__________________________________________HERE2a--------------------------------------------------------------')
print('__________________________________________HERE2a--------------------------------------------------------------')
print('__________________________________________HERE2a--------------------------------------------------------------')

pi_i = calculate_pi_i(bi_l, len(np_utk_data))
print(pi_i.shape)
print(pi_i)
print('__________________________________________HERE2a--------------------------------------------------------------')
print('__________________________________________HERE2a--------------------------------------------------------------')
print('__________________________________________HERE2b--------------------------------------------------------------')

h = e_step(S, np_utk_data, end_mk, pi_i)
print(h.shape)
print(h)
print('__________________________________________HERE2b--------------------------------------------------------------')
print('__________________________________________HERE2b--------------------------------------------------------------')
print('__________________________________________HERE2b--------------------------------------------------------------')


mnew, hnew = expectation_maximization(np_utk_data, end_mk, bi_l)

p('')
p('')
p('')
p('')
p('')
p('new M ')
print(mnew.shape)
print(mnew)
p('')
p('')
p('')
p('')
p('')
p('')
p('')
p('')
p('')
p('')
p('new h ')
print(hnew.shape)
print(hnew)
p('')
p('')
p('')
p('')
p('')

gbb = get_EM_grouping(hnew, km)

for i in range(len(gbb)):
    print('Group {:d}'.format(i))
    print(gbb[i])

grps = create_group_l(list(bi_l.tolist()), km)

print('---------------------------------------------------------------------------------------------The groupings are:')

cnt = 0
for group in grps:
    if len(group) > 0:
        p('group {:d}:'.format(cnt+1))
        p(group)
        p('')
    cnt += 1

print('Shape of bi_list:')
print(bi_l.shape)
print('Shape of end mi:')
print(end_mk.shape)


vm = numpy.transpose(vhm)

vmx = vm[:, 0]
vmy = vm[:, 1]

k = make_prop_o_var_plot(sm, km, show_it=False)

print('The value of K sh ould be {:d}'.format(k))

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
            [0, 0, 0],      # 4
            [1, 0, 1],      # 5
            [0, 1, 1],      # 6
            [.7, .5, .2],    # 7
            [.5, .2, .1],   # 8
            [.5, .5, .5],   # 9
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

legend_titles = [['Group 1'],
                 ['Group 2'],
                 ['Group 3'],
                 ['Group 4']]


legend_titles = ['Group 1',
                 'Group 2',
                 'Group 3',
                 'Group 4']

z_scatter_plot(mid_points, groups, show_it=False)
k_cluster_scatter_plot(z_array, s_name, mid_points, groups, colors=colors_a, b_list=bi_l, show_center=False,
                       title=k_cluster_title, legend=legend_titles, last=False)

#km = 5
found = False

while not found:
    try:
        #init_mk = make_given_m(z_array, rcc)
        #end_mk2,  iter2, bi_l2 = k_means_clustering(z_array, km, init_m=init_mk)
        end_mk2,  iter2, bi_l2 = k_means_clustering(z_array, km)
        grps2 = create_group_l(list(bi_l2.tolist()), km)
        # end_mk,  iter, bi_l = k_means_clustering(np_utk_data, km, mu_a, min_a, max_a)
        um2, sm2, vhm2 = numpy.linalg.svd(end_mk, full_matrices=True, compute_uv=True)
        found = check_grouping(grps2)
        mid = min_intercluster_distance(z_array, grps2)
        mxid = max_intracluster_distance(z_array, grps2)
        p('The min intercluster distance is : {:f}'.format(mid))
        p('The max intracluster distance is : {:f}'.format(mxid))
        p('the dun index is: {:f}'.format(mid/mxid))
    except numpy.linalg.LinAlgError:
        found = False
        print('we have and error')

#grps = create_group_l(list(bi_l.tolist()), km)

print('-----------------------------------------------------------------------------------------The 2 groupings are:')

cnt = 0
for group in grps2:
    if len(group) > 0:
        p('group {:d}:'.format(cnt+1))
        p(group)
        p('')
    cnt += 1

title = 'K Clustering four projected data {:d} PC'.format(ko)
#k_cluster_scatter_plot(z_array, s_name, mid_points, groups, colors=colors_a, b_list=bi_l2, show_center=False,
#                       title=title, legend=legend_titles, last=False, show_it=False)

k_cluster_scatter_plot(z_array, s_name, mid_points, groups, colors=colors_a, b_list=gbb, show_center=False,
                       title=title, legend=legend_titles, last=False, show_it=False)

found = False

while not found:
    try:
        #init_mk = make_given_m(z_array, rcc)
        #end_mk2,  iter2, bi_l2 = k_means_clustering(z_array, km, init_m=init_mk)
        end_mk2,  iter2, bi_l2 = k_means_clustering(z_array2, km)
        grps2 = create_group_l(list(bi_l2.tolist()), km)
        # end_mk,  iter, bi_l = k_means_clustering(np_utk_data, km, mu_a, min_a, max_a)
        um2, sm2, vhm2 = numpy.linalg.svd(end_mk, full_matrices=True, compute_uv=True)
        found = check_grouping(grps2)
        mid = min_intercluster_distance(z_array2, grps2)
        mxid = max_intracluster_distance(z_array2, grps2)
        p('The min intercluster distance is : {:f}'.format(mid))
        p('The max intracluster distance is : {:f}'.format(mxid))
        p('the dun index is: {:f}'.format(mid/mxid))
    except numpy.linalg.LinAlgError:
        found = False
        print('we have and error')

#grps = create_group_l(list(bi_l.tolist()), km)

print('-----------------------------------------------------------------------------------------The 2 groupings are:')

cnt = 0
for group in grps2:
    if len(group) > 0:
        p('group {:d}:'.format(cnt+1))
        p(group)
        p('')
    cnt += 1

title = 'K Clustering for projected data 2 PC'
k_cluster_scatter_plot(z_array2, s_name, mid_points, groups, colors=colors_a, b_list=bi_l2, show_center=False,
                       title=title, legend=legend_titles, last=True, show_it=True)

#get_si(np_utk_data, 0, 0)

#for row in end_mk.tolist():
#    print(row)

#make_g_m(z_array)

#mtest = make_mean_mod_m(np_utk_data, 9, mu_a, min_a, max_a)

#r_c = mtest.shape

#print('the rows are {:d} and the cols are {:d}'.format(r_c[0], r_c[1]))

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

#ta = numpy.around([2,1,3])
#print(ta -1)

'''
