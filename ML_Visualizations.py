import matplotlib
import matplotlib.pyplot as plt
import numpy


def make_scree_graph_data(np_data_array, show_it=True):
    u, s, vh = numpy.linalg.svd(np_data_array, full_matrices=True, compute_uv=True)
    v = numpy.transpose(vh)

    sum_s = sum(s.tolist())
    s_sum = numpy.cumsum(s)[-1]
    print('shape of s')
    print(s.shape)
    obs_var = np_data_array.shape
    num_obs = obs_var[0]
    num_var = obs_var[1]

    print('There are {:d} observations and {:d} variables or attributes'.format(num_obs, num_var))

    eigen_vals = s ** 2 /s_sum

    single_vals = numpy.arange(num_obs)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(single_vals, eigen_vals, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    # I don't like the default legend so I typically make mine like below, e.g.
    # with smaller fonts and a bit transparent so I do not cover up data, and make
    # it moveable by the viewer in case upper-right is a bad place for it
    leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)

    if show_it:
        plt.show()
    return u, s, vh, v


def make_scree_plot_usv(s, num_obs, show_it=True, last=False):
    sum_s = sum(s.tolist())
    s_sum = numpy.cumsum(s)[-1]
    #print('shape of s')
    #print('There are {:d} observations and {:d} variables or attributes'.format(num_obs, num_var))

    eigen_vals = s ** 2 / s_sum

    single_vals = numpy.arange(num_obs)

    if show_it:
        fig = plt.figure(figsize=(8, 5))
        plt.plot(single_vals, eigen_vals, 'ro-', linewidth=2)
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3,
                         shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                         markerscale=0.4)
        leg.get_frame().set_alpha(0.4)
        leg.draggable(state=True)
    if last:
        plt.show()
    return


def make_prop_o_var_plot(s, num_obs, show_it=True, last_plot = True):

    sum_s = sum(s.tolist())

    ss = s**2

    sum_ss = sum(ss.tolist())

    prop_list = list()

    found = False

    k = 0

    for i in range(1, num_obs+1):
        #print(s[0:i].tolist())
        #print('')
        #perct = sum(ss[0:i]) / sum_ss
        perct = sum(s[0:i]) / sum_s
        #perct = sum(s[0:i]) / sum_ss
        # print(perct)
        prop_list.append(perct)

        #print('Prop. of Var. {:f} with k = {:d}'.format(perct, i))
        #print(prop_list[-1])

    k_val = 1
    for perct in prop_list:
        if perct >= .90:
            if perct > .90:
                k_val -= 1
            break
        k_val += 1

    single_vals = numpy.arange(1,num_obs+1)

    if show_it:
        fig = plt.figure(figsize=(8, 5))
        plt.plot(single_vals, prop_list, 'ro-', linewidth=2)
        plt.title('Proportion of Variance, K should be {:d}'.format(k_val))
        plt.xlabel('Eigenvectors')
        plt.ylabel('Prop. of var.')
        leg = plt.legend(['Eigenvectors vs. Prop. of Var.'], loc='best', borderpad=0.3,
                         shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                         markerscale=0.4)
        leg.get_frame().set_alpha(0.4)
        leg.draggable(state=True)

        if last_plot:
            plt.show()

    return k_val


def dual_scree_prop_var(s, num_obs):
    sum_s = sum(s.tolist())

    eigen_vals = s ** 2 /sum_s

    single_vals = numpy.arange(num_obs)

    prop_list = list()

    for i in range(1, num_obs + 1):
        prop_list.append(sum(s[0:i].tolist()) / sum_s)


    fig, axs = plt.subplots(2,1)
    #plt.figure(1)
    #fig = plt.figure(figsize=(8, 5))
    axs[0].plot(single_vals, eigen_vals, 'ro-',  linewidth=2)
    plt.title('Scree Plot')
    #axs[0].title('Scree Plot')
    axs[0].set_xlabel('Principal Component')
    axs[0].set_ylabel('Eigenvalue')
    leg = axs[0].legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)

    #plt.figure(2)
    axs[1].plot(single_vals, prop_list, 'go-', linewidth=2)
    #plt.title('Proportion of Variance')
    axs[1].set_xlabel('Eigenvectors')
    axs[1].set_ylabel('Prop. of var.')
    leg = plt.legend(['Eigenvectors vs. Prop. of Var.'], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)
    '''
    plt.subplot(2,2,1)
    #fig = plt.figure(figsize=(8, 5))
    plt.plot(single_vals, prop_list, 'ro-', linewidth=2)
    plt.title('Proportion of Variance')
    plt.xlabel('Eigenvectors')
    plt.ylabel('Prop. of var.')
    leg = plt.legend(['Eigenvectors vs. Prop. of Var.'], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)
    '''

    plt.show()

    return


def basic_scatter_plot(x, y, x_label, y_label, title, legend):

    fig = plt.figure(figsize=(8, 5))
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    leg = plt.legend([legend], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)
    plt.show()
    return


def z_scatter_plot(Z, schools, x_label='z1', y_label='z2', title='PC1 vs. PC2 for all Observations',
                   legend='z1 vs. z2', show_it=True, last=False, point_size=20, color=[[0,0,0]]):

    if show_it:
        fig = plt.figure(figsize=(8, 5))
        i = 0
        for row in Z:
            z1 = row[0]
            z2 = row[1]
            plt.scatter(z1, z2, s=point_size, c=color)
            plt.annotate(schools.index(schools[i]), (z1, z2))
            i += 1

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        leg = plt.legend([legend], loc='best', borderpad=0.3,
                         shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                         markerscale=0.4)
        leg.get_frame().set_alpha(0.4)
        leg.draggable(state=True)

        if not last:
            plt.show()
    return


def k_cluster_scatter_plot(Z, schools, mid, groups, x_label='x1', y_label='x2', title='PC1 vs. PC2 for all Observations',
                           legend='z1 vs. z2', show_it=True, colors=[[.8, .4, .2]], b_list=[] ,g_ids = {},
                           show_center=True, last=False, groups_l=[]):


    row_mid = len(mid)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    i = 0
    for row in Z:
        z1 = row[0]
        z2 = row[1]

        if len(b_list) > 0:
            l = list(b_list[i].tolist())
            midx = l.index(1) % len(colors)
            #print(l.index(1))

        ax.scatter(z1, z2, s=20, c=colors[midx])

        if len(groups_l) > 0:
            ax.annotate(groups_l[i], (z1, z2))
        else:
            ax.annotate(schools.index(schools[i]), (z1, z2))
        i += 1
    if show_center:

        r_c = b_list.shape

        bii = list()
        '''
        for i in range(r_c[0])
        
        for 
        '''
        i = 0
        #for row in mid:
        for row, color in zip(mid, colors):
            m1 = row[0]
            m2 = row[1]
            ax.scatter(m1, m2, s=3000, c=color, alpha=.1)
            #ax.annotate(groups[i], (m1, m2), arrowprops=dict(facecolor='black', shrink=1.05))
            ax.annotate(groups[i], (m1, m2))
            i += 1

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #leg = plt.legend(legend, loc='best', borderpad=0.3,
    #                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
    #                 markerscale=0.4, )
    #leg.get_frame().set_alpha(0.4)
    #leg.draggable(state=True)
    if last:
        plt.show()
    return
