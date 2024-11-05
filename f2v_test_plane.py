import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from fig_format import fig_fontsizes
from fronttovof import Cube, get_plane, get_triangle, plot_polygons, \
    set_axis_transparent, add_axis


if __name__ == '__main__':
    fontsize_legend = fig_fontsizes["legend"]
    fontsize_label = fig_fontsizes["label"]
    fontsize_text = fig_fontsizes["text"]

    savefigs = not True

    # for plotting a cube
    axes = [1, 1, 1]
    data = np.ones(axes)

    # control Transparency
    alpha = 0.5
    colors = np.empty(axes + [4])

    colors[0] = [1, 1, 1, alpha]  # grey

    cub = Cube()
    n_tri = np.array([1., 1., 1.])
    alpha_tri = 0.5
    xo = alpha_tri * np.ones(3)
    p_tri = get_triangle(xo=xo, n=n_tri, l=3.)
    m = get_plane(p_tri)
    vol_ana = cub.cut_plane(m)
    print(cub.front2vof(p_tri, [[0, 1, 2]]), vol_ana)

    poly_cut, flag_cut = cub.clip3d(p_tri)
    figs = []
    axs = []
    for i in range(4):
        fig = plt.figure(figsize=(6.4, 6.4))
        fig.set_tight_layout(True)
        ax = fig.add_subplot(projection='3d')
        set_axis_transparent(ax)
        ax.set_xlabel(r'x', fontsize=fontsize_label)
        ax.set_ylabel(r'y', fontsize=fontsize_label)
        ax.set_zlabel(r'z', fontsize=fontsize_label)
        ax.view_init(elev=45, azim=45)
        # ax.set_xbound(-2.5, 2.5)
        # ax.set_ybound(-2.5, 2.5)
        # ax.set_zbound(-2.5, 2.5)
        ax.axes.set_xlim(left=-0.5, right=1.5)
        ax.axes.set_ylim(bottom=-0.5, top=1.5)
        ax.axes.set_zlim(bottom=-0.5, top=1.5)
        ax.set_aspect('equal')

        add_axis(ax, (-0., -0., -0.), (1.4, 1.4, 1.4), edgecolors='k',
                 arrow_length_ratio=0.1)
        # cub.plot_cube(ax)
        ax.voxels(data, facecolors=colors, edgecolors='grey')
        figs.append(fig)
        axs.append(ax)
    cub.plot_cut_procedure(axs)
    if savefigs:
        for i in range(4):
            figs[i].savefig("cut_schematic_{:d}.pdf".format(i + 1))
    # variation of volume with alpha
    nint = 10
    alphas = [np.linspace(0., 1., 3 * nint + 1),
              np.linspace(0., 1., 2 * nint + 1),
              np.linspace(0., 1., 2 * nint + 1)]
    m = np.ones(4)

    # single plane with different orientations
    n_test = np.array([[1., 1., 1.], [1., 1., 0.], [1., 0., 0.]])
    a_spe = [[1./3., 2./3.], [1./2.], []]
    for ni, n_tri in enumerate(n_test):
        vol_plane = []
        vol_plane_f2v = []
        vol_plane_f2v_invers = []

        m[:3] = n_tri[:]
        m_sum = np.sum(m[:-1])
        alpha = alphas[ni]
        for a in alpha:
            m[3] = a * m_sum
            vol_plane.append(cub.cut_plane(m))
            xo = np.ones(3) * a
            p_tri = get_triangle(xo=xo, n=m[:3], l=8.)
            vol_f2v = cub.front2vof(p_tri, [[0, 1, 2]])
            vol_plane_f2v.append(vol_f2v[0])

            vol_f2v = cub.front2vof(p_tri, [[2, 1, 0]])
            vol_plane_f2v_invers.append(vol_f2v[0])
            # print(xo)
            # print(vol_f2v, vol_plane[-1])

        vol_plane = np.array(vol_plane)
        fig_s, ax_s = plt.subplots(1, 1, figsize=(6.4, 6.4))
        ax_s.plot(alpha, vol_plane, 'r-o', markersize=8, fillstyle='none')
        ax_s.plot(alpha, vol_plane_f2v, 'b-+', markersize=8, fillstyle='none')
        ax_s.plot(alpha, 1. - vol_plane, 'g-o', markersize=8, fillstyle='none')
        ax_s.plot(alpha, vol_plane_f2v_invers, '-+', color='tab:orange', markersize=8, fillstyle='none')
        ax_s.legend([r'Ref., $\hat{\mathbf{n}} = \mathbf{m}$',
                     r'F2V, $\hat{\mathbf{n}} = \mathbf{m}$',
                     r'Ref., $\hat{\mathbf{n}} = -\mathbf{m}$',
                     r'F2V, $\hat{\mathbf{n}} = -\mathbf{m}$'], fontsize=fontsize_legend)
        ax_s.set_xlim((-0.1, 1.1))
        ax_s.set_ylim((-0.1, 1.1))
        ax_s.set_aspect('equal')
        ax_s.set_xlabel(r'$\alpha$', fontsize=fontsize_label)
        ax_s.set_ylabel(r'$V_{\Omega}(\alpha)$', fontsize=fontsize_label)
        ax_s.tick_params(labelsize=fontsize_label)
        ax_s.grid()
        for xs in a_spe[ni]:
            ax_s.axvline(xs, linestyle='--', color='k')
        fig_s.set_tight_layout(True)
        if savefigs:
            fig_s.savefig("f_a_{:.0f}_{:.0f}_{:.0f}.pdf".format(*n_tri))

    # test with multiple interface cut
    n_tri_1 = np.array([1., 1., 1.])
    n_tri_2 = -n_tri_1
    alpha_1 = 0.54
    alpha_2 = np.max([alpha_1 - 0.1, 0.])
    xo_1 = alpha_1 * np.ones(3)
    xo_2 = (1. - alpha_2) * np.ones(3)
    p_tri_1 = get_triangle(xo=xo_1, n=n_tri_1, l=4.)
    p_tri_2 = get_triangle(xo=xo_2, n=n_tri_2, l=4.)
    p_tri = np.vstack([p_tri_1, p_tri_2])

    m_1, m_2 = np.ones_like(m), np.ones_like(m)
    m_1[:3], m_2[:3] = n_tri_1[:], n_tri_1[:]
    m_sum = np.sum(m_1[:-1])
    m_1[3], m_2[3] = alpha_1 * m_sum, alpha_2 * m_sum
    vol_ana = cub.cut_plane(m_1) + cub.cut_plane(m_2)
    print(cub.front2vof(p_tri, [[0, 1, 2], [3, 4, 5]]), vol_ana)

    fig = plt.figure(figsize=(6.4, 6.4))
    fig.set_tight_layout(True)
    ax = fig.add_subplot(projection='3d')
    # set_axis_transparent(ax)
    ax.set_xlabel(r'x', fontsize=fontsize_label)
    ax.set_ylabel(r'y', fontsize=fontsize_label)
    ax.set_zlabel(r'z', fontsize=fontsize_label)
    ax.view_init(elev=45, azim=45)
    ax.set_aspect('equal')
    ax.set_xbound(-0.5, 1.5)
    ax.set_ybound(-0.5, 1.5)
    ax.set_zbound(-0.5, 1.5)

    ax.voxels(data, facecolors=colors, edgecolors='grey')
    fmts = ['r+-', 'b+--']

    colors_def = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]]
    alpha = 0.6
    plot_polygons(ax, cub.polygons_inside, linewidth=1,
                  facecolors=colors_def[0] + [alpha], edgecolors=colors_def[-1] + [alpha])

    # plot_polygen(p_tri_1, ax, fmt='r-')
    # poly_cut, flag_cut = cub.clip3d(p_tri_1)
    # cub.plot_cut_procedure(ax)
    # plot_polygen(p_tri_2, ax, fmt='r-')
    # poly_cut, flag_cut = cub.clip3d(p_tri_2)
    # cub.plot_cut_procedure(ax)

    vol_plane = []
    vol_plane_f2v = []
    vol_plane_f2v_invers = []
    alpha = []
    alphas = np.linspace(0., 1., 41)
    for alpha_1 in alphas:
        # fix the case with alpha_1 + alpha_2 = 1.
        alpha_2 = np.max([alpha_1 - 0.1, 0.])
        if alpha_1 + alpha_2 > 1.:
            break
        elif alpha_1 + alpha_2 == 1.:
            alpha_1 -= 1.e-3
            alpha_2 -= 1.e-3
        alpha.append(alpha_1)
        m_1[3], m_2[3] = alpha_1 * m_sum, alpha_2 * m_sum
        vol_plane.append(cub.cut_plane(m_1) + cub.cut_plane(m_2))

        xo_1 = alpha_1 * np.ones(3)
        xo_2 = (1. - alpha_2) * np.ones(3)
        p_tri_1 = get_triangle(xo=xo_1, n=n_tri_1, l=4.)
        p_tri_2 = get_triangle(xo=xo_2, n=n_tri_2, l=4.)
        p_tri = np.vstack([p_tri_1, p_tri_2])
        vol_f2v = cub.front2vof(p_tri, [[0, 1, 2], [3, 4, 5]])
        vol_plane_f2v.append(vol_f2v[0])

        vol_f2v = cub.front2vof(p_tri, [[2, 1, 0], [5, 4, 3]])
        vol_plane_f2v_invers.append(vol_f2v[0])
        # print(xo)
        # print(vol_f2v, vol_plane[-1])

    vol_plane = np.array(vol_plane)
    fig_s, ax_s = plt.subplots(1, 1, figsize=(6.4, 6.4))
    ax_s.plot(alpha, vol_plane, 'r-o', markersize=8, fillstyle='none')
    ax_s.plot(alpha, vol_plane_f2v, 'b-+', markersize=8, fillstyle='none')
    ax_s.plot(alpha, 1. - vol_plane, 'g-o', markersize=8, fillstyle='none')
    ax_s.plot(alpha, vol_plane_f2v_invers, '-+', color='tab:orange', markersize=8, fillstyle='none')
    ax_s.legend([r'Ref., $\hat{\mathbf{n}} = \mathbf{m}$',
                 r'F2V, $\hat{\mathbf{n}} = \mathbf{m}$',
                 r'Ref., $\hat{\mathbf{n}} = -\mathbf{m}$',
                 r'F2V, $\hat{\mathbf{n}} = -\mathbf{m}$'], fontsize=fontsize_legend)
    ax_s.set_aspect('equal')
    ax_s.set_xlim((-0.1, 1.1))
    ax_s.set_ylim((-0.1, 1.1))
    ax_s.set_xlabel(r'$\alpha$', fontsize=fontsize_label)
    ax_s.set_ylabel(r'$V_{\Omega}(\alpha)$', fontsize=fontsize_label)
    ax_s.tick_params(labelsize=fontsize_label)
    ax_s.grid()
    fig_s.set_tight_layout(True)
    if savefigs:
        fig_s.savefig("f_a_multiple.pdf")

    plt.show()