import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import axes3d
from fig_format import fig_fontsizes
from fronttovof import Cube, set_axis_transparent, plot_polygons, add_axis
from f2v_test import get_sphere


if __name__ == '__main__':
    fontsize_legend = fig_fontsizes["legend"]
    fontsize_label = fig_fontsizes["label"]
    fontsize_text = fig_fontsizes["text"]

    cub = Cube()
    savefigs = True

    r = 1.
    dl = 0.25
    xyz, u, v = get_sphere(r=r, dl=dl, equal_phi=False)
    tri = mtri.Triangulation(u, v)

    # for plotting a cube
    axes = [1, 1, 1]
    data = np.ones(axes)

    # control Transparency
    alpha = 0.5
    colors = np.empty(axes + [4])

    colors[0] = [1, 1, 1, alpha]  # grey

    figs = []
    axs = []
    for i in range(2):
        fig = plt.figure(figsize=(6.4, 6.4))
        ax = fig.add_subplot(projection='3d')
        add_axis(ax, (-0., -0., -0.), (1.4, 1.4, 1.4), edgecolors='k',
                 arrow_length_ratio=0.1)
        set_axis_transparent(ax)
        ax.axes.set_xlim(left=-0.5, right=1.5)
        ax.axes.set_ylim(bottom=-0.5, top=1.5)
        ax.axes.set_zlim(bottom=-0.5, top=1.5)

        ax.voxels(data, facecolors=colors, edgecolors='grey')
        ax.set_aspect('equal')
        ax.set_xbound(-0.2, 1.2)
        ax.set_ybound(-0.2, 1.2)
        ax.set_zbound(-0.2, 1.2)

        ax.view_init(elev=17, azim=-12)
        fig.set_tight_layout(True)

        figs.append(fig)
        axs.append(ax)

    figs_x = []
    axs_x = []
    for i in range(2):
        fig_x, ax_x = plt.subplots(1, 1, figsize=(6.4, 6.4))
        fig_x.set_tight_layout(True)
        ax_x.set_xlim([-0.01, 1.01])
        ax_x.set_ylim([-0.01, 1.01])
        ax_x.set_frame_on(False)
        ax_x.tick_params(axis='both', which='both',
                         bottom=False, top=False, left=False, right=False,
                         labelbottom=False, labeltop=False,
                         labelleft=False, labelright=False)
        # plot the axis
        ax_x.quiver([0., 0.], [0., 0.], [1, 0], [0, 1], scale=4.)
        xo = -0.04
        ax_x.text(xo, xo, r"$o$", fontsize=fontsize_label)
        ax_x.text(0.25, 0.02, r"$\hat{\mathbf{e}}_{y}$", fontsize=fontsize_label)
        ax_x.text(0.02, 0.25, r"$\hat{\mathbf{e}}_{z}$", fontsize=fontsize_label)

        figs_x.append(fig_x)
        axs_x.append(ax_x)

    r0 = [2., 1.2]
    xc = np.array([[1.5, -1.4, -0.1], [1, 1.5, 1.5]])

    # red, green, blue, yellow
    colors_def = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]]

    corner = [[[1., 0., 0.], [1., 0., 1]], [[1., 1., 1.]]]
    patches = []
    patches_upper = []
    yz_polys = []
    for i in range(2):
        _xyz = r0[i] * xyz + xc[i]
        vol = cub.front2vof(_xyz, tri.triangles)

        # display the intersection on x = 1 face
        xyz_face = cub.get_points_on_face(1)
        xyz_face = np.vstack([xyz_face, corner[i]])[:, [1, 2]]
        hull = ConvexHull(xyz_face)
        xy = xyz_face[hull.vertices, :]
        yz_polys.append(xy)
        patches.append(Polygon(xy, closed=True))

        # for plots in 3D figure
        xy_upper = np.ones([xy.shape[0], 1])
        xy_upper = np.hstack([xy_upper, xy])
        patches_upper.append(xy_upper)

        # 3D polygons
        alpha_tri = 0.8
        alpha_poly = 0.6
        for _tri in cub.triangles_inside:
            plot_polygons(axs[0], [_tri], linewidth=1,
                          facecolors=colors_def[0] + [alpha_tri], edgecolors=colors_def[-1] + [alpha_tri])

        plot_polygons(axs[1], cub.polygons_inside, linewidth=1,
                      facecolors=colors_def[0] + [alpha_tri], edgecolors=colors_def[-1] + [alpha_tri])

    alpha_upper = 0.7
    plot_polygons(axs[1], patches_upper, linewidth=1,
                  facecolors=colors_def[2] + [alpha_upper], edgecolors=colors_def[2] + [alpha_upper])

    p_square = PatchCollection([Rectangle([0., 0.], 1., 1.)], facecolors=[0, 0, 0, 0], edgecolors=[0, 0, 0, 1])
    p = PatchCollection(patches, alpha=alpha_upper, facecolors='b', edgecolors='b')
    axs_x[0].add_collection(p)
    axs_x[0].add_collection(p_square)

    p_square = PatchCollection([Rectangle([0., 0.], 1., 1.)], facecolors=[0, 0, 0, 0], edgecolors=[0, 0, 0, 1])
    axs_x[1].add_collection(p_square)
    xcs = []
    dns = []
    for poly in yz_polys:
        for i in range(poly.shape[0]):
            i1 = (i + 1) % poly.shape[0]
            _y = np.array([poly[i, 0], poly[i1, 0]])
            _z = np.array([poly[i, 1], poly[i1, 1]])
            dt = [_y[1] - _y[0], _z[1] - _z[0]]
            dn = np.array([dt[1], -dt[0]])
            dn /= np.linalg.norm(dn)
            xcs.append([_y[1] + _y[0], _z[1] + _z[0]])
            dns.append(dn)
            if abs(dt[0]) < 1.e-12 or (abs(dt[1]) < 1.e-12 and abs(_y[0]) < 1.e-12):
                continue
            elif abs(dt[1]) < 1.e-12:
                fmt = 'r-'
            else:
                fmt = 'b-'
            axs_x[1].plot(_y, _z, fmt, linewidth=3)

    xcs = 0.5 * np.array(xcs)
    dns = np.array(dns)
    v_mask = [2, 11]
    # print(xcs[v_mask, 0], xcs[v_mask, 1])
    for i in range(2):
        axs_x[i].quiver(xcs[v_mask, 0], xcs[v_mask, 1], dns[v_mask, 0], dns[v_mask, 1], scale=10)
        axs_x[i].text(0.4, 0.72, r"$\hat{\mathbf{n}}_{\perp}$", fontsize=fontsize_label)

    if savefigs:
        figs[0].savefig("f2v_schematic_a.pdf")
        figs[1].savefig("f2v_schematic_b.pdf")
        figs_x[0].savefig("f2v_schematic_c.pdf")
        figs_x[1].savefig("f2v_schematic_d.pdf")
    plt.show()