import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from fig_format import fig_fontsizes
from fronttovof import Cube, plot_polygons, set_axis_transparent, add_axis
import matplotlib.tri as mtri


def get_sphere(xo=(0., 0., 0.), r=1., dl=0.1, equal_phi=False):
    ntheta = int(pi * r / dl) + 1
    theta = np.linspace(0., pi, ntheta)
    xyz = np.array([[0., 0., 1.], [0., 0., -1.]])
    u = np.array([0., 1.])
    v = np.array([1., 1.])
    for _th in theta[1:-1]:
        sth = np.sin(_th)
        cth = np.cos(_th)
        if equal_phi:
            nphi = 2 * (ntheta - 1) + 1
        else:
            rt = r * np.sin(_th)
            nphi = max(int(2. * pi * rt / dl), 4) + 1
        phi = np.linspace(0., 2. * pi, nphi)
        _m = np.ones_like(phi)
        _xyz = np.vstack([sth * np.cos(phi), sth * np.sin(phi), cth * _m]).T
        xyz = np.vstack([xyz, _xyz])

        u = np.hstack([u, _m * _th / pi])
        v = np.hstack([v, phi / pi])
    xyz = r * xyz + np.array(xo)
    return xyz, u, v


if __name__ == '__main__':
    fontsize_legend = fig_fontsizes["legend"]
    fontsize_label = fig_fontsizes["label"]
    fontsize_text = fig_fontsizes["text"]

    cub = Cube()
    savefigs = not True

    nn = 2
    r = 0.4
    dl = r / 2**nn
    xyz, u, v = get_sphere(r=r, dl=dl, equal_phi=False)
    tri = mtri.Triangulation(u, v)
    nx = xyz.shape[0]
    ind_tri = tri.triangles

    # for plotting a cube
    axes = [1, 1, 1]
    data = np.ones(axes)

    # control Transparency
    alpha = 0.4
    colors = np.empty(axes + [4])

    colors[0] = [1, 1, 1, alpha]  # grey

    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('equal')
    ax.set_xbound(-0.2, 1.2)
    ax.set_ybound(-0.2, 1.2)
    ax.set_zbound(-0.2, 1.2)
    ax.set_xlabel(r'x', fontsize=fontsize_label)
    ax.set_ylabel(r'y', fontsize=fontsize_label)
    ax.set_zlabel(r'z', fontsize=fontsize_label)
    set_axis_transparent(ax)

    add_axis(ax, (-0., -0., -0.), (1.2, 1.2, 1.2), edgecolors='k',
             arrow_length_ratio=0.1)
    ax.voxels(data, facecolors=colors, edgecolors='grey')
    ax.view_init(elev=26, azim=133)
    fig.set_tight_layout(True)

    xcs = np.array([[1., 1., 1.], [1., 0., 1.]])
    xyz_final = np.copy(xyz)
    ind_final = np.copy(ind_tri)

    ns = 3
    for i in range(ns - 1):
        _xyz = xyz + xcs[i]
        _ind_tri = ind_tri + (i + 1) * nx

        xyz_final = np.vstack([xyz_final, _xyz])
        ind_final = np.vstack([ind_final, _ind_tri])

    # vol = cub.front2vof(xyz, tri.triangles)
    vol = cub.front2vof(xyz_final, ind_final)
    pi43 = 4. / 3. * np.pi
    vol_ana = ns * pi43 / 8. * r**3
    error = np.abs(vol[0] - vol_ana) / vol_ana
    print("V_num components: {:f}, {:f}, {:f}".format(vol[0], vol[1], vol[2]))
    print("V_num:{:f}, V_ana:{:f}, error:{:f}".format(vol[0], vol_ana, error))

    plot_polygons(ax, cub.polygons_inside, linewidth=1,
                  facecolors=[0, 0, 1, 1.], edgecolors=[0, 0, 0, 1.])

    fig_uv, ax_uv = plt.subplots(1, 1, figsize=(6.4, 6.4))
    fig_uv.set_tight_layout(True)
    ax_uv.set_xlabel(r'$\theta (\pi)$', fontsize=fontsize_label)
    ax_uv.set_ylabel(r'$\phi (\pi)$', fontsize=fontsize_label)

    ax_uv.triplot(u, v, 'ko-')
    ax_uv.tick_params(labelsize=fontsize_label)
    if savefigs:
        fig.savefig("sphere_{:d}.pdf".format(nn))
        fig_uv.savefig("sphere_uv_{:d}.pdf".format(nn))
    plt.show()