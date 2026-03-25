import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.tri as mtri
from fig_format import fig_fontsizes
from fronttovof import Cube, plot_polygons, set_axis_transparent, add_axis, get_sphere
from numpy import pi
import time


def get_dis(path):
    dis = []
    with open(path, mode="r") as f:
        for line in f:
            xx = line.split()
            dis.append([int(xx[0]), int(xx[1]), int(xx[2]), float(xx[3])])

    return dis


def get_ellipsoid(xo=(0., 0., 0.), r=(1., 1., 1.), dl=0.1, equal_phi=False,
                  rotate=None):
    """ Generate vertices on an ellipsoial surface. """
    ntheta = int(pi * r[2] / dl) + 1
    theta = np.linspace(0., pi, ntheta)
    xyz = np.array([[0., 0., r[2]], [0., 0., -r[2]]])
    u = np.array([0., 1.])
    v = np.array([1., 1.])
    for _th in theta[1:-1]:
        sth = np.sin(_th)
        cth = np.cos(_th)
        if equal_phi:
            nphi = 2 * (ntheta - 1) + 1
        else:
            rt = r[2] * np.sin(_th)
            nphi = max(int(2. * pi * rt / dl), 4) + 1
        phi = np.linspace(0., 2. * pi, nphi)
        _m = np.ones_like(phi)
        _xyz = np.vstack([r[0] * sth * np.cos(phi), r[1] * sth * np.sin(phi), r[2] * cth * _m]).T
        xyz = np.vstack([xyz, _xyz])

        u = np.hstack([u, _m * _th / pi])
        v = np.hstack([v, phi / pi])

    if rotate is not None:
        xyz = xyz.dot(rotate.T)
    xyz = xyz + np.array(xo)
    return xyz, u, v


if __name__ == '__main__':
    fontsize_legend = fig_fontsizes["legend"]
    fontsize_label = fig_fontsizes["label"]
    fontsize_text = fig_fontsizes["text"]

    savefigs = not True

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
    ax.view_init(elev=12, azim=-27)
    fig.set_tight_layout(True)

    # rotation matrix A for ellipsoid x = A x'
    alpha_r = 30. / 180. * pi
    cos_a = np.cos(alpha_r)
    sin_a = np.sin(alpha_r)

    matrix_a = np.array([[cos_a, -sin_a, 0.],
                         [sin_a, cos_a, 0.],
                         [0., 0., 1.]])

    gamma_r = 45. / 180. * pi
    cos_g = np.cos(gamma_r)
    sin_g = np.sin(gamma_r)
    matrix_g = np.array([[1., 0., 0.],
                         [0., cos_g, -sin_g],
                         [0., sin_g, cos_g]])

    matrix_r = matrix_a.dot(matrix_g)

    nn = 2
    r = (0.2, 0.25, 0.3)
    dl = r[2] / 2**nn
    xo = np.array([0.5, 0.5, 0.5])

    p_axis = np.array([[1., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.]])
    p_axis = 0.5 * p_axis.dot(matrix_r.T)
    xyz_o = np.zeros((3, 3)) + xo
    # Principal axes of the ellipsoid
    ax.quiver(xyz_o[:, 0], xyz_o[:, 1], xyz_o[:, 2],
              p_axis[:, 0], p_axis[:, 1], p_axis[:, 2], edgecolors='tab:orange',
              arrow_length_ratio=0.1, alpha=0.9)
    texts = [r"$\hat{\mathbf{e}}_{x}'$", r"$\hat{\mathbf{e}}_{y}'$", r"$\hat{\mathbf{e}}_{z}'$"]

    for i in range(3):
        _xt = xo + 1.1 * p_axis[i]
        ax.text(_xt[0], _xt[1], _xt[2], texts[i], fontsize=14, color="tab:orange")

    xyz, u, v = get_ellipsoid(xo=(0.5, 0.5, 0.5), r=r, dl=dl, equal_phi=False,
                              rotate=matrix_r)
    tri = mtri.Triangulation(u, v)

    # analytical solution
    vol_ana = 4. / 3. * pi * np.prod(r)

    # triangularization
    tri = mtri.Triangulation(u, v)
    nx = xyz.shape[0]
    ind_tri = tri.triangles

    print("Total number of elements: {:d}\n".format(len(tri.triangles)))
    polys = []
    for _ivs in tri.triangles:
        _p_tri = xyz[_ivs, :]
        polys.append(_p_tri)

    plot_polygons(ax, polys, linewidth=1,
                  facecolors=[0, 0, 1, 0.9], edgecolors=[0, 0, 0, 1.])

    # schematics of the meshes
    nmesh = 8
    dx = 1. / nmesh
    cub = Cube()
    cub.plot_cube(ax, color='k', linestyle='-', linewidth=1)
    for i in range(1):
        for j in range(1):
            for k in range(nmesh):
                x0 = np.array([i, j, k]) * dx
                dl = np.ones(3) * dx

                cub = Cube(dl, x0)
                cub.plot_cube(ax, color='k', linestyle='-', alpha=0.5, linewidth=1)

    if savefigs:
        fig.savefig("ellipsoid_{:d}.pdf".format(nn))

    # identify the cell that elements belong to
    indices = []
    for _ivs in tri.triangles:
        _p_tri = xyz[_ivs, :]
        xc = np.sum(_p_tri, axis=0) / 3.
        indices.append([int(xc[0] / dx), int(xc[1] / dx), int(xc[2] / dx)])

    vol_n = 0.

    # volume fraction distribution with F2V
    dis_f2v = np.zeros((nmesh, nmesh, nmesh))

    start = time.perf_counter()
    for i in range(nmesh):
        for j in range(nmesh):
            for k in range(nmesh):
                # select the elements neighboring the current cell under consideration
                tri_nei = []
                ind_min = 100
                for it, _ivs in enumerate(tri.triangles):
                    di = np.abs(indices[it][0] - i)
                    dj = np.abs(indices[it][1] - j)
                    dk = np.abs(indices[it][2] - k)
                    ind_min = min(ind_min, di + dj + dk)
                    if di <= 1 and dj <= 1 and dk <= 1:
                        tri_nei.append(_ivs)

                if len(tri_nei) == 0:
                    for it, _ivs in enumerate(tri.triangles):
                        di = np.abs(indices[it][0] - i)
                        dj = np.abs(indices[it][1] - j)
                        dk = np.abs(indices[it][2] - k)
                        if di + dj + dk == ind_min:
                            tri_nei.append(_ivs)

                x0 = np.array([i, j, k]) * dx
                dl = np.ones(3) * dx

                cub = Cube(dl, x0)

                vol = cub.front2vof(xyz, tri_nei)
                vol_n += vol

                vol_f = vol / dx**3
                dis_f2v[i, j, k] = vol_f[0]

    end = time.perf_counter()
    print("Elapsed time:", end - start)

    error_f2v = np.abs(vol_n[0] - vol_ana) / vol_ana
    print("F2V results:")
    print("V_num components: {:.12e}, {:.12e}, {:.12e}".format(vol_n[0], vol_n[1], vol_n[2]))
    print("V_num:{:.12e}, V_ana:{:.12e}, error:{:.12e}".format(vol_n[0], vol_ana, error_f2v))

    # read the results obtained with VOFi
    path = "data/vofi_ellipsoid.dat"
    list_vofi = get_dis(path)
    dis_vofi = np.zeros((nmesh, nmesh, nmesh))

    vol_vofi = 0.
    for xx in list_vofi:
        vol_vofi += xx[3]
        dis_vofi[xx[0], xx[1], xx[2]] = xx[3]

    # error between F2V and VOFi
    error_l1 = np.max(np.fabs(dis_vofi - dis_f2v))

    vol_vofi *= dx**3
    error_vofi = np.abs(vol_vofi - vol_ana) / vol_ana
    print("VOFi: V_num:{:.12e}, V_ana:{:.12e}, error:{:.12e}".format(vol_vofi, vol_ana, error_vofi))
    print("F2V-VOFi: V_num:{:.12e}, V_ana:{:.12e}, error:{:.12e}".format(vol_vofi, vol_ana, error_l1))
    plt.show()