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


def get_ana_sin(k, phi):
    return (np.cos(k + phi) - np.cos(phi)) / k


if __name__ == '__main__':
    fontsize_legend = fig_fontsizes["legend"]
    fontsize_label = fig_fontsizes["label"]
    fontsize_text = fig_fontsizes["text"]

    savefigs = not True

    nn = 5
    nx = ny = 2**nn + 1
    u = np.linspace(0., 1., nx + 1)
    v = np.linspace(0., 1., ny + 1)

    x, y = np.meshgrid(u, v)
    x, y = x.flatten(), y.flatten()

    k_x, k_y = 8. / 5., 8. / 5.
    phi_x, phi_y = 1./7., 1./5.
    a0, a1 = 0.5, 1. / 6.
    z = a0 + a1 * np.sin(pi * (k_x * x + phi_x)) * np.sin(pi * (k_y * y + phi_y))

    # analytical solution
    vol_ana = a0 + a1 * get_ana_sin(k_x * pi, phi_x * pi) * get_ana_sin(k_y * pi, phi_y * pi)

    # triangularization
    tri = mtri.Triangulation(x, y)
    xyz = np.vstack([x, y, z]).T
    nx = xyz.shape[0]
    ind_tri = tri.triangles
    print("Total number of elements: {:d}\n".format(len(tri.triangles)))

    polys = []
    for _ivs in tri.triangles:
        _p_tri = xyz[_ivs, :]
        polys.append(_p_tri)

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
    ax.view_init(elev=20, azim=-31)
    fig.set_tight_layout(True)

    # for plotting a shaded cube
    axes = [1, 1, 1]
    data = np.ones(axes)

    # control Transparency
    alpha = 0.4
    colors = np.empty(axes + [4])

    colors[0] = [1, 1, 1, alpha]  # grey
    # ax.voxels(data, facecolors=colors, edgecolors='grey')

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

    plot_polygons(ax, polys, linewidth=1,
                  facecolors=[0, 0, 1, 0.9], edgecolors=[0, 0, 0, 1.])

    if savefigs:
        fig.savefig("sin3D_{:d}.pdf".format(nn))

    # identify the cell that elements belong to
    indices = []
    for _ivs in tri.triangles:
        _p_tri = xyz[_ivs, :]
        xc = np.sum(_p_tri, axis=0) / 3.
        indices.append([int(xc[0] / dx), int(xc[1] / dx), int(xc[2] / dx)])

    vol_n = 0

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

                vol_f = vol / dx ** 3
                dis_f2v[i, j, k] = vol_f[0]

    end = time.perf_counter()
    print("Elapsed time:", end - start)

    error_f2v = np.abs(vol_n[0] - vol_ana) / vol_ana
    print("F2V results:")
    print("V_num components: {:.12e}, {:.12e}, {:.12e}".format(vol_n[0], vol_n[1], vol_n[2]))
    print("V_num:{:.12e}, V_ana:{:.12e}, error:{:.12e}".format(vol_n[0], vol_ana, error_f2v))

    # read the results obtained with VOFi
    path = "data/vofi_sin.dat"
    list_vofi = get_dis(path)
    dis_vofi = np.zeros((nmesh, nmesh, nmesh))

    vol_vofi = 0.
    for xx in list_vofi:
        vol_vofi += xx[3]
        dis_vofi[xx[0], xx[1], xx[2]] = xx[3]

    error_l1 = np.max(np.fabs(dis_vofi - dis_f2v))

    vol_vofi *= dx ** 3
    error_vofi = np.abs(vol_vofi - vol_ana) / vol_ana
    print("VOFi: V_num:{:.12e}, V_ana:{:.12e}, error:{:.12e}".format(vol_vofi, vol_ana, error_vofi))
    print("F2V-VOFi: V_num:{:.12e}, V_ana:{:.12e}, error:{:.12e}".format(vol_vofi, vol_ana, error_l1))

    plt.show()