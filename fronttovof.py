import numpy as np
from numpy import pi
from numpy.linalg import norm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


#  for plotting purpose
def set_axis_transparent(ax):
    """ Make the background of a 3D figure transparent. """
    # make the planes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.set_axis_off()


def add_axis(ax, xo=(0., 0., 0.), dx=(1.2, 1.2, 1.2), **kwargs):
    """ Add three arrows to indicate the coordinate system. """
    _x = np.zeros(3) + xo[0]
    _y = np.zeros(3) + xo[1]
    _z = np.zeros(3) + xo[2]
    _xo = np.array(xo)
    _u = np.array([1., 0., 0.]) * dx[0]
    _v = np.array([0., 1., 0.]) * dx[1]
    _w = np.array([0., 0., 1.]) * dx[2]
    _dx = np.vstack([_u, _v, _w]) * 1.05
    ax.quiver(_x, _y, _z, _u, _v, _w, **kwargs)
    texts = [r"$\hat{\mathbf{e}}_{x}$", r"$\hat{\mathbf{e}}_{y}$", r"$\hat{\mathbf{e}}_{z}$"]

    ax.text(xo[0] - 0.1, xo[0] - 0.1, xo[0] - 0.1, r"$o$", fontsize=14)
    for i in range(3):
        _xt = _xo + _dx[i]
        ax.text(_xt[0], _xt[1], _xt[2], texts[i], fontsize=14)


def plot_polygons(ax, polygons, **kwargs):
    poly3d = Poly3DCollection(polygons, **kwargs)
    ax.add_collection3d(poly3d)


# Generate different basic geometries for testing the F2V
def get_plane(pv):
    """ Get the equation of plane on which a polygon is located:
    m[0]*x + m[1]*y + m[2]*z = m[3]. """
    _pv = np.array(pv)
    dp1 = _pv[1] - _pv[0]
    dp2 = _pv[2] - _pv[1]
    _m = np.cross(dp1, dp2)
    _m /= np.linalg.norm(_m)
    _alpha = np.dot(_m, _pv[0])
    return np.hstack([_m, _alpha])


def get_triangle(xo=(0., 0., 0.), n=(0., 0., 1.), l=2.):
    """ Generate an equilateral with normal n, side length l and centroid at xo. """
    sq3 = np.sqrt(3.)
    _tri = np.array([[sq3 / 3., 0., 0.], [-sq3 / 6., 0.5, 0.], [-sq3 / 6., -0.5, 0.]]) * l
    _nz = np.array(n)
    _nz /= norm(_nz)

    if abs(_nz[2]) == 1.:
        _nx = np.array([1., 0., 0.])
    else:
        _nx = np.copy(_nz)
        _nx[2] = 0.
        _nx[0], _nx[1] = _nx[1], -_nx[0]
        _nx /= norm(_nx)

    _ny = np.cross(_nz, _nx)

    m_matrix = np.vstack([_nx, _ny, _nz])
    _tri_n = np.dot(_tri, m_matrix) + np.array(xo)
    return _tri_n


def get_sphere(xo=(0., 0., 0.), r=1., dl=0.1, equal_phi=False):
    """ Generate vertices on a spherical surface. """
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


class Polygon():
    def __init__(self, pv):
        self.pv = np.array(pv)
        _pv = self.pv
        nv = _pv.shape[0]
        self.nv = nv
        self.dn = np.array([1., 0., 0.])

        _xc = np.sum(_pv, axis=0) / nv
        xcen = []
        area = []
        a_max = 0.
        for i in range(nv):
            ip = (i + 1) % nv
            xcen.append((_pv[i] + _pv[ip] + _xc) / 3.)
            _dn = np.cross(_pv[i] - _xc, _pv[ip] - _xc)
            _a = np.linalg.norm(_dn)
            area.append(0.5 * abs(_a))
            if _a > a_max:
                a_max = _a
                self.dn = _dn / _a
        xcen = np.array(xcen)
        area = np.array(area)
        self.area = np.sum(area)

        if nv == 2:
            print("Cannot construct a polygon with only two vertices.")
        if self.area == 0.:
            self.xcen = _xc
            print("Degenerated polygon with zero area: ", _pv)
        else:
            self.xcen = np.sum(xcen * area[:, np.newaxis], axis=0) / self.area

    def plot_polygen(self, ax, **kwargs):
        poly3d = Poly3DCollection([self.pv], **kwargs)
        ax.add_collection3d(poly3d)


class Cube():
    def __init__(self, dx=(1., 1., 1.), xo=(0., 0., 0.)):
        """ Cuboid with size lengths dx, one vertex at xo"""
        self.dx = np.array(dx)
        self.xo = np.array(xo)
        self.xrange = (np.vstack([[0., 0., 0.], self.dx]) + self.xo).T
        # eight cuboid vertices
        _xv = [[0., 0., 0.], [1., 0., 0.],
               [0., 1., 0.], [1., 1., 0.],
               [0., 1., 1.], [1., 1., 1.],
               [0., 0., 1.], [1., 0., 1.]]
        self.xv = np.array(_xv) * self.dx + self.xo

        self.pv = None  # final polygon after clipping algorithm
        self.polys_cut = []  # polygons obtained after the clipping of each direction
        self.flag_cut = []  # flag of polygon vertex after each clipping
        self.triangles_inside = []  # triangles within cube or with intersection with cube
        self.polygons_inside = []   # the corresponding polygons
        self.flag_inside = []
        self.index_face = [1 << i for i in range(6)]    # flag of each cube face

    def clip3d(self, pv, verbose=False):
        """ Clipping function: for an input triangle pv, return the polygon part within
        the cuboid. """

        eps = 1.e-12
        self.pv = np.array(pv)
        poly = [self.pv[0], self.pv[1], self.pv[2]]
        flag_face = []  # flag used to identify the face to which the vertex belong
        self.polys_cut = []
        self.polys_cut.append(np.array(poly))

        # identify the vertex of the original triangle located on the cube surface
        for _poly in poly:
            _flag = 0   # 0 for vertex inside the cube
            for i_face in range(3):
                x_min, x_max = self.xrange[i_face, 0], self.xrange[i_face, 1]
                if np.fabs(_poly[i_face] - x_min) < eps:
                    _poly[i_face] = x_min
                    _flag = _flag | self.index_face[2 * i_face]
                elif np.fabs(_poly[i_face] - x_max) < eps:
                    _poly[i_face] = x_max
                    _flag = _flag | self.index_face[2 * i_face + 1]
            flag_face.append(_flag)
        self.flag_cut = []
        self.flag_cut.append(np.array(flag_face))

        # clip the triangle sequentially
        for i_face in range(3):
            npoly = len(poly)
            x_min, x_max = self.xrange[i_face, 0], self.xrange[i_face, 1]
            flag_min, flag_max = self.index_face[2 * i_face], self.index_face[2 * i_face + 1]

            # store the information of points which should be added to or removed from the polygon
            i_ins = []
            x_ins = []
            flag_ins = []
            rm_ind = []
            for i in range(npoly):
                x1, x0 = poly[(i + 1) % npoly], poly[i]
                flag_1, flag_0 = flag_face[(i + 1) % npoly], flag_face[i]
                flag_c = flag_1 & flag_0
                _dxy = x1 - x0
                dh = _dxy[i_face]

                # identify the vertex which should be removed,
                # the flag test is used to keep the vertex located exactly on cube face
                if (x0[i_face] < x_min and flag_0 & flag_min != flag_min) \
                        or (x0[i_face] > x_max and flag_0 & flag_max != flag_max):
                    rm_ind.append(1)
                else:
                    rm_ind.append(0)

                # compute the intersection point between the edge and cube face
                if dh != 0.:
                    t1 = (x_min - x0[i_face]) / dh
                    t2 = (x_max - x0[i_face]) / dh

                    # two intersection points should be inserted in ascending order of t
                    if t1 > t2:
                        t1, t2 = t2, t1

                    if 0. < t1 < 1.:
                        _xy_i = x0 + t1 * _dxy
                        x_ins.append(_xy_i)
                        i_ins.append(i)
                        flag_ins.append(flag_c)

                    if 0. < t2 < 1.:
                        _xy_i = x0 + t2 * _dxy
                        x_ins.append(_xy_i)
                        i_ins.append(i)
                        flag_ins.append(flag_c)

            for i, _x in enumerate(x_ins):
                # fix the coordinates error due to finite accuracy
                for i_fix in range(3):
                    xx_min, xx_max = self.xrange[i_fix, 0], self.xrange[i_fix, 1]
                    if np.fabs(_x[i_fix] - xx_min) < eps:
                        _x[i_fix] = xx_min
                        flag_ins[i] = flag_ins[i] | self.index_face[2 * i_fix]
                    elif np.fabs(_x[i_fix] - xx_max) < eps:
                        _x[i_fix] = xx_max
                        flag_ins[i] = flag_ins[i] | self.index_face[2 * i_fix + 1]

                ind = i_ins[i] + i + 1
                poly.insert(ind, _x)
                flag_face.insert(ind, flag_ins[i])
                rm_ind.insert(ind, 0)

            # remove the vertices outside the cube
            i_rm = []
            for i, _ind in enumerate(rm_ind):
                if _ind == 1:
                    i_rm.append(i)

            for i, _irm in enumerate(i_rm):
                ind = _irm - i
                poly.pop(ind)
                flag_face.pop(ind)

            self.polys_cut.append(np.array(poly))
            self.flag_cut.append(np.array(flag_face))

            # Stop cutting when there are less than three vertices
            if len(poly) <= 2:
                if len(poly) > 0 and verbose:
                    print("Degenerated polygon after the cutting in direction: {:d} ".format(i_face))
                    print("Original triangle:", self.pv)
                    print("Final shape after cutting:", np.array(poly))
                break
        return np.array(poly), np.array(flag_face)

    def plot_cut_procedure(self, ax, verbose=False):
        """ plot the polygon obtained after each clipping. """
        fmts = ['r', 'b', 'g', 'tab:orange']

        print("\n--- Plotting the cutting procedure ---")
        if verbose:
            print("The polygons generated after the cutting in each direction:")
            for _i, _pv in enumerate(self.polys_cut):
                flag_face = self.flag_cut[_i]
                print("Step {:d}".format(_i), "Polygon:")
                for _iv, _xv in enumerate(_pv):
                    print("p{:d}: {:.16e} {:.16e} {:.16e}".format(_iv, _xv[0], _xv[1], _xv[2]))
                face_bi = ["{:06b}".format(i) for i in flag_face]
                print("face_index：", flag_face, face_bi)

        for _i, _pv in enumerate(self.polys_cut):
            if len(_pv) > 0:
                if _i == 0:
                    Polygon(_pv).plot_polygen(ax[_i], facecolors=[1, 0, 0., 0.], edgecolors=fmts[_i], linewidth=3)
                else:
                    _pvm = self.polys_cut[_i - 1]
                    Polygon(_pv).plot_polygen(ax[_i], facecolors=[0, 0, 0., 0.],
                                              edgecolors=fmts[_i], linewidth=2, linestyle='-')
                    Polygon(_pvm).plot_polygen(ax[_i], facecolors=[0, 0, 0., 0.],
                                               edgecolors=fmts[_i - 1], linewidth=3, linestyle='--')
        print("--- End plotting the cutting procedure ---\n")

    def cut_plane(self, m):
        swap = False
        _m = np.array(m)[:3]
        _alpha = m[3] + np.sum(np.maximum(0., -_m))
        _m = np.abs(_m)
        _alpha_max = np.sum(_m)

        _m /= _alpha_max
        _alpha /= _alpha_max

        if _alpha > 1. - 1.e-12:
            return 1.
        elif _alpha < 1.e-12:
            return 0.

        if _alpha > 0.5:
            swap = True
            _alpha = 1. - _alpha

        _m = np.sort(_m)
        _v1 = _m[0]**2 / np.max([6. * _m[1] * _m[2], 1.e-32])
        _m12 = _m[0] + _m[1]
        _mm = np.min([_m[2], _m12])
        _coef_m = 6. * np.prod(_m)

        if 0. <= _alpha < _m[0]:
            vol = _alpha**3 / _coef_m
        elif _alpha < _m[1]:
            vol = _alpha * (_alpha - _m[0]) / (2. * _m[1] * _m[2]) + _v1
        elif _alpha < _mm:
            _coef_n = _alpha**2 * (3. * _m12 - _alpha) + np.sum(_m[:-1]**2 * (_m[:-1] - 3. * _alpha))
            vol = _coef_n / _coef_m
        elif _m[2] < _m12:
            _coef_n = _alpha**2 * (3. - 2. * _alpha) + np.sum(_m**2 * (_m - 3. * _alpha))
            vol = _coef_n / _coef_m
        else:
            vol = (2. * _alpha - _m12) / (2. * _m[2])

        if swap:
            vol = 1. - vol
        # print(vol, swap, _alpha)
        return vol

    def plot_cube(self, ax):
        xy = [[0., 0., 0.], [1., 0., 0.],
              [0., 1., 0.], [1., 1., 0.],
              [0., 1., 1.], [1., 1., 1.],
              [0., 0., 1.], [1., 0., 1.]]
        xy = np.array(xy) + self.xo
        for idir in range(3):
            ix = idir % 3
            iy = (idir + 1) % 3
            iz = (idir + 2) % 3
            _x = xy[:, ix] * self.dx[0]
            _y = xy[:, iy] * self.dx[1]
            _z = xy[:, iz] * self.dx[2]
            for i in range(4):
                ii = 2 * i
                ax.plot(_x[ii:ii + 2], _y[ii:ii + 2], _z[ii:ii + 2], 'k-', markersize=2,
                        fillstyle='none', alpha=0.7, linewidth=1)

    def poly2vof(self, polys, flags):
        """ Return the volume fraction in the cube cell cut by polygons.
        Make sure that the polygons and the cube form a closed region.
        When computing the normal by following the right-hand side rule,
        the normal points out from the reference phase.
        The three elements of vol represents the cases with
        F = (x, 0, 0), (0, y, 0) and (0, 0, z), respectively."""
        vol = np.zeros(3)
        area = np.zeros(3)
        length = np.zeros(3)
        _flags_dir = [self.index_face[2 * i + 1] for i in range(3)]

        for _ip, _poly in enumerate(polys):
            _flag = flags[_ip]
            coef_n = np.ones(3)
            py1 = Polygon(_poly)
            _nv = _poly.shape[0]

            if py1.area < 1.e-16:
                continue

            for i_dir in range(3):
                _flag_face = _flags_dir[i_dir]
                for _iv in range(_nv):
                    _flag_face = (_flag_face & _flag[_iv])
                    if _flag_face != _flags_dir[i_dir]:
                        break
                if _flag_face == _flags_dir[i_dir]:
                    coef_n[i_dir] = 0.

            vol += py1.xcen * py1.area * py1.dn * coef_n
            for _iv, _x in enumerate(_poly):
                _iv1 = (_iv + 1) % _nv
                _x2, _x1 = _poly[_iv1], _poly[_iv]
                _flag2, _flag1 = _flag[_iv1], _flag[_iv]
                _dx = _poly[_iv1] - _poly[_iv]
                _ds = norm(_dx)
                for i_dir in range(3):
                    _flag_dir = self.index_face[2 * i_dir + 1]
                    i_dir_1 = (i_dir + 1) % 3
                    i_dir_2 = (i_dir + 2) % 3
                    # do not take into account the element located exactly on the cell surface
                    if coef_n[i_dir] == 0.:
                        continue
                    _dn = np.zeros(3)
                    _dn[i_dir] = 1.
                    if _flag1 & _flag2 & _flags_dir[i_dir]:
                        # edge located on the cell face
                        _dn_e = np.copy(py1.dn)
                        _dn_e[i_dir] = 0.
                        _dn_e /= norm(_dn_e)
                        area[i_dir] += 0.5 * _ds * _dn_e[i_dir_2] * (_x2 + _x1)[i_dir_2]
                        _flag_edge = _flags_dir[i_dir] | _flags_dir[i_dir_2]

                        if _flag1 & _flag2 & _flag_edge != _flag_edge:
                            # vertex located on the cell edge
                            if _flag2 & _flags_dir[i_dir_2]:
                                _xz = _x2[i_dir_1]
                            elif _flag1 & _flags_dir[i_dir_2]:
                                _xz = _x1[i_dir_1]
                            else:
                                _xz = 0.
                            length[i_dir] += _xz * np.sign(_dn_e[i_dir_1])
        # print(vol, area, length)

        length[np.fabs(length) < 1.e-12] = 0.
        _msk = length < 0.
        length[_msk] = 1. + length[_msk]

        area[np.fabs(area) < 1.e-12] = 0.
        area += length
        _msk = area < 0.
        area[_msk] = 1. + area[_msk]

        vol[np.fabs(vol) < 1.e-12] = 0.
        vol += area
        _msk = vol < 0.
        vol[_msk] = 1. + vol[_msk]
        return vol

    def front2vof(self, points, triangles):
        """ Identify the triangles cut by the cube,
        and compute the polygons obtained using the clipping algorithm.
        Main function of the F2V algorithm. """
        self.triangles_inside = []
        self.polygons_inside = []
        self.flag_inside = []

        for _ivs in triangles:
            _p_tri = points[_ivs, :]
            _poly, _flag = self.clip3d(_p_tri)
            if _poly.shape[0] > 2:
                self.triangles_inside.append(_p_tri)
                self.polygons_inside.append(_poly)
                self.flag_inside.append(_flag)

        vol = self.poly2vof(self.polygons_inside, self.flag_inside)

        if np.min(np.abs(vol)) < 1.e-12:
            # For corner cases, empty or full cell
            xc = 0.5 * self.dx + self.xo  # cuboid center

            d_min = 1.e32
            xc_min = np.zeros(3)
            dn_min = np.ones(3)
            # search the triangle closest to the cuboid center
            for _ivs in triangles:
                _p_tri = points[_ivs, :]
                _poly = Polygon(_p_tri)
                _d = np.fabs(_poly.dn.dot(_poly.xcen - xc))
                if _d < d_min:
                    xc_min = _poly.xcen
                    dn_min = _poly.dn
                    d_min = _d

            # search the cuboid vertex not on the element plane
            for _xv in self.xv:
                dx = xc_min - _xv
                dx /= (norm(dx) + 1.e-32)
                if abs(dn_min.dot(dx)) > 1.e-6:
                    break
            phase_sign = np.sign(dn_min.dot(dx))
            vol = np.array([1., 1., 1.] if phase_sign > 0. else [0., 0., 0.])

        return vol

    def get_points_on_face(self, ind_face):
        xyz = []
        flag_face = self.index_face[ind_face]
        for ip, pv in enumerate(self.polygons_inside):
            for iv in range(pv.shape[0]):
                if self.flag_inside[ip][iv] & flag_face:
                    xyz.append(pv[iv])

        return np.array(xyz)
