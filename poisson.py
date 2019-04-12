import numba
import numpy

numba_device = numba.jit(nopython=True)
numba_global = numba.jit(nopython=True)

@numba_device
def laplace_cr(r, t):
    drc = (r[2] - r[0]) / 2
    dt = t[1] - t[0]
    return r[1] * dt / drc

@numba_device
def laplace_ct(t, r):
    dtc = (t[2] - t[0]) / 2
    dr = r[1] - r[0]
    return dr / (dtc * (r[0] + r[1]) / 2)

@numba_device
def laplace_point(p, c):
    return (p[1] - p[0]) * c[0] + \
           (p[2] - p[0]) * c[1] + \
           (p[3] - p[0]) * c[2] + \
           (p[4] - p[0]) * c[3]

@numba_device
def laplace_point_iter(p, c, bxArea, relaxation):
    res = laplace_point(p, c) - bxArea
    return p[0] + relaxation * res / (c[0] + c[1] + c[2] + c[3])

@numba_global
def laplace_iter(p, b, r, t, nIters):
    for iIter in range(nIters):
        for ir in range(r.size - 3):
            for it in range(t.size - 3):
                crp = laplace_cr(r[ir+1:ir+4], t[it+1:it+3])
                crm = laplace_cr(r[ir+0:ir+3], t[it+1:it+3])
                ctp = laplace_ct(t[it+1:it+4], r[ir+1:ir+3])
                ctm = laplace_ct(t[it+0:it+3], r[ir+1:ir+3])
                cStencil = [crm, crp, ctm, ctp]
                p0 = p[ir, it]
                prp = p[ir + 1, it] if ir < r.size - 4 else p0
                prm = p[ir - 1, it] if ir > 0 else p0
                ptp = p[ir, it + 1] if it < t.size - 4 else p[ir, 0]
                ptm = p[ir, it - 1] if it > 0 else p[ir, -1]
                pStencil = [p0, prm, prp, ptm, ptp]
                area = (r[ir+2]**2 - r[ir+1]**2) / 2 * (t[it+2] - t[it+1])
                bxArea = b[ir, it] * area
                p[ir, it] = laplace_point_iter(pStencil, cStencil, bxArea, 1)

def test_laplace_radial():
    r0 = 100
    r = numpy.arange(4) + r0 - 1.5
    t = numpy.arange(4) / 180
    crp = laplace_cr(r[1:], t[1:-1])
    crm = laplace_cr(r[:-1], t[1:-1])
    ctp = laplace_ct(t[1:], r[1:-1])
    ctm = laplace_ct(t[:-1], r[1:-1])
    area = (r[2]**2 - r[1]**2) / 2 * (t[2] - t[1])
    cStencil = numpy.array([crm, crp, ctm, ctp])
    pStencil = numpy.array([0, -1, 1, 0 ,0], float)
    laplP = laplace_point(pStencil, cStencil) / area
    assert(abs(laplP * r0 - 1) < 1E-8)

def test_laplace_iter_concentric():
    nr, nt = 20, 20
    p = numpy.zeros([nr, nt])
    r = 0.5 * (1 + numpy.arange(-1, nr + 2) / nr)
    t = numpy.arange(-1, nt + 2) / nt * 2 * numpy.pi
    rc, tc = (r[1:-2] + r[2:-1]) / 2, (t[1:-2] + t[2:-1]) / 2
    rGrid, tGrid = numpy.meshgrid(rc, tc, indexing='ij')
    f = 1 + (rc - 0.5)**2 * (rc - 1.25)
    fp = 2 * (rc - 0.5) * (rc - 1.25) + (rc - 0.5)**2
    fpp = 4 * (rc - 0.5) + 2 * (rc - 1.25)
    b = numpy.outer(fpp + fp / rc, numpy.ones_like(tc))
    laplace_iter(p, b, r, t, 2000)
    pExact = numpy.outer(f, numpy.ones_like(tc))
    p -= p.mean()
    pExact -= pExact.mean()
    assert(abs(p - pExact).max() < 1E-3)

def test_laplace_iter_radial():
    nr, nt = 20, 120
    p = numpy.zeros([nr, nt])
    r = 0.5 * (1 + numpy.arange(-1, nr + 2) / nr)
    t = numpy.arange(-1, nt + 2) / nt * 2 * numpy.pi
    rc, tc = (r[1:-2] + r[2:-1]) / 2, (t[1:-2] + t[2:-1]) / 2
    rGrid, tGrid = numpy.meshgrid(rc, tc, indexing='ij')
    f = 1 + (rc - 0.5)**2 * (rc - 1.25)
    fp = 2 * (rc - 0.5) * (rc - 1.25) + (rc - 0.5)**2
    fpp = 4 * (rc - 0.5) + 2 * (rc - 1.25)
    b = numpy.outer(fpp + fp / rc - f / rc**2, numpy.sin(tc))
    laplace_iter(p, b, r, t, 10000)
    pExact = numpy.outer(f, numpy.sin(tc))
    p -= p.mean()
    pExact -= pExact.mean()
    assert(abs(p - pExact).max() < 1E-3)

if __name__ == '__main__':
    pass
    #test_laplace_concentric()
    #test_laplace_radial()
