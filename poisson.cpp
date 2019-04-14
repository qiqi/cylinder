#include "poisson.h"

__device__
float laplace_cr(float * r, float * t) {
    float drc = (r[2] - r[0]) / 2.0;
    float dt = t[1] - t[0];
    return r[1] * dt / drc;
}

__device__
float laplace_ct(float * t, float * r) {
    float dtc = (t[2] - t[0]) / 2;
    float dr = r[1] - r[0];
    return dr / (dtc * (r[0] + r[1]) / 2);
}

__device__
float laplace_point(float * p, float * c) {
    return (p[1] - p[0]) * c[0] +
           (p[2] - p[0]) * c[1] +
           (p[3] - p[0]) * c[2] +
           (p[4] - p[0]) * c[3];
}

__device__
float laplace_point_iter(float * p, float * c, float bxArea,
                         float relaxation) {
    float res = laplace_point(p, c) - bxArea;
    return p[0] + relaxation * res / (c[0] + c[1] + c[2] + c[3]);
}

__device__
void laplace_iter_i(float * pNext, float * p, float * b,
                    float * r, float * t, int ir, int it,
                    int nr, int nt)
{
    float cStencil[4] = {
          laplace_cr(r + ir + 1, t + it + 1),
          laplace_cr(r + ir + 0, t + it + 1),
          laplace_ct(t + it + 1, r + ir + 1),
          laplace_ct(t + it + 0, r + ir + 1)};
    float p0 = p[ir * nt + it];
    float pStencil[5] = {p0,
          ir < nr - 1 ? p[(ir + 1) * nt + it] : p0,
          ir > 0      ? p[(ir - 1) * nt + it] : p0,
          it < nt - 1 ? p[ir * nt + it + 1] : p[ir * nt + 0],
          it > 0      ? p[ir * nt + it - 1] : p[ir * nt + nt - 1]};
    float area = (r[ir+2]*r[ir+2] - r[ir+1]*r[ir+1]) / 2
               * (t[it+2] - t[it+1]);
    float bxArea = b[ir * nt + it] * area;
    pNext[ir * nt + it] = laplace_point_iter(pStencil, cStencil, bxArea, 1);
}

__global__
void laplace_iter(float * pNext, float * p, float * b, float * r, float * t,
                  int nr, int nt)
{
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    int it = blockIdx.y * blockDim.y + threadIdx.y;
    if (ir < nr and it < nt)
        laplace_iter_i(pNext, p, b, r, t, ir, it, nr, nt);
}

float * laplace_iters(float * pNext, float * p, float * b,
                      float * r, float * t, int nr, int nt, int nIters)
{
    for (int iIter = 0; iIter < nIters; ++iIter) {
        laplace_iter
            <<<dim3(ceil(nr / 16.), ceil(nt / 16.)), dim3(16, 16)>>>
            (pNext, p, b, r, t, nr, nt);
        float * tmp = p;
        p = pNext;
        pNext = tmp;
    }
    return p;
}
