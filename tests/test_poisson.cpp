#include<cstdio>
#include<cstdlib>
#include<cassert>
#include"poisson.h"

int main(int argc, char ** args)
{
    assert(argc == 4);
    int nr = atoi(args[1]);
    int nt = atoi(args[2]);
    int nIters = atoi(args[3]);

    float * bHost = new float[nr * nt];
    assert(fread(bHost, nr * nt * sizeof(float), 1, stdin) == 1);
    float * bDev;
    cudaMalloc(&bDev, nr * nt * sizeof(float));
    cudaMemcpy(bDev, bHost, nr * nt * sizeof(float), cudaMemcpyHostToDevice);
    delete[] bHost;

    float * rHost = new float[nr + 3];
    float * tHost = new float[nt + 3];
    assert(fread(bHost, (nr + 3) * sizeof(float), 1, stdin) == 1);
    assert(fread(tHost, (nt + 3) * sizeof(float), 1, stdin) == 1);
    float * rDev, * tDev;
    cudaMalloc(&rDev, (nr + 3) * sizeof(float));
    cudaMalloc(&tDev, (nt + 3) * sizeof(float));
    cudaMemcpy(rDev, rHost, (nr + 3) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(tDev, tHost, (nt + 3) * sizeof(float), cudaMemcpyHostToDevice);
    delete[] rHost;
    delete[] tHost;

    float * pDev0, * pDev1;
    cudaMalloc(&pDev0, nr * nt * sizeof(float));
    cudaMalloc(&pDev1, nr * nt * sizeof(float));
    cudaMemset(pDev0, 0, nr * nt * sizeof(float));
    cudaMemset(pDev1, 0, nr * nt * sizeof(float));
    float * pDev = laplace_iters(pDev1, pDev0, bDev, rDev, tDev,
                                 nr, nt, nIters);

    float * pHost = new float[nr * nt];
    cudaMemcpy(pHost, pDev, nr * nt * sizeof(float), cudaMemcpyDeviceToHost);
    delete[] pHost;

    cudaFree(pDev0);
    cudaFree(pDev1);
    cudaFree(bDev);
    return 0;
}
