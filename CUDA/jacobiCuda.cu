#include <stdio.h>
#include <stdlib.h>
// #include <lcutil.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "datatypes.h"

__global__ void recvKernel(sendtype *d_snd, recvtype *d_rec, double *d_u, double *d_u_old){
    #define SRC(XX,YY) d_u_old[(YY)*(d_snd->n+2)+(XX)]
    #define DST(XX,YY) d_u[(YY)*(d_snd->n+2)+(XX)]

    double xLeft = -1.0;
    double yBottom = -1.0;
    double xLeft_local, yBottom_local;
    int n_local, m_local, coords[2];

    double deltaX = (2.0)/(d_snd->n-1);
    double deltaY = (2.0)/(d_snd->m-1);

    double cx = 1.0/(deltaX*deltaX);
    double cy = 1.0/(deltaY*deltaY);
    double cc = -2.0*cx-2.0*cy-(d_snd->alpha);

    SRC(0,0) = 1.0;
    double *temp;
    temp = d_u;
    d_u = d_u_old;
    d_u_old = temp;
    d_rec->elem1 = DST(0,0);
    d_rec->elem2 = DST(1,0);    
}


int main(){
    sendtype *snd;
    snd = (sendtype *) malloc(sizeof(sendtype));

    scanf("%d,%d", &(snd->n), &(snd->m));
    scanf("%lf", &(snd->alpha));
    scanf("%lf", &(snd->relax));
    scanf("%lf", &(snd->tol));
    scanf("%d", &(snd->mits));
    printf("-> %d, %d, %g, %g, %g, %d\n", snd->n, snd->m, snd->alpha, snd->relax, snd->tol, snd->mits);
    
    sendtype *d_snd;
    cudaMalloc((void **) &d_snd, sizeof(sendtype));
    cudaMemcpy(d_snd, snd, sizeof(sendtype), cudaMemcpyHostToDevice);

    recvtype *rec, *d_rec;
    rec = (recvtype *) malloc(sizeof(recvtype));
    cudaMalloc((void **) &d_rec, sizeof(recvtype));

    double *d_u, *d_u_old;    
    cudaError_t err = cudaMalloc((void **) &d_u, (snd->n + 2)*(snd->m + 2)*sizeof(double));
    if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(err));
    }
    err = cudaMalloc((void **) &d_u_old, (snd->n + 2)*(snd->m + 2)*sizeof(double));
    if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(err));
    }
    cudaMemset(d_u, 0, (snd->n + 2)*(snd->m + 2)*sizeof(double));
    cudaMemset(d_u_old, 0, (snd->n + 2)*(snd->m + 2)*sizeof(double));

    recvKernel<<<1,1>>>(d_snd, d_rec, d_u, d_u_old);
    cudaMemcpy(rec, d_rec, sizeof(recvtype), cudaMemcpyDeviceToHost);
    printf("elem1 = %lf, elem2 = %lf\n", rec->elem1, rec->elem2);
    
    cudaFree(d_snd);
    cudaFree(d_rec);
    cudaFree(d_u);
    cudaFree(d_u_old);
    free(snd);
    free(rec);

    return 0;
}
