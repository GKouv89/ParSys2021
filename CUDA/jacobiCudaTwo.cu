#include <stdio.h>
#include <stdlib.h>
// #include <lcutil.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include "datatypes.h"
#include <omp.h>

#define ERR(XX,YY) d_error[(YY)*(d_snd->n)+(XX)]


__global__ void coordCalc(sendtype *d_snd, double *fZ_squared){
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if(index < d_snd->n){
        fZ_squared[index] = -1.0 + index*d_snd->delta;
        fZ_squared[index] = fZ_squared[index]*fZ_squared[index];
    }
}

__global__ void jacobi(sendtype *d_snd, double *fXsquared, double *fYsquared, double *d_u_old, double *d_u, double *d_error){
    #define SRC(XX,YY) d_u_old[(YY)*(d_snd->n+2)+(XX)]
    #define DST(XX,YY) d_u[(YY)*(d_snd->n+2)+(XX)]

    int xIndex = threadIdx.x + blockDim.x*blockIdx.x + 1;
    int yIndex = threadIdx.y + blockDim.y*blockIdx.y + 1;
    double f, updateVal;

    if(xIndex < (d_snd->n + 1) && yIndex < (d_snd->m + 1)){
        f = -d_snd->alpha*(1.0-fXsquared[xIndex-1])*(1.0-fYsquared[yIndex-1]) - 2.0*(2.0-fXsquared[xIndex-1]-fYsquared[yIndex-1]);
        updateVal = (	(SRC(xIndex-1,yIndex) + SRC(xIndex+1,yIndex))*d_snd->cx +
                        (SRC(xIndex,yIndex-1) + SRC(xIndex,yIndex+1))*d_snd->cy +
                        SRC(xIndex,yIndex)*d_snd->cc - f
                    )/d_snd->cc;
        DST(xIndex,yIndex) = SRC(xIndex,yIndex) - d_snd->relax*updateVal;
        ERR(xIndex-1,yIndex-1) = updateVal*updateVal;
    }
}

__global__ void reduceError(double *d_error){
  int step_size = 1;
  int number_of_threads = 1024; // We'll ALWAYS START with this many active PER BLOCK

  int index;
  int fst, snd;

	while (number_of_threads > 0)
	{
    if (threadIdx.x < number_of_threads) // still alive?
    {
      index = threadIdx.x + number_of_threads*blockIdx.x;
      fst = index * step_size * 2;
      snd = fst + step_size;
      d_error[fst] += d_error[snd];
    }

    step_size *= 2; 
		number_of_threads = number_of_threads/2;
    if(threadIdx.x == 0 && number_of_threads == 0){ 
      // DONE, COPYING TO FIRST NUMBLOCKS 
      // OF D_ERROR ARRAY
      d_error[blockIdx.x] = d_error[2*blockIdx.x*blockDim.x];
    }
    __syncthreads();
	}
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
    snd->delta = 2.0/(snd->n-1);
    snd->cx = 1.0/(snd->delta*snd->delta);
    snd->cy = 1.0/(snd->delta*snd->delta);
    snd->cc = -2.0*snd->cx-2.0*snd->cy-snd->alpha;

    sendtype *d_snd, *d_snd_two;
    cudaMalloc((void **) &d_snd, sizeof(sendtype));
    snd->m /= 2;
    cudaMemcpy(d_snd, snd, sizeof(sendtype), cudaMemcpyHostToDevice);
    snd->m *= 2;

    recvtype *rec, *d_rec;
    rec = (recvtype *) malloc(sizeof(recvtype));
    cudaMalloc((void **) &d_rec, sizeof(recvtype));

    double *d_u, *d_u_old, *d_fXsquared, *d_fYsquared, *d_error; 
    double *d_u_two, *d_u_old_two, *d_fYsquared_two, *d_fXsquared_two, *d_error_two;
    double *fXsquared_temp, *fYsquared_temp;        
    fYsquared_temp = (double *)malloc(snd->m*sizeof(double));
    fXsquared_temp = (double *)malloc(snd->n*sizeof(double));
    cudaError_t err = cudaMalloc((void **) &d_u, (snd->n + 2)*(snd->m/2 + 2)*sizeof(double));
    if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(err));
    }
    err = cudaMalloc((void **) &d_u_old, (snd->n + 2)*(snd->m/2 + 2)*sizeof(double));
    if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(err));
    }
    err = cudaMalloc((void **) &d_fXsquared, snd->n*sizeof(double));
    if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(err));
    }
    err = cudaMalloc((void **) &d_fYsquared, snd->m*sizeof(double));
    if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(err));
    }
    int zeroPaddedMemory = pow(2, ceil(log2((snd->m/2)*snd->n))); 
    err = cudaMalloc((void **) &d_error, zeroPaddedMemory*sizeof(double));
    if (err != cudaSuccess){
		fprintf(stderr, "GPUassert for error array: %s\n", cudaGetErrorString(err));
    }
    cudaMemset(d_u, 0, (snd->n + 2)*(snd->m/2 + 2)*sizeof(double));
    cudaMemset(d_u_old, 0, (snd->n + 2)*(snd->m/2 + 2)*sizeof(double));
    cudaMemset(d_fXsquared, 0, snd->n*sizeof(double));
    cudaMemset(d_error, 0, zeroPaddedMemory*sizeof(double));
    cudaMemset(d_fYsquared, 0, snd->m*sizeof(double));

    cudaSetDevice(1);
    snd->m /= 2;
    cudaMalloc((void **) &d_snd_two, sizeof(sendtype));
    cudaMemcpy(d_snd_two, snd, sizeof(sendtype), cudaMemcpyHostToDevice);
    snd->m *= 2;
    err = cudaMalloc((void **) &d_fXsquared_two, snd->n*sizeof(double));
    if (err != cudaSuccess){
		  fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(err));
    }
    err = cudaMalloc((void **) &d_fYsquared_two, (snd->m/2)*sizeof(double));
    if (err != cudaSuccess){
		  fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(err));
    }
    err = cudaMalloc((void **) &d_u_two, (snd->n + 2)*(snd->m/2 + 2)*sizeof(double));
    if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(err));
    }
    err = cudaMalloc((void **) &d_u_old_two, (snd->n + 2)*(snd->m/2 + 2)*sizeof(double));
    if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(err));
    }
    err = cudaMalloc((void **) &d_error_two, zeroPaddedMemory*sizeof(double));
    if (err != cudaSuccess){
		fprintf(stderr, "GPUassert for error array: %s\n", cudaGetErrorString(err));
    }
    cudaMemset(d_u_two, 0, (snd->n + 2)*(snd->m/2 + 2)*sizeof(double));
    cudaMemset(d_u_old_two, 0, (snd->n + 2)*(snd->m/2 + 2)*sizeof(double));
    cudaMemset(d_fXsquared_two, 0, snd->n*sizeof(double));
    cudaMemset(d_error_two, 0, zeroPaddedMemory*sizeof(double));
    cudaMemset(d_fYsquared_two, 0, (snd->m/2)*sizeof(double));
    cudaSetDevice(0);

    int threadNum = 128;
    int blocksNum = ceil((double)snd->n/(double)threadNum);
    

    // I for sure will have 128 threads per block
    // So we now wish to find how many blocks are necessary for
    // dividing our problem size's *side* by 128
    clock_t start = clock(), diff;    
    coordCalc<<<blocksNum, threadNum>>>(d_snd, d_fYsquared);
    coordCalc<<<blocksNum, threadNum>>>(d_snd, d_fXsquared);
    cudaDeviceSynchronize();
    cudaMemcpy(fYsquared_temp, d_fYsquared, snd->m*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(fXsquared_temp, d_fXsquared, snd->n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaSetDevice(1);    
    cudaMemcpy(d_fYsquared_two, &fYsquared_temp[snd->m/2], (snd->m/2)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fXsquared_two, fXsquared_temp, snd->n*sizeof(double), cudaMemcpyHostToDevice);
    cudaSetDevice(0);    
    free(fYsquared_temp);
    free(fXsquared_temp);

    // For the actual arrays, I choose 256 threads per block
    // in a 16x16 cartesian fashion. So now I need to find how
    // many blocks I need per side to have a 2D block grid
    dim3 threadsPerBlock(16, 16);
    blocksNum = ceil((double)snd->n/16.0);
    dim3 blocksInGrid(blocksNum, blocksNum);

    // For error reduction, we can treat the error array as one-dimensional
    // We can use the code that the professor sent pretty much as is, despite
    // not having one block. We'll just find the thread's global id and use that
    // to sum, and the error will be in the very first element of the array
    double *temp;
        
    int iterationCount = 0;
    double error_all = HUGE_VAL;
    double error_all_two;
    double *halo = (double *)malloc(snd->n*sizeof(double));

    threadNum = 1024;
    // I wish for each block to have 2048 elements of the array to reduce 
    blocksNum = zeroPaddedMemory/2048;
    while(iterationCount < snd->mits && error_all > snd->tol){
    	error_all = 0.0;
      jacobi<<<blocksInGrid, threadsPerBlock>>>(d_snd, d_fXsquared, d_fYsquared, d_u_old, d_u, d_error);
      cudaSetDevice(1);
      jacobi<<<blocksInGrid, threadsPerBlock>>>(d_snd_two, d_fXsquared_two, d_fYsquared_two, d_u_old_two, d_u_two, d_error_two);
      cudaDeviceSynchronize();
      cudaSetDevice(0);
      cudaDeviceSynchronize();
      reduceError<<<blocksNum,threadNum>>>(d_error);
      cudaSetDevice(1);
      reduceError<<<blocksNum,threadNum>>>(d_error_two);
      cudaDeviceSynchronize();
      cudaSetDevice(0);
      cudaDeviceSynchronize();
      do{
        if(blocksNum < 2048){
          break;
        }else{
          blocksNum /= 2048;
          reduceError<<<blocksNum, threadNum>>>(d_error);
          cudaSetDevice(1);
          reduceError<<<blocksNum, threadNum>>>(d_error_two);
          cudaDeviceSynchronize();
          cudaSetDevice(0);
          cudaDeviceSynchronize();
        }
      }while(blocksNum != 1);
      cudaMemset(&d_error[blocksNum], 0, (2048 - blocksNum)*sizeof(double));
      reduceError<<<1,threadNum>>>(d_error);
      cudaSetDevice(1);
      cudaMemset(&d_error_two[blocksNum], 0, (2048 - blocksNum)*sizeof(double));
      reduceError<<<1,threadNum>>>(d_error_two);
      cudaDeviceSynchronize();
      cudaSetDevice(0); cudaDeviceSynchronize();
      
      cudaMemcpy(&error_all, &d_error[0], sizeof(double), cudaMemcpyDeviceToHost);
      cudaSetDevice(1);
      cudaMemcpy(&error_all_two, &d_error_two[0], sizeof(double), cudaMemcpyDeviceToHost);
      cudaSetDevice(0);
      error_all += error_all_two;
      error_all = sqrt(error_all)/(snd->n*snd->m);
      
      temp = d_u;
      d_u = d_u_old;
      d_u_old = temp;
      
      temp = d_u_two;
      d_u_two = d_u_old_two;
      d_u_old_two = temp;

      iterationCount++;
      cudaMemcpy(halo, &d_u_old[(snd->m/2)*(snd->n + 2) + 1],snd->n*sizeof(double), cudaMemcpyDeviceToHost);
      cudaSetDevice(1);
      cudaMemcpy(&d_u_old_two[1], halo, snd->n*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(halo, &d_u_old_two[(snd->n + 2) + 1], snd->n*sizeof(double), cudaMemcpyDeviceToHost);
      cudaSetDevice(0);
      cudaMemcpy(halo, &d_u_old[(snd->m/2 + 1)*(snd->n + 2) + 1], snd->n*sizeof(double), cudaMemcpyHostToDevice);

      blocksNum = zeroPaddedMemory/2048;
    }
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Iterations: %d\nResidual: %g\n", iterationCount, error_all);
    printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);

    cudaFree(d_u);
    cudaFree(d_u_old);
    cudaFree(d_fXsquared);
    cudaFree(d_fYsquared);
    cudaFree(d_snd);
    cudaFree(d_rec);
    cudaFree(d_error);

    cudaSetDevice(1);
    cudaFree(d_u_two);
    cudaFree(d_u_old_two);
    cudaFree(d_fXsquared_two);
    cudaFree(d_fYsquared_two);
    cudaFree(d_snd_two);
    cudaFree(d_error_two);
    cudaSetDevice(0);

    free(snd);
    free(rec);
    free(halo);
    return 0;
}
