#include <stdio.h>
#include <stdlib.h>
// #include <lcutil.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <numeric>
using namespace std;

__global__ void sum(int* input)
{
	const int tid = threadIdx.x;
	int fst, snd;

	auto step_size = 1;
	int number_of_threads = blockDim.x;

	while (number_of_threads > 0)
	{
		if (tid < number_of_threads) // still alive?
		{
			fst = tid * step_size * 2;
			snd = fst + step_size;
			input[fst] += input[snd];
		}

	    if(tid == (number_of_threads - 1) && number_of_threads%2 != 0){
			// We have an odd count of partial sums
			// And we wish to make it even so the division of
			// The number of threads will work in the next step
			// So we add the last partial sum to the second to last sum
			fst = (tid - 1)*step_size*2;
			snd = tid*step_size*2;
			input[fst] += input[snd];
		}

		step_size <<= 1; 
		number_of_threads >>= 1;
        __syncthreads();
    }
}

int main()
{
	const auto count = 30;
	const int size = count * sizeof(int);
	int h[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

	int* d;
	
	cudaMalloc(&d, size);
	cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

	sum<<<1, count / 2 >>>(d);

	int result;
	cudaMemcpy(&result, d, sizeof(int), cudaMemcpyDeviceToHost);

	// cout << "Sum is " << result << endl;
    printf("Sum is: %d\n\n", result);

	cudaFree(d);
	return 0;
}