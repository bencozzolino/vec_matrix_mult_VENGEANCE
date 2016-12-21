#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <time.h>
#include <cuda_runtime_api.h>
/*Realization of the multiplication between a vector and a matrix, on the GPU.

It realizes multiplication between a vector of 4 element and a matrix (4x4),
so as result is expected a vector of 4 element.


We initialized the input array on the host (CPU) and trasferred to device (GPU).
Then, CPU launches the kernel which is elaborated on GPU.
Finally, the results is trasferred from GPU to CPU and printed out*/


//old kernel

#define N 4
__global__ void dot(int *a, int *b, int *c) {
	// Shared memory for results of multiplication
	__shared__ int temp[N];
	temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
	// Thread 0 sums the pairwise products
	if (0 == threadIdx.x) {
		int sum = 0;
		for (int i = 0; i < N; i++)
			sum += temp[i];
		*c = sum;
	}
}


__global__ void vector_matrix_mult(int *d_mat, int *d_out0, int *d_out1, int *d_out2, int *d_out3, const int M)
{


	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int tjd = threadIdx.y + blockIdx.y*blockDim.y;

	if (tid < M) {

		d_out0[tid] = d_mat[N*tid];
		d_out1[tid] = d_mat[N*tid + 1];
		d_out2[tid] = d_mat[N*tid + 2];
		d_out3[tid] = d_mat[N*tid + 3];



	}
}


int main(int argc, char ** argv) {



	float elapsed = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	const int M = N;
	const int ARRAY_SIZE = N;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
	const int MATRIX_SIZE = ARRAY_SIZE*ARRAY_SIZE;
	const int MATRIX_BYTES = MATRIX_SIZE * sizeof(int);

	// generate the input vector and input matrix on the host

	int *c0, *c1, *c2, *c3;
	int *d_c0, *d_c1, *d_c2, *d_c3;
	int *h_out0, *h_out1, *h_out2, *h_out3;
	int *h_vec;
	//allocate space for the variables on the host
	h_out0 = (int *)malloc(ARRAY_BYTES);
	h_out1 = (int *)malloc(ARRAY_BYTES);
	h_out2 = (int *)malloc(ARRAY_BYTES);
	h_out3 = (int *)malloc(ARRAY_BYTES);

	h_vec = (int*)malloc(ARRAY_BYTES);

	h_vec[0] = 0;
	h_vec[1] = 0;
	h_vec[2] = 0;
	h_vec[3] = 1;


	int h_mat[ARRAY_SIZE][ARRAY_SIZE] = { 2, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 };


	c0 = (int *)malloc(sizeof(int));
	c1 = (int *)malloc(sizeof(int));
	c2 = (int *)malloc(sizeof(int));
	c3 = (int *)malloc(sizeof(int));
	*c0 = 0;
	*c1 = 0;
	*c2 = 0;
	*c3 = 0;




	//declare GPU memory pointers

	int * d_vec;
	int * d_mat;
	int * d_out0;
	int * d_out1;
	int * d_out2;
	int * d_out3;


	//allocate GPU memory

	cudaMalloc((void **)&d_c0, sizeof(int));
	cudaMalloc((void **)&d_c1, sizeof(int));
	cudaMalloc((void **)&d_c2, sizeof(int));
	cudaMalloc((void **)&d_c3, sizeof(int));
	cudaMalloc((void**)&d_vec, ARRAY_BYTES);

	cudaMalloc((void**)&d_mat, MATRIX_BYTES);
	cudaMalloc((void**)&d_out0, ARRAY_BYTES);
	cudaMalloc((void**)&d_out1, ARRAY_BYTES);
	cudaMalloc((void**)&d_out2, ARRAY_BYTES);
	cudaMalloc((void**)&d_out3, ARRAY_BYTES);

	//transfer the input from CPU to GPU

	cudaMemcpy(d_vec, h_vec, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out0, h_out0, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out1, h_out1, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out2, h_out2, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out3, h_out3, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat, h_mat, MATRIX_BYTES, cudaMemcpyHostToDevice);

	cudaMemcpy(d_c0, c0, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c1, c1, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c2, c2, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c3, c3, sizeof(int), cudaMemcpyHostToDevice);

	//Cuda stream
	cudaStream_t stream1, stream2, stream3;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);


	cudaEventRecord(start, 0);
	//launch the kernel
	vector_matrix_mult << <1, ARRAY_SIZE >> > (d_mat, d_out0, d_out1, d_out2, d_out3, M);
	cudaDeviceSynchronize();

	dot << <1, ARRAY_SIZE >> > (d_vec, d_out0, d_c0);

	dot << <1, ARRAY_SIZE, 0, stream1 >> > (d_vec, d_out1, d_c1);

	dot << <1, ARRAY_SIZE, 0, stream2 >> > (d_vec, d_out2, d_c2);

	dot << <1, ARRAY_SIZE, 0, stream3 >> > (d_vec, d_out3, d_c3);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);

	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	cudaStreamDestroy(stream3);

	//trasfer the results from GPU to CPU
	cudaMemcpy(h_out0, d_out0, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_out1, d_out1, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_out2, d_out2, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_out3, d_out3, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(c0, d_c0, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(c1, d_c1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(c2, d_c2, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(c3, d_c3, sizeof(int), cudaMemcpyDeviceToHost);




	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);

	//print out the resulting array
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%d", h_out0[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%d", h_out1[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%d", h_out2[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%d", h_out3[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}
	printf("\n""\n""\n");
	printf("%d" "\t", *c0);
	printf("%d" "\t", *c1);
	printf("%d" "\t", *c2);
	printf("%d" "\t", *c3);

	printf("The GPU time elapsed is %.6f ms \"", elapsed);

	//free GPU location memory

	cudaFree(d_vec);
	cudaFree(d_c0);
	cudaFree(d_c1);
	cudaFree(d_c2);
	cudaFree(d_c3);
	cudaFree(d_mat);
	cudaFree(d_out0);
	cudaFree(d_out1);
	cudaFree(d_out2);
	cudaFree(d_out3);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}



