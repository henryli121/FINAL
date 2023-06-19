#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define START 16
#define MAX_DIMENSION 8192
#define NTRIAL 3

int main() {   
    for (int n = START; n <= MAX_DIMENSION; n*= 2) {
         // Allocate host memory for square matrices
        double *h_A = new double[n * n];
        double *h_B = new double[n * n];
        double *h_C = new double[n * n];

        // Initialize input matrices
        for (int i = 0; i < n * n; ++i) {
            h_A[i] = static_cast<double>(rand()) / RAND_MAX;
            h_B[i] = static_cast<double>(rand()) / RAND_MAX;
            h_C[i] = static_cast<double>(rand()) / RAND_MAX;
        }

        // Allocate device memory for matrices
        double *d_A, *d_B, *d_C;
        cudaMalloc((void **)&d_A, n * n * sizeof(double));
        cudaMalloc((void **)&d_B, n * n * sizeof(double));
        cudaMalloc((void **)&d_C, n * n * sizeof(double));

        // Copy input matrices from host to device
        cudaMemcpy(d_A, h_A, n * n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, n * n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, h_C, n * n * sizeof(double), cudaMemcpyHostToDevice);

        // Create cuBLAS handle
        cublasHandle_t handle;
        cublasCreate(&handle);

        // Perform matrix multiplication
        const double alpha = rand();
        const double beta = rand();
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        double totaltime = 0;

        for (int i = 0; i < NTRIAL; i++) {
            cudaEventRecord(start, 0);
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            // Calculate elapsed time
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            totaltime += milliseconds;
        }

        float avgTimeInSec = totaltime / (NTRIAL * 1000);  // convert time to seconds
        double flops = (2.0 * n * n * n + 2.0 * n * n) / (avgTimeInSec * 1e6); // megaFLOPS
        // Print performance information
         std::cout << "Matrix dimension: " << n << ", PERF: " << flops << " MFLOPS" << std::endl;

        // Destroy cuBLAS handle
        cublasDestroy(handle);

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // Free host memory
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
    }
    return 0;
}

