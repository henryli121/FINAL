#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>


const int MIN_DATA = 8;
const int MAX_DATA = 256*1024*1024;
// Function to allocate and initialize memory
void allocateAndInitMemory(char **h_data, char **d_data, size_t size) {
    // Allocate host and device memory
    *h_data = (char*)malloc(size);
    cudaMalloc((void**)d_data, size);

    // Initialize the host data
    for (size_t i = 0; i < size; i++) {
        (*h_data)[i] = i % 583; // Any arbitrary value
    }
}

int main() {
    // Declare CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float duration = 0;

    // Set precision for output
    std::cout << std::fixed << std::setprecision(5);

    // Starting at 8B, go up to 256MB
    for (size_t size = MIN_DATA; size <= MAX_DATA; size *= 2) {
        char *h_data;
        char *d_data;

        allocateAndInitMemory(&h_data, &d_data, size);

        // Copy data from host to device
        cudaEventRecord(start);
        cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&duration, start, stop);

        // Calculate bandwidth in GB/s
        float seconds = duration / 1e3;
        float gb = size / (float)1e9;
        float bandwidth_HtoD = gb / seconds;
        std::cout << "Host to Device for size: " << size << " bytes, Bandwidth: " << bandwidth_HtoD << " GB/s" << std::endl;

        // Copy data back from device to host
        cudaEventRecord(start);
        cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&duration, start, stop);

        // Calculate bandwidth in GB/s
        seconds = duration / 1e3;
        float bandwidth_DtoH = gb / seconds;
        std::cout << "Device to Host for size: " << size << " bytes, Bandwidth: " << bandwidth_DtoH << " GB/s" << std::endl;

        // Clean up memory
        free(h_data);
        cudaFree(d_data);
    }

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
