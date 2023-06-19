#include <iostream>
#include <cufft.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <vector>
#include <chrono>

__device__ cuDoubleComplex complexToCuComplex(double real, double imag)
{
    return make_cuDoubleComplex(real, imag);
}

__device__ void cuComplexToComplex(cuDoubleComplex c, double* real, double* imag)
{
    *real = cuCreal(c);
    *imag = cuCimag(c);
}

__device__ cuDoubleComplex operator*(cuDoubleComplex a, cuDoubleComplex b)
{
    return make_cuDoubleComplex(cuCreal(a) * cuCreal(b) - cuCimag(a) * cuCimag(b),
                                cuCreal(a) * cuCimag(b) + cuCimag(a) * cuCreal(b));
}

__device__ cuDoubleComplex operator/(cuDoubleComplex a, double b)
{
    return make_cuDoubleComplex(cuCreal(a) / b, cuCimag(a) / b);
}

__global__ void computeDdx(cuDoubleComplex* wave_gpu, cuDoubleComplex* fft_3_gpu, double kx, int nxyz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nxyz) {
        cuDoubleComplex factor = make_cuDoubleComplex(0.0, kx);
        fft_3_gpu[idx] = wave_gpu[idx] * factor;
    }
}

__global__ void computeScale(cuDoubleComplex* fft_3_gpu, cuDoubleComplex* d_dx_gpu, double scale, int nxyz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nxyz) {
        d_dx_gpu[idx] = fft_3_gpu[idx] / scale;
    }
}

int main()
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    // Parameters
    const int ntrial = 3;
    const double lx = 1.0; // Length of the cubic lattice in x-direction
    const double ly = 1.0; // Length of the cubic lattice in y-direction
    const double lz = 1.0; // Length of the cubic lattice in z-direction
    const int START = 16;
    const int END = 256;

    for (int n = START; n <= END; n *= 2) {
        const int nx = n;   // Number of lattice points in x-direction
        const int ny = n;   // Number of lattice points in y-direction
        const int nz = n;   // Number of lattice points in z-direction
        const int nxyz = nx * ny * nz;

        // Wave vector components in reciprocal lattice units
        const double kx = 2.0 * M_PI / lx * 2; // Wave vector component in x-direction
        const double ky = 2.0 * M_PI / ly * 3; // Wave vector component in y-direction
        const double kz = 2.0 * M_PI / lz * 4; // Wave vector component in z-direction

        // Compute distances between spatial lattice sites
        const double dx = lx / nx;
        const double dy = ly / ny;
        const double dz = lz / nz;

            // Allocate memory for wave, fft_3, d_dx, d_dy, and d_dz arrays
        cuDoubleComplex* wave = new cuDoubleComplex[nxyz];
        cuDoubleComplex* fft_3 = new cuDoubleComplex[nxyz];
        cuDoubleComplex* d_dx = new cuDoubleComplex[nxyz];
        cuDoubleComplex* d_dy = new cuDoubleComplex[nxyz];
        cuDoubleComplex* d_dz = new cuDoubleComplex[nxyz];

        // Allocate memory on the GPU for wave, fft_3, d_dx, d_dy, and d_dz arrays
        cuDoubleComplex* wave_gpu;
        cuDoubleComplex* fft_3_gpu;
        cuDoubleComplex* d_dx_gpu;
        cuDoubleComplex* d_dy_gpu;
        cuDoubleComplex* d_dz_gpu;

        cudaMalloc((void**)&wave_gpu, nxyz * sizeof(cuDoubleComplex));
        cudaMalloc((void**)&fft_3_gpu, nxyz * sizeof(cuDoubleComplex));
        cudaMalloc((void**)&d_dx_gpu, nxyz * sizeof(cuDoubleComplex));
        cudaMalloc((void**)&d_dy_gpu, nxyz * sizeof(cuDoubleComplex));
        cudaMalloc((void**)&d_dz_gpu, nxyz * sizeof(cuDoubleComplex));

        // Create cuFFT plans
        cufftHandle forward_plan;
        cufftHandle backward_plan;
        cufftPlan3d(&forward_plan, nx, ny, nz, CUFFT_Z2Z);
        cufftPlan3d(&backward_plan, nx, ny, nz, CUFFT_Z2Z);
        
        // Generate complex plane wave on the cubic lattice
        for (int i = 0; i < nxyz; ++i)
        {
            const int ix = i % nx;
            const int iy = (i / nx) % ny;
            const int iz = i / (nx * ny);
            const double x = (ix - nx / 2) * dx;
            const double y = (iy - ny / 2) * dy;
            const double z = (iz - nz / 2) * dz;
            const double phase = kx * x + ky * y + kz * z;

            wave[i] = make_cuDoubleComplex(cos(phase), sin(phase));
        }

        // for CUDA kernel to compute d/dx
        int blockSize = 256;
        int numBlocks = (nxyz + blockSize - 1) / blockSize;

        cudaEventRecord(start);
        for (int trial = 0; trial < ntrial; trial++) {

            // Copy data from CPU to GPU
            cudaMemcpy(wave_gpu, wave, nxyz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

            // Perform forward FFT
            cufftExecZ2Z(forward_plan, wave_gpu, fft_3_gpu, CUFFT_FORWARD);

            // make a copy of the forward transform
            cudaMemcpy(wave_gpu, fft_3_gpu, nxyz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);

            // Compute d/dx
            computeDdx<<<numBlocks, blockSize>>>(wave_gpu, fft_3_gpu, kx, nxyz);
            cufftExecZ2Z(backward_plan, fft_3_gpu, fft_3_gpu, CUFFT_INVERSE);
            computeScale<<<numBlocks, blockSize>>>(fft_3_gpu, d_dx_gpu, static_cast<double>(nxyz),nxyz);

            // Compute d/dy
            computeDdx<<<numBlocks, blockSize>>>(wave_gpu, fft_3_gpu, ky, nxyz);
            cufftExecZ2Z(backward_plan, fft_3_gpu, fft_3_gpu, CUFFT_INVERSE);
            computeScale<<<numBlocks, blockSize>>>(fft_3_gpu, d_dy_gpu, static_cast<double>(nxyz),nxyz);

            // Compute d/dz
            computeDdx<<<numBlocks, blockSize>>>(wave_gpu, fft_3_gpu, kz, nxyz);
            cufftExecZ2Z(backward_plan, fft_3_gpu, fft_3_gpu, CUFFT_INVERSE);
            computeScale<<<numBlocks, blockSize>>>(fft_3_gpu, d_dz_gpu, static_cast<double>(nxyz),nxyz);

            // Copy data from GPU to CPU
            cudaMemcpy(d_dx, d_dx_gpu, nxyz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
            cudaMemcpy(d_dy, d_dy_gpu, nxyz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
            cudaMemcpy(d_dz, d_dz_gpu, nxyz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        double duration = milliseconds / 1000.0; // Convert from milliseconds to seconds
        duration /= ntrial; // Take the average
        double fp_op = 24. * n * n * n * log(n) + 6. * n * n * n;
        double flops = fp_op / duration;
        std::cout << "Average flops for n = " << n << " is " << flops/1.e6 << " MFLOPs.\n";    

        // Free GPU memory
        cudaFree(wave_gpu);
        cudaFree(fft_3_gpu);
        cudaFree(d_dx_gpu);
        cudaFree(d_dy_gpu);
        cudaFree(d_dz_gpu);

        // Destroy cuFFT plans
        cufftDestroy(forward_plan);
        cufftDestroy(backward_plan);

        // Free CPU memory
        delete[] wave;
        delete[] d_dx;
        delete[] d_dy;
        delete[] d_dz;
        delete[] fft_3;
    }
    // Destroy the events when done
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}