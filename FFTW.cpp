#include <iostream>
#include <fftw3.h>
#include <complex>
#include <cmath>
#include <chrono>

int main()
{
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
        std::complex<double> *wave = new std::complex<double>[nxyz];
        std::complex<double> *fft_3 = new std::complex<double>[nxyz];
        std::complex<double> *d_dx = new std::complex<double>[nxyz];
        std::complex<double> *d_dy = new std::complex<double>[nxyz];
        std::complex<double> *d_dz = new std::complex<double>[nxyz];

        // Create forward and backward FFTW plans
        fftw_plan forward_plan = fftw_plan_dft_3d(nx, ny, nz, reinterpret_cast<fftw_complex *>(wave),
                                              reinterpret_cast<fftw_complex *>(fft_3), FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_plan backward_plan = fftw_plan_dft_3d(nx, ny, nz, reinterpret_cast<fftw_complex *>(fft_3),
                                               reinterpret_cast<fftw_complex *>(fft_3), FFTW_BACKWARD, FFTW_ESTIMATE);

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

            wave[i] = std::polar(1.0, phase);
        }
        double duration = 0.0;
        
        for (int trial = 0; trial < ntrial; trial++) {
            auto start = std::chrono::high_resolution_clock::now();
             // Perform forward FFT
            fftw_execute(forward_plan);
            // Copy the forward transform to wave 
            for (int i = 0; i < nxyz; ++i)
            {
                wave[i] = fft_3[i];
            }
            // Compute d/dx
            for (int i = 0; i < nxyz; ++i)
            {
                fft_3[i] = wave[i] * std::complex<double>(0.0, kx);
            }
            fftw_execute(backward_plan);
            for (int j = 0; j < nxyz; ++j)
            {
                d_dx[j] = fft_3[j] / static_cast<double>(nxyz);
            }
            // Compute d/dy
            for (int i = 0; i < nxyz; ++i)
            {
                fft_3[i] = wave[i] * std::complex<double>(0.0, ky);
            }
            fftw_execute(backward_plan);
            for (int j = 0; j < nxyz; ++j)
            {
                d_dy[j] = fft_3[j] / static_cast<double>(nxyz);
            }
            // Compute d/dz
            for (int i = 0; i < nxyz; ++i)
            {
                fft_3[i] = wave[i] * std::complex<double>(0.0, kz);
            }
            fftw_execute(backward_plan);
            for (int j = 0; j < nxyz; ++j)
            {
                d_dz[j] = fft_3[j] / static_cast<double>(nxyz);
            }
            auto end = std::chrono::high_resolution_clock::now();
            duration += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        }

        duration /= ntrial; // Take the average
        double fp_op = 24. * n * n * n * log(n) + 6. * n * n * n;
        double mflops = fp_op / duration;
        std::cout << "Average flops for n = " << n << " is " << mflops << " MFLOPs.\n";    

        // Clean up
        fftw_destroy_plan(forward_plan);
        fftw_destroy_plan(backward_plan);
        delete[] wave;
        delete[] fft_3;
        delete[] d_dx;
        delete[] d_dy;
        delete[] d_dz; 
    }

    return 0;
}