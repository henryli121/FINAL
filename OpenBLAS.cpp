#include <iostream>
#include <chrono>
#include <cblas.h>
#include <vector>
#include <cstdlib>

int main()
{
    const int MAX_DIMENSION = 8192;
    const double alpha = static_cast<double>(rand()) / RAND_MAX;
    const double beta = static_cast<double>(rand()) / RAND_MAX;
    const int NTRIAL = 3;

    for (int N = 16; N <= MAX_DIMENSION; N *= 2)
    {
        std::vector<double> A(N * N);
        std::vector<double> B(N * N);
        std::vector<double> C(N * N);

        double fp_op = 2.0 * N * N * N + 2.0 * N * N;
        // Initialize matrices A, B, and C with random values
        for (int i = 0; i < N * N; ++i)
        {
            A[i] = static_cast<double>(rand()) / RAND_MAX;
            B[i] = static_cast<double>(rand()) / RAND_MAX;
            C[i] = static_cast<double>(rand()) / RAND_MAX;
        }

        // Perform matrix multiplication and measure the execution time
        long double totaltime = 0.L;

        for (int j = 0; j < NTRIAL; ++j) {
            // Perform matrix multiplication and measure the execution time
            auto start = std::chrono::high_resolution_clock::now();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A.data(), N, B.data(), N, beta, C.data(), N);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            totaltime += duration.count();
        }

        long double avg_time = totaltime * 1.e-9 / NTRIAL;

        std::cout << "Matrix dimension: " << N << ", PERF: " << fp_op / 1.e6 / avg_time << " MFLOPS" << std::endl;
    }

    return 0;
}
