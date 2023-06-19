#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <chrono>
#include "file_swaps.hpp"

const int START = 16;
const int MAX_DIMENSION = 16384;
const int NTRIAL = 3;

std::pair<int, int> getRandomIndices(int n) {
    int i = std::rand() % n;
    int j = std::rand() % (n - 1);
    if (j >= i) {
        j++;
    }
    return std::make_pair(i, j);
}

int main(int argc, char *argv[]) {
    std::string filename = "matrix.bin";
    for (int n = START; n <= MAX_DIMENSION; n *= 2) {
        // Generate the matrix
        std::vector<double> matrix(n * n);
        for (auto& elem : matrix) {
            elem = static_cast<double>(std::rand()) / RAND_MAX;
        }

        std::fstream file(filename, std::ios::out | std::ios::binary);
        file.write(reinterpret_cast<char *>(&matrix[0]), n * n * sizeof(double));
        file.close();

        double totalRowtime = 0.0;
        double totalColtime = 0.0;

        for (int ntrial = 0; ntrial < NTRIAL; ntrial++) {
            std::fstream fileToSwap(filename, std::ios::in | std::ios::out | std::ios::binary);

            std::pair<int, int> rowIndices = getRandomIndices(n);
            std::pair<int, int> colIndices = getRandomIndices(n);

            auto start = std::chrono::high_resolution_clock::now();
            swapRows(fileToSwap, n, n, rowIndices.first, rowIndices.second);
            auto end = std::chrono::high_resolution_clock::now();
            totalRowtime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            start = std::chrono::high_resolution_clock::now();
            swapCols(fileToSwap, n, n, colIndices.first, colIndices.second);
            end = std::chrono::high_resolution_clock::now();
            totalColtime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            fileToSwap.close();
        }
    std::cout << "For matrix size n: " << n << ", the row swap time t: " 
                  << totalRowtime / NTRIAL << " ms\n";

    std::cout << "For matrix size n:  " << n << ", the column swap time t: " 
                  << totalColtime / NTRIAL << " ms\n";

    // Delete the test file after each problem size
    std::remove(filename.c_str());
    }

    return 0;
}