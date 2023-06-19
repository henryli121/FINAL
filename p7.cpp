#include "mem_swaps.hpp"
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <utility> // For std :: pair

std::pair<int , int> getRandomIndices (int n) {
    int i = std::rand()%n;
    int j = std::rand() % (n - 1); if (j >= i) {
        j++;
    }
return std::make_pair(i , j); 
}

int main() {
    std::srand(std::time(0));
    const int NTRIAL = 3;
    for (int dim = 16; dim <= 16384; dim *= 2) {
        std::vector<double> matrix(dim * dim);
        std::pair<int, int> rowIndices = getRandomIndices(dim);
        std::pair<int, int> colIndices = getRandomIndices(dim);

        double totalRowtime = 0.0;
        double totalColtime = 0.0;

        for (int ntrial = 0; ntrial < NTRIAL; ntrial++) {
            for (auto& elem : matrix) {
                elem = static_cast<double>(std::rand()) / RAND_MAX;
            }

            auto start = std::chrono::high_resolution_clock::now();
            swapRows(matrix, dim, dim, rowIndices.first, rowIndices.second);
            auto end = std::chrono::high_resolution_clock::now();
            totalRowtime += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

            start = std::chrono::high_resolution_clock::now();
            swapCols(matrix, dim, dim, colIndices.first, colIndices.second);
            end = std::chrono::high_resolution_clock::now();
            totalColtime += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
    
    std::cout << "For matrix size n: " << dim << ", the row swap time t: " 
                  << totalRowtime / NTRIAL << " ns\n";

    std::cout << "For matrix size n:  " << dim << ", the column swap time t: " 
                  << totalColtime / NTRIAL << " ns\n";
    }
    return 0;
}
