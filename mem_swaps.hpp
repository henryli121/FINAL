#include <chrono>
#include <algorithm>
#include <iostream>
#include <cstdlib>

void swapRows(std::vector<double>& matrix, int nRows, int nCols, int i, int j) {
    for (int col = 0; col < nCols; ++col) {
        std::swap(matrix[i + col * nRows], matrix[j + col * nRows]);
    }
}

void swapCols(std::vector<double>& matrix, int nRows, int nCols, int i, int j) {
    std::swap_ranges(matrix.begin() + i * nRows, matrix.begin() + (i + 1) * nRows, matrix.begin() + j * nRows);
}