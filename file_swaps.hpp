#include <vector>
#include <fstream>
#include <algorithm>

void swapRows(std::fstream& file, int nRows, int nCols, int i, int j) {
    std::vector<double> row_i(nCols), row_j(nCols);
    //read the data from the file
    for (int col = 0; col < nCols; col++) {
        file.seekg((i + col * nRows) * sizeof(double));
        //cast data from double to char, so we can read the data
        file.read(reinterpret_cast<char*>(&row_i[col]), sizeof(double));

        file.seekg((j + col * nRows) * sizeof(double));
        file.read(reinterpret_cast<char*>(&row_j[col]), sizeof(double));
    }

    //swap the elements 
    for (int col = 0; col < nCols; ++col) {
        file.seekp((i + col * nRows) * sizeof(double));
        file.write(reinterpret_cast<const char*>(&row_j[col]), sizeof(double));

        file.seekp((j + col * nRows) * sizeof(double));
        file.write(reinterpret_cast<const char*>(&row_i[col]), sizeof(double));
    }
}

void swapCols(std::fstream& file, int nRows, int nCols, int i, int j) {
    std::vector<double> col_i(nRows), col_j(nRows);
    //read the i,j-th col from the file
    file.seekg(i * nRows * sizeof(double));
    file.read(reinterpret_cast<char*>(col_i.data()), nRows * sizeof(double));
    file.seekg(j * nRows * sizeof(double));
    file.read(reinterpret_cast<char*>(col_j.data()), nRows * sizeof(double));

    //swap the elements
    file.seekp(i * nRows * sizeof(double));
    file.write(reinterpret_cast<const char*>(col_j.data()), nRows * sizeof(double));
    file.seekp(j * nRows * sizeof(double));
    file.write(reinterpret_cast<const char*>(col_i.data()), nRows * sizeof(double));
}