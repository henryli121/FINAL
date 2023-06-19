#include<iostream>
#include<vector>
#include<cmath>

// Function to find the value of the polynomial
double polynomial_value(std::vector<double>& coeffs, double x) {
    double val = 0.0;
    for(int i = coeffs.size()-1; i >= 0; --i) {
        val = val * x + coeffs[i];
    }
    return val;
}

// Function to find the derivative of the polynomial
double derivative_value(std::vector<double>& coeffs, double x) {
    double val = 0.0;
    for(int i = coeffs.size()-1; i > 0; --i) {
        val = val * x + coeffs[i]*i;
    }
    return val;
}

// Function to implement the Newton Raphson Method
double newtonRaphson(std::vector<double>& coeffs, double x) {
    const double EPSILON = 0.0001; // precision
    double h = polynomial_value(coeffs, x) / derivative_value(coeffs, x);
    while(fabs(h) >= EPSILON) {
        h = polynomial_value(coeffs, x) / derivative_value(coeffs, x);
        // x(i+1) = x(i) - f(x) / f'(x)
        x = x - h;
    }
    return x;
}

bool isDistinct(const std::vector<double>& roots, double root, double EPSILON) {
    return std::none_of(roots.begin(), roots.end(), 
        [root, EPSILON](double existingRoot) { return fabs(existingRoot - root) < EPSILON; });
}

int main() {
    int degree;
    std::cout << "Enter the degree of the polynomial: ";
    std::cin >> degree;
  
    std::vector<double> coefficients(degree + 1);

    for (int i = degree; i >= 0; --i) {
        std::cout << "Enter coefficient " << i << ": ";
        std::cin >> coefficients[i];
    }
  
    double domainStart, domainEnd;
    std::cout << "Enter the start of the domain: ";
    std::cin >> domainStart;
    std::cout << "Enter the end of the domain: ";
    std::cin >> domainEnd;
  
    std::cout << "Roots found:\n";
    std::vector<double> roots;

    const int numGuesses = 100;
    double guessIncrement = (domainEnd - domainStart) / numGuesses;
    const double EPSILON = 0.001;
    for(int i = 0; i <= numGuesses; ++i) {
        double initialGuess = domainStart + guessIncrement * i;
        double root = newtonRaphson(coefficients, initialGuess);
        // Check if the root is in the domain and is distinct
        if(root >= domainStart && root <= domainEnd && isDistinct(roots, root, EPSILON)) {
            roots.push_back(root);
            std::cout << root << "\n";
        }
    }

    return 0;
}
