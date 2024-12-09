#include <iostream>

int main() {
    // Declare variables
    int a, b;

    // Input values
    std::cout << "Enter the first number: ";
    std::cin >> a;

    std::cout << "Enter the second number: ";
    std::cin >> b;

    // Perform arithmetic operations
    std::cout << "Results:" << std::endl;
    std::cout << "Sum: " << a + b << std::endl;
    std::cout << "Difference: " << a - b << std::endl;
    std::cout << "Product: " << a * b << std::endl;
    std::cout << "Quotient: " << (b != 0 ? a / b : 0) << std::endl;

    return 0;
}
