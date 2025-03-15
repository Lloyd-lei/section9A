# Task 1: Black Body Radiation

This task involves calculating the Stefan-Boltzmann constant by evaluating the integral in the black body radiation formula.

## Problem Description

The total rate at which energy is radiated by a black body per unit area over all frequencies is given by:

W = (2πk⁴T⁴)/(c²h³) ∫₀^∞ (x³)/(e^x-1) dx

Where:
- k = 1.38064852 × 10⁻²³ J/K (Boltzmann constant)
- h = 6.626 × 10⁻³⁴ J·s (Planck constant)
- c = 3 × 10⁸ m/s (Speed of light)

## Tasks

### Part A
Evaluate the integral by changing variables from an infinite range to a finite range using the transformation:
z = x/(1+x) or equivalently x = z/(1-z)

### Part B
Calculate the Stefan-Boltzmann constant (σ) using the fixed_quad function to perform the integral.

### Part C
Use the built-in 'quad' function to perform the integration from 0 to ∞ and compare the results.

## Running the Code

To run the code, execute the following command in the terminal:

```bash
python black_body_radiation.py
```

## Expected Output

The code will:
1. Calculate the Stefan-Boltzmann constant using both integration methods
2. Compare the results with the true value (5.670367 × 10⁻⁸ W/m²K⁴)
3. Generate a plot of the transformed integrand

## Dependencies

- NumPy
- SciPy
- Matplotlib 