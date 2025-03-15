import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# Constants
k_B = 1.38064852e-23  # Boltzmann constant (J/K)
h = 6.626e-34  # Planck constant (J·s)
c = 3e8  # Speed of light (m/s)
hbar = h / (2 * np.pi)  # Reduced Planck constant

# Part A: Evaluate the integral with variable transformation
def integrand_transformed(z):
    """
    Transformed integrand for black body radiation.
    The transformation is x = z/(1-z) to change from [0, ∞) to [0, 1)
    """
    x = z / (1 - z)
    # Additional factor due to change of variables: dx/dz = 1/(1-z)^2
    jacobian = 1 / (1 - z)**2
    return (x**3 / (np.exp(x) - 1)) * jacobian

def calculate_stefan_boltzmann_constant():
    """
    Calculate the Stefan-Boltzmann constant using numerical integration
    """
    # Prefactor calculation
    prefactor = (k_B**4) / (c**2 * (hbar**3) * 4 * (np.pi**2))
    
    # Part B: Using fixed_quad for integration from 0 to 1 (transformed)
    result_fixed, _ = integrate.fixed_quad(integrand_transformed, 0, 1, n=100)
    sigma_fixed = prefactor * result_fixed * (2 * np.pi**5)
    
    # Part C: Using quad for direct integration from 0 to infinity
    def original_integrand(x):
        return x**3 / (np.exp(x) - 1)
    
    result_quad, _ = integrate.quad(original_integrand, 0, np.inf)
    sigma_quad = prefactor * result_quad * (2 * np.pi**5)
    
    # True value of Stefan-Boltzmann constant
    true_value = 5.670367e-8  # W/m²K⁴
    
    print("Results of Black Body Radiation Integration:")
    print(f"Using variable transformation and fixed_quad: {sigma_fixed:.8e} W/m²K⁴")
    print(f"Using direct integration with quad: {sigma_quad:.8e} W/m²K⁴")
    print(f"True value: {true_value:.8e} W/m²K⁴")
    print(f"Relative error (fixed_quad): {abs(sigma_fixed - true_value) / true_value * 100:.6f}%")
    print(f"Relative error (quad): {abs(sigma_quad - true_value) / true_value * 100:.6f}%")
    
    return sigma_fixed, sigma_quad, true_value

if __name__ == "__main__":
    # Calculate Stefan-Boltzmann constant
    sigma_fixed, sigma_quad, true_value = calculate_stefan_boltzmann_constant()
    
    # Plot the integrand for visualization
    z_values = np.linspace(0, 0.999, 1000)
    integrand_values = [integrand_transformed(z) for z in z_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(z_values, integrand_values)
    plt.title('Transformed Integrand for Black Body Radiation')
    plt.xlabel('z (transformed variable)')
    plt.ylabel('Integrand value')
    plt.grid(True)
    plt.savefig('integrand_plot.png')
    plt.close()
    
    print("\nPlot of the transformed integrand has been saved.") 