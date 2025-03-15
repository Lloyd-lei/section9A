import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import os

# Define the potential function: H = θ^4 - 8θ^2 - 2cos(4πθ)
def potential(theta):
    """
    Calculate the potential energy for the noisy φ^4 theory
    
    Parameters:
        theta (float or array): Order parameter
    
    Returns:
        float or array: Potential energy
    """
    return theta**4 - 8 * theta**2 - 2 * np.cos(4 * np.pi * theta)

# Calculate the gradient of the potential
def gradient(theta):
    """
    Calculate the gradient of the potential energy
    
    Parameters:
        theta (float): Order parameter
    
    Returns:
        float: Gradient of the potential
    """
    return 4 * theta**3 - 16 * theta + 8 * np.pi * np.sin(4 * np.pi * theta)

# Part A: Gradient Descent Method
def gradient_descent(initial_theta, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    """
    Perform gradient descent optimization
    
    Parameters:
        initial_theta (float): Initial guess for the order parameter
        learning_rate (float): Learning rate for gradient descent
        max_iterations (int): Maximum number of iterations
        tolerance (float): Convergence tolerance
    
    Returns:
        tuple: List of theta values, potential values, and convergence flag
    """
    theta = initial_theta
    theta_history = [theta]
    potential_history = [potential(theta)]
    
    for i in range(max_iterations):
        # Calculate gradient
        grad = gradient(theta)
        
        # Adaptive learning rate
        if i > 0 and potential_history[-1] > potential_history[-2]:
            learning_rate *= 0.5
        else:
            learning_rate *= 1.05
        
        # Update theta
        theta_new = theta - learning_rate * grad
        
        # Store history
        theta_history.append(theta_new)
        potential_history.append(potential(theta_new))
        
        # Check convergence
        if abs(theta_new - theta) < tolerance:
            return theta_history, potential_history, True
        
        theta = theta_new
    
    return theta_history, potential_history, False

# Part B: Metropolis-Hastings Algorithm
def metropolis_hastings(initial_theta, beta=1.0, step_size=0.1, max_iterations=10000):
    """
    Perform Metropolis-Hastings optimization
    
    Parameters:
        initial_theta (float): Initial guess for the order parameter
        beta (float): Inverse temperature (1/kT)
        step_size (float): Standard deviation for the proposal distribution
        max_iterations (int): Maximum number of iterations
    
    Returns:
        tuple: List of theta values and potential values
    """
    theta = initial_theta
    theta_history = [theta]
    potential_history = [potential(theta)]
    
    for _ in range(max_iterations):
        # Propose a new state
        theta_proposed = theta + np.random.normal(0, step_size)
        
        # Calculate energy difference
        current_potential = potential(theta)
        proposed_potential = potential(theta_proposed)
        delta_H = proposed_potential - current_potential
        
        # Metropolis acceptance criterion
        if delta_H <= 0:
            # Accept if the proposed state has lower energy
            theta = theta_proposed
        else:
            # Accept with probability exp(-beta * delta_H)
            acceptance_probability = np.exp(-beta * delta_H)
            if np.random.random() < acceptance_probability:
                theta = theta_proposed
        
        # Store history
        theta_history.append(theta)
        potential_history.append(potential(theta))
    
    return theta_history, potential_history

# Part C: Simulated Annealing
def simulated_annealing(initial_theta, beta_initial=0.1, beta_final=10.0, step_size=0.1, max_iterations=10000):
    """
    Perform simulated annealing optimization
    
    Parameters:
        initial_theta (float): Initial guess for the order parameter
        beta_initial (float): Initial inverse temperature
        beta_final (float): Final inverse temperature
        step_size (float): Standard deviation for the proposal distribution
        max_iterations (int): Maximum number of iterations
    
    Returns:
        tuple: List of theta values, potential values, and beta values
    """
    theta = initial_theta
    theta_history = [theta]
    potential_history = [potential(theta)]
    
    # Linear cooling schedule
    beta_values = np.linspace(beta_initial, beta_final, max_iterations)
    beta_history = [beta_initial]
    
    for i in range(max_iterations):
        # Current inverse temperature
        beta = beta_values[i]
        
        # Propose a new state
        theta_proposed = theta + np.random.normal(0, step_size)
        
        # Calculate energy difference
        current_potential = potential(theta)
        proposed_potential = potential(theta_proposed)
        delta_H = proposed_potential - current_potential
        
        # Metropolis acceptance criterion with current temperature
        if delta_H <= 0:
            # Accept if the proposed state has lower energy
            theta = theta_proposed
        else:
            # Accept with probability exp(-beta * delta_H)
            acceptance_probability = np.exp(-beta * delta_H)
            if np.random.random() < acceptance_probability:
                theta = theta_proposed
        
        # Store history
        theta_history.append(theta)
        potential_history.append(potential(theta))
        beta_history.append(beta)
    
    return theta_history, potential_history, beta_history

# Create a visualization of the potential function
def plot_potential():
    """
    Create a visualization of the potential function
    
    Returns:
        tuple: Arrays of theta values and potential values
    """
    theta_range = np.linspace(-3, 3, 1000)
    potential_values = potential(theta_range)
    
    plt.figure(figsize=(10, 6))
    plt.plot(theta_range, potential_values)
    plt.grid(True)
    plt.xlabel('θ')
    plt.ylabel('Potential H(θ)')
    plt.title('Noisy φ⁴ Potential: H = θ⁴ - 8θ² - 2cos(4πθ)')
    
    # Mark the global minimum
    min_idx = np.argmin(potential_values)
    min_theta = theta_range[min_idx]
    min_potential = potential_values[min_idx]
    plt.plot(min_theta, min_potential, 'ro', markersize=8, label=f'Global Minimum: θ = {min_theta:.4f}')
    
    plt.legend()
    plt.savefig('potential_function.png')
    
    return theta_range, potential_values

# Create a video of the gradient descent process
def create_gradient_descent_video(theta_range, potential_values, theta_history, initial_theta):
    """
    Create a video showing the gradient descent optimization process
    
    Parameters:
        theta_range (array): Range of theta values for plotting
        potential_values (array): Potential values corresponding to theta_range
        theta_history (list): History of theta values during optimization
        initial_theta (float): Initial guess for theta
    """
    # Create output directory if it doesn't exist
    os.makedirs('frames', exist_ok=True)
    
    # Create frames for the video
    for i, theta in enumerate(theta_history):
        plt.figure(figsize=(10, 6))
        plt.plot(theta_range, potential_values)
        plt.grid(True)
        plt.xlabel('θ')
        plt.ylabel('Potential H(θ)')
        plt.title(f'Gradient Descent Optimization (Initial θ = {initial_theta})')
        
        # Mark the current position
        current_potential = potential(theta)
        plt.plot(theta, current_potential, 'ro', markersize=8, 
                 label=f'Current: θ = {theta:.4f}, H = {current_potential:.4f}')
        
        # Mark the initial position
        initial_potential = potential(initial_theta)
        plt.plot(initial_theta, initial_potential, 'go', markersize=8, 
                 label=f'Initial: θ = {initial_theta:.4f}')
        
        plt.legend()
        plt.xlim(min(theta_range), max(theta_range))
        plt.ylim(min(potential_values) - 1, max(min(potential_values) + 20, current_potential + 5))
        
        # Save frame
        frame_path = f'frames/frame_{i:04d}.png'
        plt.savefig(frame_path)
        plt.close()
    
    # Create video from frames
    frame = cv2.imread('frames/frame_0000.png')
    height, width, _ = frame.shape
    
    video_path = f'gradient_descent_video_initial_{initial_theta}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
    
    for i in range(len(theta_history)):
        frame_path = f'frames/frame_{i:04d}.png'
        frame = cv2.imread(frame_path)
        video.write(frame)
    
    video.release()
    
    # Clean up frames
    for i in range(len(theta_history)):
        frame_path = f'frames/frame_{i:04d}.png'
        os.remove(frame_path)

# Compare optimization methods
def compare_methods(initial_guesses):
    """
    Compare different optimization methods starting from various initial guesses
    
    Parameters:
        initial_guesses (list): List of initial theta values
    """
    # Create visualization of the potential function
    theta_range, potential_values = plot_potential()
    
    # Results storage
    results = {
        'Gradient Descent': [],
        'Metropolis-Hastings': [],
        'Simulated Annealing': []
    }
    
    # Run optimizations for each initial guess
    for initial_theta in initial_guesses:
        print(f"\nOptimizing with initial θ = {initial_theta}")
        
        # Gradient Descent
        print("Running Gradient Descent...")
        theta_history_gd, potential_history_gd, converged = gradient_descent(initial_theta)
        final_theta_gd = theta_history_gd[-1]
        final_potential_gd = potential_history_gd[-1]
        results['Gradient Descent'].append((final_theta_gd, final_potential_gd, len(theta_history_gd)))
        
        # Create video for gradient descent
        create_gradient_descent_video(theta_range, potential_values, theta_history_gd, initial_theta)
        
        # Metropolis-Hastings with different beta values
        print("Running Metropolis-Hastings...")
        beta_values = [0.1, 1.0, 10.0]
        for beta in beta_values:
            theta_history_mh, potential_history_mh = metropolis_hastings(initial_theta, beta=beta)
            final_theta_mh = theta_history_mh[-1]
            final_potential_mh = potential_history_mh[-1]
            results[f'Metropolis-Hastings (β={beta})'] = (final_theta_mh, final_potential_mh, len(theta_history_mh))
        
        # Simulated Annealing
        print("Running Simulated Annealing...")
        theta_history_sa, potential_history_sa, beta_history = simulated_annealing(initial_theta)
        final_theta_sa = theta_history_sa[-1]
        final_potential_sa = potential_history_sa[-1]
        results['Simulated Annealing'].append((final_theta_sa, final_potential_sa, len(theta_history_sa)))
        
        # Plot comparison for this initial guess
        plt.figure(figsize=(12, 8))
        
        # Plot potential function
        plt.plot(theta_range, potential_values, 'k-', alpha=0.3)
        
        # Plot optimization trajectories
        plt.plot(theta_history_gd, potential_history_gd, 'r.-', label='Gradient Descent', alpha=0.7)
        plt.plot(theta_history_mh, potential_history_mh, 'b.-', label=f'Metropolis-Hastings (β={beta_values[-1]})', alpha=0.7)
        plt.plot(theta_history_sa, potential_history_sa, 'g.-', label='Simulated Annealing', alpha=0.7)
        
        # Mark initial and final points
        plt.plot(initial_theta, potential(initial_theta), 'ko', markersize=8, label=f'Initial θ = {initial_theta}')
        plt.plot(final_theta_gd, final_potential_gd, 'ro', markersize=8)
        plt.plot(final_theta_mh, final_potential_mh, 'bo', markersize=8)
        plt.plot(final_theta_sa, final_potential_sa, 'go', markersize=8)
        
        plt.grid(True)
        plt.xlabel('θ')
        plt.ylabel('Potential H(θ)')
        plt.title(f'Optimization Methods Comparison (Initial θ = {initial_theta})')
        plt.legend()
        
        plt.savefig(f'comparison_initial_{initial_theta}.png')
        plt.close()
    
    # Print summary of results
    print("\nOptimization Results Summary:")
    print("-" * 80)
    print(f"{'Method':<25} {'Initial θ':<12} {'Final θ':<12} {'Final H(θ)':<15} {'Iterations':<10}")
    print("-" * 80)
    
    for i, initial_theta in enumerate(initial_guesses):
        for method in ['Gradient Descent', 'Simulated Annealing']:
            final_theta, final_potential, iterations = results[method][i]
            print(f"{method:<25} {initial_theta:<12.4f} {final_theta:<12.4f} {final_potential:<15.6f} {iterations:<10}")
        
        for beta in beta_values:
            method_name = f'Metropolis-Hastings (β={beta})'
            if method_name in results:
                final_theta, final_potential, iterations = results[method_name]
                print(f"{method_name:<25} {initial_theta:<12.4f} {final_theta:<12.4f} {final_potential:<15.6f} {iterations:<10}")
    
    print("-" * 80)

if __name__ == "__main__":
    # Create output directory
    os.makedirs('frames', exist_ok=True)
    
    # Initial guesses as specified in the problem
    initial_guesses = [-1.0, 0.5, 3.0]
    
    # Run comparison of optimization methods
    compare_methods(initial_guesses) 