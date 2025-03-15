import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def explicit_euler(e, num_steps, final_time):
    """
    Simulate planetary orbit using explicit Euler method
    
    Parameters:
        e (float): Eccentricity of the orbit
        num_steps (int): Number of time steps
        final_time (float): Final simulation time
    
    Returns:
        tuple: Arrays of positions and time
    """
    # Time step
    dt = final_time / num_steps
    
    # Initialize arrays
    q1 = np.zeros(num_steps + 1)
    q2 = np.zeros(num_steps + 1)
    p1 = np.zeros(num_steps + 1)
    p2 = np.zeros(num_steps + 1)
    time = np.linspace(0, final_time, num_steps + 1)
    
    # Initial conditions
    q1[0] = 1 - e
    q2[0] = 0
    p1[0] = 0
    p2[0] = np.sqrt((1 + e) / (1 - e))
    
    # Simulation loop
    for i in range(num_steps):
        # Update position
        q1[i+1] = q1[i] + dt * p1[i]
        q2[i+1] = q2[i] + dt * p2[i]
        
        # Calculate radius
        r = np.sqrt(q1[i+1]**2 + q2[i+1]**2)
        
        # Update momentum
        p1[i+1] = p1[i] - dt * q1[i+1] / r**3
        p2[i+1] = p2[i] - dt * q2[i+1] / r**3
    
    return q1, q2, time

def symplectic_euler(e, num_steps, final_time):
    """
    Simulate planetary orbit using symplectic Euler method
    
    Parameters:
        e (float): Eccentricity of the orbit
        num_steps (int): Number of time steps
        final_time (float): Final simulation time
    
    Returns:
        tuple: Arrays of positions and time
    """
    # Time step
    dt = final_time / num_steps
    
    # Initialize arrays
    q1 = np.zeros(num_steps + 1)
    q2 = np.zeros(num_steps + 1)
    p1 = np.zeros(num_steps + 1)
    p2 = np.zeros(num_steps + 1)
    time = np.linspace(0, final_time, num_steps + 1)
    
    # Initial conditions
    q1[0] = 1 - e
    q2[0] = 0
    p1[0] = 0
    p2[0] = np.sqrt((1 + e) / (1 - e))
    
    # Simulation loop
    for i in range(num_steps):
        # Update momentum first
        r = np.sqrt(q1[i]**2 + q2[i]**2)
        p1[i+1] = p1[i] - dt * q1[i] / r**3
        p2[i+1] = p2[i] - dt * q2[i] / r**3
        
        # Then update position using updated momentum
        q1[i+1] = q1[i] + dt * p1[i+1]
        q2[i+1] = q2[i] + dt * p2[i+1]
    
    return q1, q2, time

def calculate_energy(q1, q2, p1, p2):
    """
    Calculate the Hamiltonian (energy) of the system
    
    Parameters:
        q1, q2 (array): Position coordinates
        p1, p2 (array): Momentum coordinates
    
    Returns:
        array: Energy at each time step
    """
    kinetic = 0.5 * (p1**2 + p2**2)
    potential = -1 / np.sqrt(q1**2 + q2**2)
    return kinetic + potential

if __name__ == "__main__":
    # Parameters
    e = 0.6  # Eccentricity
    final_time = 200
    
    # Part A: Explicit Euler method
    q1_euler, q2_euler, time_euler = explicit_euler(e, 100000, final_time)
    
    # Part B: Symplectic Euler method
    q1_symplectic, q2_symplectic, time_symplectic = symplectic_euler(e, 400000, final_time)
    
    # Calculate energy for both methods
    p1_euler = np.gradient(q1_euler, time_euler)
    p2_euler = np.gradient(q2_euler, time_euler)
    energy_euler = calculate_energy(q1_euler, q2_euler, p1_euler, p2_euler)
    
    p1_symplectic = np.gradient(q1_symplectic, time_symplectic)
    p2_symplectic = np.gradient(q2_symplectic, time_symplectic)
    energy_symplectic = calculate_energy(q1_symplectic, q2_symplectic, p1_symplectic, p2_symplectic)
    
    # Plot orbits
    plt.figure(figsize=(12, 10))
    
    # Plot orbit comparison
    plt.subplot(2, 1, 1)
    plt.plot(q1_euler, q2_euler, 'r-', label='Explicit Euler', alpha=0.7)
    plt.plot(q1_symplectic, q2_symplectic, 'b-', label='Symplectic Euler', alpha=0.7)
    plt.plot(0, 0, 'yo', markersize=10, label='Star')  # Star at origin
    plt.grid(True)
    plt.xlabel('q1')
    plt.ylabel('q2')
    plt.title(f'Planetary Orbit Comparison (e = {e})')
    plt.legend()
    plt.axis('equal')
    
    # Plot energy conservation
    plt.subplot(2, 1, 2)
    plt.plot(time_euler, energy_euler, 'r-', label='Explicit Euler')
    plt.plot(time_symplectic, energy_symplectic, 'b-', label='Symplectic Euler')
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy Conservation Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('orbit_comparison.png')
    plt.close()
    
    # Create animation of the orbit
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.grid(True)
    ax.set_xlabel('q1')
    ax.set_ylabel('q2')
    ax.set_title(f'Planetary Orbit Animation (e = {e})')
    
    # Plot star
    ax.plot(0, 0, 'yo', markersize=10, label='Star')
    
    # Plot orbit paths
    ax.plot(q1_euler, q2_euler, 'r-', alpha=0.3, label='Explicit Euler')
    ax.plot(q1_symplectic, q2_symplectic, 'b-', alpha=0.3, label='Symplectic Euler')
    
    # Create points for animation
    point_euler, = ax.plot([], [], 'ro', markersize=6)
    point_symplectic, = ax.plot([], [], 'bo', markersize=6)
    
    ax.legend()
    
    def init():
        point_euler.set_data([], [])
        point_symplectic.set_data([], [])
        return point_euler, point_symplectic
    
    def animate(i):
        # Use fewer points for smoother animation
        idx = i * 100
        if idx < len(q1_euler):
            point_euler.set_data(q1_euler[idx], q2_euler[idx])
        if idx < len(q1_symplectic):
            point_symplectic.set_data(q1_symplectic[idx], q2_symplectic[idx])
        return point_euler, point_symplectic
    
    ani = FuncAnimation(fig, animate, frames=1000, init_func=init, blit=True, interval=50)
    ani.save('orbit_animation.gif', writer='pillow', fps=30)
    
    print("Orbit simulation completed.")
    print("Plots and animation saved in current directory.") 