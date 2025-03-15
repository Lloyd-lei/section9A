# Physics 129AL Computational Physics - Week 9A

This repository contains the solutions for the Week 9A section worksheet of UCSB's Physics 129AL Computational Physics course.

## Project Structure

The project is organized into three main directories, each corresponding to a specific task:

1. **Task1_BlackBodyRadiation**: Calculating the Stefan-Boltzmann constant using numerical integration
2. **Task2_Orbits**: Simulating planetary orbits using explicit and symplectic Euler methods
3. **Task3_GradientDescent**: Implementing and comparing optimization methods (Gradient Descent, Metropolis-Hastings, Simulated Annealing)

## Requirements

To run the code in this repository, you need the following dependencies:

```
numpy
scipy
matplotlib
opencv-python
tqdm
```

You can install these dependencies using pip:

```bash
pip install numpy scipy matplotlib opencv-python tqdm
```

## Running the Code

Each task directory contains its own Python script and README file with specific instructions. To run a task, navigate to its directory and execute the corresponding Python script.

For example:

```bash
# For Task 1
cd Task1_BlackBodyRadiation
python black_body_radiation.py

# For Task 2
cd Task2_Orbits
python orbits.py

# For Task 3
cd Task3_GradientDescent
python optimization_methods.py
```

## Task Descriptions

### Task 1: Black Body Radiation

This task involves calculating the Stefan-Boltzmann constant by evaluating the integral in the black body radiation formula using different numerical integration methods.

### Task 2: Planetary Orbits

This task involves simulating a two-body problem where a planet orbits around a large star, using different numerical integration methods (explicit Euler and symplectic Euler) and comparing their accuracy.

### Task 3: Optimization Methods

This task involves implementing and comparing different optimization methods (Gradient Descent, Metropolis-Hastings algorithm, and Simulated Annealing) to find the global minimum of a noisy φ⁴ potential function.

## Author

[Your Name]

## Acknowledgments

- UCSB Physics Department
- Zihang Wang (TA) 