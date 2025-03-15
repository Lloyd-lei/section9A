# Task 3: Optimization Methods

This task involves implementing and comparing different optimization methods: Gradient Descent, Metropolis-Hastings algorithm, and Simulated Annealing.

## Problem Description

We consider a noisy φ⁴ theory in 1D, given by the potential function:
H(θ) = θ⁴ - 8θ² - 2cos(4πθ)

Where θ is an order parameter. The goal is to find the ground state order parameter and energy.

## Tasks

### Part A: Gradient Descent

Implement the gradient descent method to locate the global minimum starting with three initial guesses: θ₀ = -1, 0.5, 3.

The update rule for gradient descent is:
θᵢ₊₁ = θᵢ - αᵢ·∇H(θᵢ)

Where αᵢ is the learning rate, which should be tuned at each step.

### Part B: Metropolis-Hastings Algorithm

Implement the Metropolis-Hastings algorithm to estimate the minimum of the noisy φ⁴ potential with the same initial guesses.

The algorithm works as follows:
1. Start with an initial parameter guess θ₀
2. Randomly move from θ₁ → θ₀ + Δθ, where Δθ ~ N(0, σ)
3. Calculate the ratio r = e^(-βΔH(θ*, θ))
4. If r > 1, accept the move; if r < 1, accept with probability r

Try different values of β (inverse temperature).

### Part C: Simulated Annealing

Add a cooling schedule to the Metropolis-Hastings algorithm:
βᵢ₊₁ = βᵢ + δᵢ

Estimate the minimum of the noisy φ⁴ potential with the same initial guesses, trying different cooling schedules.

Make a graphical comparison of the convergence steps with cooling and without cooling.

## Running the Code

To run the optimization methods, execute the following command in the terminal:

```bash
python optimization_methods.py
```

## Expected Output

The code will:
1. Visualize the potential function
2. Run all three optimization methods with different initial guesses
3. Create videos showing the gradient descent process
4. Generate comparison plots of the different methods
5. Print a summary of the optimization results

## Dependencies

- NumPy
- Matplotlib
- OpenCV (for video creation)
- tqdm (for progress bars) 