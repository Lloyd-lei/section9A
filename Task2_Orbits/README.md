# Task 2: Planetary Orbits

This task involves simulating a two-body problem where a planet orbits around a large star, using different numerical integration methods.

## Problem Description

We consider a two-dimensional elliptical orbit of a planet around a star. The position of the planet is given by coordinates q = (q₁, q₂), with the planet's velocity given by p = q̇.

Newton's laws, with suitable normalization, yield the following ordinary differential equations:
- q̈₁ = -q₁/(q₁² + q₂²)^(3/2)
- q̈₂ = -q₂/(q₁² + q₂²)^(3/2)

This is equivalent to a Hamiltonian system with the Hamiltonian:
H(p, q) = (1/2)(p₁² + p₂²) - 1/√(q₁² + q₂²)

The initial position and velocity of the planet are:
- q₁(0) = 1 - e
- q₂(0) = 0
- q̇₁(0) = 0
- q̇₂(0) = √((1 + e)/(1 - e))

Where e is the eccentricity of the orbit.

## Tasks

### Part A
Using 100,000 steps, implement the explicit Euler method to simulate the orbit of the planet. Assume e = 0.6 and integrate to a final time of Tf = 200.

The explicit Euler update rules are:
- qₙ₊₁ = qₙ + Δt·q̇ₙ
- q̇ₙ₊₁ = pₙ₊₁ = pₙ + Δt·ṗₙ

### Part B
Using 400,000 steps, implement the symplectic Euler method to simulate the orbit. Compare the results with Part A by plotting both solutions in the same figure.

The symplectic Euler update rules are:
- pₙ₊₁ = pₙ - ΔtHq(pₙ₊₁, qₙ)
- qₙ₊₁ = qₙ + ΔtHp(pₙ₊₁, qₙ)

Where Hp and Hq denote the partial derivatives of the Hamiltonian with respect to p and q.

## Running the Code

To run the simulation, execute the following command in the terminal:

```bash
python orbits.py
```

## Expected Output

The code will:
1. Simulate the planetary orbit using both explicit and symplectic Euler methods
2. Generate a comparison plot showing both orbits
3. Create an animation of the orbits
4. Calculate and plot the energy conservation for both methods

## Dependencies

- NumPy
- Matplotlib
- OpenCV (for animation) 