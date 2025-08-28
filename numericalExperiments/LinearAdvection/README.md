# Numerical Experiments for the Linear Advection Equation

This directory contains a suite of numerical experiments designed to test, validate, and compare the various meshfree numerical schemes implemented in this project. The focus is on the one-dimensional linear advection equation.

### The Test Problem: Linear Advection

The linear advection equation is the simplest hyperbolic conservation law, given by:
$$
u_t + a u_x = 0
$$
where $u(x,t)$ is the conserved quantity and $a$ is a constant wave speed. The exact solution is simply the initial condition, $u_0(x)$, advected with speed $a$, i.e., $u(x,t) = u_0(x - at)$.

Despite its simplicity, this equation is a fundamental benchmark for numerical methods. It allows for the clear and isolated study of key numerical properties such as:
-   **Numerical Diffusion:** The tendency of a scheme to smear out sharp features.
-   **Numerical Dispersion:** The tendency of a scheme to produce non-physical oscillations, especially near sharp gradients.
-   **Order of Accuracy:** Verifying that a scheme's error decreases at the theoretically predicted rate as the grid is refined.
-   **Stability:** Assessing the robustness of a scheme, particularly on non-uniform grids.

By testing our methods on this equation with both smooth (e.g., Gaussian) and non-smooth (e.g., box, Riemann problem) initial conditions, we can rigorously validate their performance before applying them to more complex nonlinear systems like the Burgers' or Euler equations.

### Global Simulation Settings

While each experiment has specific parameters, the following settings are shared across most simulations in this directory to ensure a consistent baseline:

* **`CFL`**: `0.2`. The Courant-Friedrichs-Lewy number is kept low to ensure that for high-order time integrators, the spatial error dominates the temporal error, which is crucial for convergence studies.
* **`SEED`**: `10`. The master seed for the random number generator, used to ensure that experiments on "random" or "irregular" grids are reproducible.
* **`interp_alpha`**: `1.0`. The default shape parameter for the exponential weight function used in the meshfree MLS reconstruction.
* **`interp_range`**: `3.5`. The default neighbor search radius, defined as a multiple of the nominal grid spacing $\Delta x$.
* **`bc` (Boundary Conditions)**: The boundary conditions are chosen based on the initial condition.
    * For localized, periodic profiles like the `gauss` and `box` functions, `:periodic` boundary conditions are used.
    * For the `riemann` problem (a step function), `:outflow` boundary conditions are used to prevent the wave from wrapping around the domain.

### Experiment Descriptions

This directory contains the following sub-folders, each corresponding to a specific numerical experiment:

* **`Box`**: Contains stability and accuracy tests for the box initial condition on both uniform and non-uniform grids. This is designed to assess the robustness of the schemes when grid quality is degraded.
* **`IMEXvsDirect`**: Compares the performance of direct explicit schemes (like `RK2`) against relaxation-based IMEX schemes (like `ARS233`). This is particularly important for analyzing the "double diffusion" effect of applying limiters within a relaxation framework.
* **`Older Plots`**: An archive of previous plots and results.
* **`ShockConvergence`**: Contains convergence studies for a discontinuous initial condition (a Riemann problem or "shock"). This tests if the schemes can maintain accuracy while stabilizing a sharp front.
* **`SmoothConvergence`**: Contains convergence studies for a smooth initial condition (a Gaussian pulse). This is a classical test to verify that the implemented schemes achieve their theoretical order of accuracy.
* **`Solutions`**: Contains various solution plots from different simulations. These are used for visual inspection and qualitative comparison of how well different methods resolve waves and discontinuities.
