# Simulation Parameter Documentation

In the following, we describe the parameters used for the numerical experiments in more detail. The simulation framework is designed to solve one-dimensional hyperbolic conservation laws using meshfree generalized finite difference methods. A key feature of the framework is the decoupling of the time integration from the spatial discretization, which allows for a modular combination of different numerical components.

The code handles two main types of problems. First, it can solve scalar conservation laws of the form $u_t + F(u)_x = 0$. For the scalar experiments presented in this work, no source terms are considered. Second, the framework is extended to solve systems of conservation laws, specifically the 1D Euler equations. This is achieved through a relaxation scheme, which transforms the original nonlinear system, $\mathbf{U}_t + \mathbf{F}(\mathbf{U})_x = \mathbf{0}$, into a larger, semi-linear system of advection equations coupled by a stiff source term:
$$
\frac{\partial \mathbf{v}}{\partial t} + \mathbf{\Lambda} \frac{\partial \mathbf{v}}{\partial x} = \frac{1}{\epsilon}(\mathbf{M}(\mathbf{U}) - \mathbf{v})
$$
In this formulation, the complex nonlinear fluxes are replaced by simple linear advection with constant speeds (the diagonal matrix $\mathbf{\Lambda}$), and all physical coupling and nonlinearity are moved into the relaxation source term. The macroscopic state $\mathbf{U}$ is recovered by summing components of the kinetic state vector $\mathbf{v}$. This approach allows for the use of robust implicit-explicit (IMEX) time-stepping schemes to handle the stiff source term efficiently.

## Domain and Discretization Parameters

These parameters define the computational domain and the particle (or grid point) distribution.

* **`xmin`, `xmax`**: These floating-point values define the start and end of the one-dimensional spatial domain, $[x_{min}, x_{max}]$.
* **`N`**: An integer specifying the number of physical particles or cells within the domain. For simulations with ghost cells, the total number of particles in the grid will be larger than `N`.
* **`bc`**: A symbol specifying the boundary condition type. Common values are `:periodic` for periodic domains and `:fixed` or `:outflow` for domains with fixed boundaries where ghost cells are used.
* **`randomness_factor`**: A floating-point value that controls the regularity of the particle distribution.
    * If `randomness_factor = 0.0`, a perfectly uniform grid is generated with spacing $\Delta x = (x_{max} - x_{min}) / N$.
    * If `randomness_factor > 0.0`, each particle's position is perturbed by a random amount up to `randomness_factor * \Delta x`, creating a non-uniform grid. This is used to test the robustness of the meshfree methods.
* **`SEED`**: An integer used to seed the random number generator. This ensures that simulations with the same `randomness_factor` are perfectly reproducible.

## Time Integration Parameters

These parameters control the time-stepping process.

* **`tmax`**: A floating-point value for the final simulation time $t_{max}$. All simulations start at $t=0$.
* **`timestepper`**: A string that selects the time integration scheme. Examples include `'EulerUpwind'` (first-order explicit), `'RalstonRK2'` (a second-order explicit Runge-Kutta), `'RK4'` (classical fourth-order Runge-Kutta), and various IMEX (Implicit-Explicit) schemes like `'ARS233'` for relaxation methods.
* **`CFL`**: The Courant-Friedrichs-Lewy number, a floating-point value typically between 0 and 1. If provided, the time step $\Delta t$ is calculated dynamically at each step to satisfy the CFL stability condition for the explicit advection part of the scheme, i.e., $\Delta t = \text{CFL} \cdot \frac{\Delta x_{min}}{|\lambda|_{max}}$.
* **`dt`**: A fixed floating-point value for the time step $\Delta t$. This is used if the `CFL` parameter is not provided. Using a fixed `dt` is generally only recommended for unconditionally stable implicit methods or for specific analyses where the time step needs to be decoupled from the spatial resolution.

## Equation and Initial Condition

These parameters define the specific PDE and its initial state.

* **`PDE`**: A string specifying the hyperbolic equation to be solved, e.g., `'linear'` for the linear advection equation or `'burgers'` for the inviscid Burgers' equation.
* **`PDE_params`**: A tuple containing parameters for the chosen PDE. For linear advection, this would be the advection speed `a`. For Burger's equation, this is typically empty.
* **`init_func`**: A string specifying the initial condition, e.g., `'gauss'`, `'sine'`, `'box'`, or `'riemann'`.
* **`init_params`**: A tuple containing the parameters for the chosen initial condition function. For example, for a `'gauss'` profile, this would be `(amplitude, mean, width)`.

## Meshfree Interpolator Parameters

For the meshfree schemes (`MUSCL`, `UpwindGradient`), these parameters control the spatial discretization.

* **`main_gradient`**: A string selecting the primary gradient interpolation method used to approximate the spatial derivative (flux divergence). Common choices are `'MUSCL'` or `'Upwind'`.
* **`order`**: An integer that specifies the consistency order of the spatial reconstruction. For `'MUSCL'`, an `order` of 2 corresponds to a linear reconstruction ($1^{st}$ order MUSCL), `order` 3 to quadratic, and so on.
* **`interp_range`**: A floating-point value that defines the cutoff radius for the neighbor search. It is typically defined as a multiple of the nominal grid spacing $\Delta x$. All particles within this radius are included in the stencil for the MLS fit.
* **`interp_alpha`**: A floating-point parameter for the `exponentialWeightFunction`, which controls how quickly the influence of a neighbor decreases with distance.
* **`main_flux`**: A string selecting the numerical flux function (e.g., `'Rusanov'`, `'Upwind'`) used at the midpoints between particles to handle upwinding and provide stability.

## MOOD and Fallback Parameters

These parameters control the MOOD (Multi-dimensional Optimal Order Detection) scheme, which is used to add robustness to high-order methods.

* **`MOOD`**: A string specifying the MOOD criterion, e.g., `'U1'`, `'U2'`, or `'none'` to disable it. The criterion checks if the candidate solution from the high-order method is physically admissible (e.g., preserves positivity or avoids new extrema).
* **`fallback_gradient`**: The `GradientInterpolator` (e.g., `'Upwind'`) to use if the `MOOD` criterion detects a problematic cell. This is typically a more robust, lower-order method.
* **`fallback_flux`**: The `NumericalFluxFunction` used by the fallback gradient interpolator.
* **`limiter`**: For `MUSCLlimited` methods, this string (e.g., `'VK'` for Venkatakrishnan, `'BJ'` for Barth-Jespersen) selects the slope limiting strategy.
* **`delta_relax`**: A boolean that enables or disables the "delta relaxation" modification in some MOOD criteria, which can help prevent unnecessary switching to the fallback method.
* **`switch_tol`**: A tolerance used by smooth switching methods like `RalstonRK2SmoothSwitch`.

## Relaxation Scheme Parameters

For solving systems like the Euler equations, a relaxation scheme is used. These parameters are only active when `relax_method` is true.

* **`relax_method`**: A boolean that, when `true`, activates the relaxation scheme. The scalar advection schemes are then applied to a set of kinetic variables.
* **`relax_velocities`**: A tuple or vector defining the constant speeds for the kinetic advection equations. For a scalar problem, this might be `(1.0, -1.0)`. For a system, it's a list of speed pairs, one for each macroscopic variable.
* **`relax_epsilon`**: The relaxation parameter $\epsilon$. This controls the stiffness of the source term $S = (M(U) - V)/\epsilon$. A smaller $\epsilon$ drives the kinetic variables $V$ towards their macroscopic Maxwellian equilibrium $M(U)$ more quickly.