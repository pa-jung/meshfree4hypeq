# Meshfree4ScalarEq

This package provides a modular framework for testing and analyzing meshfree numerical methods for 1D and 2D scalar and systems of hyperbolic equations.

### Core Capabilities

The code implements a wide range of modern numerical techniques:
- **Discretization:** 1D and 2D particle grids with support for uniform, randomized, and fixed (ghost particle) configurations.
- **Equations:**
    - 1D Scalar Laws: Linear Advection and the inviscid Burgers' equation.
    - 1D Systems: The Euler equations of gas dynamics, solved via a relaxation approach.
- **Spatial Schemes:**
    - A meshless MUSCL method with user-selectable reconstruction order (up to 5th order consistency).
    - A meshless WENO method based on the work of [Tiwari et al.][1].
    - A 2D positive meshless scheme from [Praveen et al.][3].
- **Time Integration:**
    - Standard explicit Runge-Kutta schemes (e.g., RK2, RK4).
    - Implicit-Explicit (IMEX) Runge-Kutta schemes for stiff relaxation systems.
- **Stabilization:**
    - Several implementations of the Multidimensional Optimal Order Detection (MOOD) method, see [Clain et al.][4], [Diot et al.][5], [Diot et al.][6].
    - A modular `MUSCLlimited` scheme that can use modern slope limiters like Barth-Jespersen and Venkatakrishnan.
- **Boundary Conditions:** Support for both periodic and fixed (Dirichlet/outflow) boundary conditions.

### Project Structure and Workflow

This code is built in a modular way. Particle grids, equations, time integration methods, and spatial interpolators are defined as structs, allowing for easy implementation and testing of new combinations.

The primary workflow for running experiments has been updated to use the interactive plotting package `IPlotPDESols.jl`.

1.  **Configuration:** Each numerical experiment is defined in a single Julia script within the `numericalExperiments/` directory (e.g., `burgersMakie.jl`, `euler1D_relax_Makie.jl`). These scripts contain `SimulationConfig` objects that define all parameters for a set of simulations.
2.  **Execution:** Simulations are typically run from within an active Julia REPL by executing the desired experiment file. For example:
    ```julia
    # In the Julia REPL, after activating the project environment
    include("numericalExperiments/nonlinearTest/burgersMakie.jl")
    ```
3.  **Analysis:** Running the script will launch an interactive Makie plot window. This window, powered by `IPlotPDESols.jl`, allows for dynamic selection of methods, adjustment of parameters, on-the-fly calculation of convergence data, and saving of plots and simulation results. This replaces the previous workflow of running a command-line script and analyzing the data separately in a Jupyter notebook.

### Overview of Numerical Experiments

Below is a quick overview of each of the numerical tests in the `numericalExperiments/` directory.
- **algorithmEfficiency**: Plots the error vs. the computational time for several algorithms.
- **convergence**: Plots the error vs. the number of grid points for several algorithms to determine their order of accuracy.
- **gradientTest**: Checks the approximation error of several spatial discretizations.
- **L2Stability**: Checks the spectra of the semi-discretized PDEs for unstable eigenvalues.
- **linearAdvectionTest**: Contains various simple simulations of the linear advection equation for visual testing and demonstration of scheme properties (diffusion, dispersion, stability).
- **massLoss**: Plots the mass as a function of time to check how non-conservative the schemes are, particularly when MOOD is active.
- **MOODComparision**: Compares the solution for several MOOD methods on challenging problems.
- **nonlinearTest**: Contains experiments for the inviscid Burgers' equation.
- **shockPositionTest**: Numerically checks if meshless schemes can capture the correct shock speed for Burgers' equation.
- **euler1D**: Contains experiments for the 1D Euler equations using the relaxation scheme.

[1]: https://www.sciencedirect.com/science/article/pii/S0021999122001504
[2]: https://arxiv.org/abs/2504.05942
[3]: https://www.researchgate.net/profile/Praveen-Chandrashekar-3/publication/277759856_A_positive_meshless_method_for_hyperbolic_equations/links/5630d66b08ae0530378cdee7/A-positive-meshless-method-for-hyperbolic-equations.pdf
[4]: https://www.sciencedirect.com/science/article/pii/S002199911100115X?via%3Dihub
[5]: https://www.sciencedirect.com/science/article/pii/S0045793012001909?via%3Dihub
[6]: https://onlinelibrary.wiley.com/doi/10.1002/fld.3804