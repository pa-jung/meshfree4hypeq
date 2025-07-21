# Meshfree4ScalarEq

This package is for testing meshfree numerical methods for 1D and 2D scalar hyperbolic equations.

Currently, the code implements:
- 1D and 2D periodic grids
- Linear advection equation and Burgers' equation
- among other algorithms, a meshless WENO method from [Tiwari et al.][1], a meshless MUSCL method from [Willems et al.][2] and a 2D positive scheme from [Praveen et al.][3]
- four Runge-Kutta time integration routines
- several implementations of the Multidimensional Optimal Order Detection method (MOOD), see [Clain et al.][4], [Diot et al.][5], [Diot et al.][6] and [Willems et al.][2]

This code is built in a modular way. Grid points, grids, equations, time integration methods and numerical methods are defined as structs allowing easy implementation and testing (see test/). Basic unit tests are provided for testing interpolation and grid management routines. The numerical methods are tested using simple simulations, convergence tests, plots of the spectra of the semi-discretized PDEs, ... (see numericalExperiments). All numerical tests have the same structure. There is one Julia (.jl) file that executes the simulations. The data of the simulation is stored in /data. The simulation results are then analyzed in a Jupyter notebook. For example, to run the 1D convergence plot execute 
```
  julia --project numericalExperiments/convergence1D/convergence.jl.
```
Sometimes simluations can be performed in parallel, e.g., 
```
  julia --project -p 20 numericalExperiments/convergence2D/convergence.jl.
```

Below is a quick overview of each of the numerical tests in numericalExperiments.
- algorithmEfficiency*: Plot the error vs the computational time for several algorithms.
- convergence*: Plot the error vs the amount of grid points for several algorithms.
- gradientTest: Check the approximation error of several spatial discretizations.
- L2Stability*: Check the spectra of the semi-discretized PDEs for unstable eigenvalues for one or more grids. In addition, simulations are performed to check if stabilities indeed occur.
- linearAdvectionTest: Do a simple simulation of the linear advection equation to test for visual testing and getting to know the code.
- massLoss: Plot the mass as a function of time to check how unconservative the schemes are.
- MOODComparision: Compare the solution for several MOOD methods.
- nonlinearTest: First attemp at applying the meshless schemes to Burgers' equation.
- shockPositionTest: Numerically check if meshless schemes can capture the correct shock speed for Burgers' equation.  

[1]: https://www.sciencedirect.com/science/article/pii/S0021999122001504
[2]: https://arxiv.org/abs/2504.05942
[3]: https://www.researchgate.net/profile/Praveen-Chandrashekar-3/publication/277759856_A_positive_meshless_method_for_hyperbolic_equations/links/5630d66b08ae0530378cdee7/A-positive-meshless-method-for-hyperbolic-equations.pdf
[4]: https://www.sciencedirect.com/science/article/pii/S002199911100115X?via%3Dihub
[5]: https://www.sciencedirect.com/science/article/pii/S0045793012001909?via%3Dihub
[6]: https://onlinelibrary.wiley.com/doi/10.1002/fld.3804
