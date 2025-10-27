using Test
using Meshfree4ScalarEq
using Meshfree4ScalarEq.Interpolations

# Assume your Interpolator struct and its functors are defined as previously discussed
# struct Interpolator{D, ORDER, SCHEME} ... end
# function (interp::Interpolator{...})(...) ... end
# function ensure_capacity!(interp::Interpolator, n::Int) ... end

"""
    test_interpolator(interp::Interpolator{D, ORDER, SCHEME}; tol=1e-10)

Tests the given Interpolator instance against known analytical solutions
for various function types based on its dimension (D), order (ORDER),
and scheme (SCHEME: 0 for fit, 1 for gradient).
"""
function test_interpolator(interp::Interpolator{D, ORDER, SCHEME}; tol=1e-10) where {D, ORDER, SCHEME}
    
    # --- Test Setup ---
    # Define some sample points (adjust as needed)
    n_pts = 10 # Number of neighbor points for the test
    
    # Simple weights (all ones)
    wVec = ones(Float64, n_pts) 
    
    # Pre-allocate dfVec (will be filled based on the test function)
    dfVec = Vector{Float64}(undef, n_pts) 
    
    # Ensure interpolator buffers are sized (only matters for ORDER=2)
    #ensure_capacity!(interp, n_pts) 

    println("Testing Interpolator{D=$D, ORDER=$ORDER, SCHEME=$SCHEME}")

    # --- Dispatch based on Dimension ---
    if D == 1
        # --- 1D Test Cases ---
        
        # Simple points on a line
        dxVec = collect(range(-1.0, 1.0, length=n_pts)) 
        
        @testset "1D ORDER=$ORDER, SCHEME=$SCHEME" begin
            if SCHEME == 0 # Function Fit Tests
                if ORDER == 0 # Mean Value (f(x) = constant)
                    f_const = 5.0
                    dfVec .= 0.0 # df = f(xi) - f(0) = 5.0 - 5.0 = 0
                    result = interp(wVec, dfVec, n_pts) # Should return mean df
                    @test result ≈ 0.0 atol=tol skip=(ORDER != 0)
                    
                elseif ORDER == 1 # Linear Fit (f(x) = ax + b)
                    a, b = 2.0, 3.0
                    dfVec .= a .* dxVec # df = f(xi) - f(0) = (a*xi + b) - b = a*xi
                    res0, res1 = interp(dxVec, wVec, dfVec, n_pts)
                    @test res0 ≈ b atol=tol skip=(ORDER != 1) # Intercept (fit value at x=0)
                    @test res1 ≈ a atol=tol skip=(ORDER != 1) # Slope

                elseif ORDER == 2 # Quadratic Fit (f(x) = cx^2/2 + ax + b)
                    a, b, c = 2.0, 3.0, 4.0
                    # df = f(xi) - f(0) = (c*xi^2/2 + a*xi + b) - b = c*xi^2/2 + a*xi
                    dfVec .= c .* dxVec.^2 ./ 2.0 .+ a .* dxVec 
                    res0, res1, res2 = interp(dxVec, wVec, dfVec, n_pts)
                    @test res0 ≈ b atol=tol skip=(ORDER != 2) # Intercept (fit value at x=0)
                    @test res1 ≈ a atol=tol skip=(ORDER != 2) # Coeff for x
                    @test res2 ≈ c atol=tol skip=(ORDER != 2) # Coeff for x^2/2
                end

            elseif SCHEME == 1 # Gradient Calculation Tests
                if ORDER == 1 # Gradient of Linear Function
                    a, b = 2.0, 3.0
                    # df = f(xi) - f(0) = a*xi 
                    dfVec .= a .* dxVec
                    result = interp(dxVec, wVec, dfVec, n_pts)
                    @test result ≈ a atol=tol skip=(ORDER != 1) # Should return the gradient 'a'

                elseif ORDER == 2 # Gradient of Quadratic Function (Should still give linear part)
                    a, b, c = 2.0, 3.0, 4.0
                    # df = c*xi^2/2 + a*xi
                    dfVec .= c .* dxVec.^2 ./ 2.0 .+ a .* dxVec 
                    res1, res2 = interp(dxVec, wVec, dfVec, n_pts) # Returns grads df/dx, d2f/dx2
                    @test res1 ≈ a atol=tol skip=(ORDER != 2) # Gradient at x=0
                    @test res2 ≈ c atol=tol skip=(ORDER != 2) # Second derivative
                end
            end
        end # end testset 1D

    elseif D == 2
        # --- 2D Test Cases ---
        
        # Simple points in a square grid (or random)
        dxVec = randn(n_pts) 
        dyVec = randn(n_pts)

         @testset "2D ORDER=$ORDER, SCHEME=$SCHEME" begin
            if SCHEME == 0 # Function Fit Tests (Not typically used, but can test)
                # Add fit tests if needed, similar to 1D
                @test true # Placeholder if no fit tests needed

            elseif SCHEME == 1 # Gradient Calculation Tests
                 if ORDER == 1 # Gradient of Linear Function (f = ax + by + d)
                    a, b, d = 2.0, 3.0, 5.0
                    # df = f(xi,yi) - f(0,0) = (a*xi + b*yi + d) - d = a*xi + b*yi
                    dfVec .= a .* dxVec .+ b .* dyVec 
                    res1, res2 = interp(dxVec, dyVec, wVec, dfVec, n_pts)
                    @test res1 ≈ a atol=tol skip=(ORDER != 1) # df/dx at origin
                    @test res2 ≈ b atol=tol skip=(ORDER != 1) # df/dy at origin

                elseif ORDER == 2 # Gradient of Quadratic Function 
                    # f = cxx*x^2/2 + cyy*y^2/2 + cxy*xy + cx*x + cy*y + c0
                    cx, cy, c0 = 2.0, 3.0, 5.0
                    cxx, cyy, cxy = 4.0, 6.0, 1.0
                    # df = f(xi,yi) - f(0,0) = ...
                    dfVec .= cxx .* dxVec.^2 ./ 2.0 .+ cyy .* dyVec.^2 ./ 2.0 .+ cxy .* dxVec .* dyVec .+
                             cx .* dxVec .+ cy .* dyVec
                    
                    res1, res2, res3, res4, res5 = interp(dxVec, dyVec, wVec, dfVec, n_pts)
                    @test res1 ≈ cx  atol=tol skip=(ORDER != 2) # df/dx
                    @test res2 ≈ cy  atol=tol skip=(ORDER != 2) # df/dy
                    @test res3 ≈ cxx atol=tol skip=(ORDER != 2) # d2f/dx2
                    @test res4 ≈ cyy atol=tol skip=(ORDER != 2) # d2f/dy2
                    @test res5 ≈ cxy atol=tol skip=(ORDER != 2) # d2f/dxdy
                 end
            end
         end # end testset 2D
    end # end if D
end

# --- Example Usage ---
# Create interpolator instances (ensure constructor matches definition)
interp1D_O1_S1 = Interpolator{1, 1, 1}(100) 
interp1D_O2_S0 = Interpolator{1, 2, 1}(100) 
interp2D_O1_S1 = Interpolator{2, 1, 1}(100) 
interp2D_O2_S1 = Interpolator{2, 2, 1}(100) 

# Run tests
println("\n--- Testing 1D, Order 1, Gradient ---")
test_interpolator(interp1D_O1_S1)

println("\n--- Testing 1D, Order 2, Fit ---")
test_interpolator(interp1D_O2_S0)

println("\n--- Testing 2D, Order 1, Gradient ---")
test_interpolator(interp2D_O1_S1)

println("\n--- Testing 2D, Order 2, Gradient ---")
test_interpolator(interp2D_O2_S1)

# Add tests for other combinations (e.g., Order 0, Scheme 0) if needed