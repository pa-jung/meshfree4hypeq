# Optional but recommended for scripts: Ensure correct environment is active
using Pkg
Pkg.activate(".") # Activate Testing's environment

using IPlotPDESols

# Improved 1D Test
"""
Generates SimData1D for testing visualization with time-dependent functions.
"""
function doTestSimulation1D(params::ParamDictType)

    # --- Parameters ---
    dx = params["dx"]       # Spatial step size
    Lx = params["Lx"]       # Domain length
    T = params["T"]         # End time
    dt = params["dt"]       # Time step
    method = params["method"] # Simulation method name

    # --- Time Vector ---
    t = collect(0.0:dt:T)
    num_steps = length(t)

    # --- Spatial Grid (Static) ---
    x_grid = collect(0.0:dx:Lx)
    num_points = length(x_grid)

    # Replicate static grid over time
    x_over_time = [x_grid for _ in 1:num_steps]

    # --- Solution Vector Initialization ---
    u_over_time = Vector{Vector{Float64}}(undef, num_steps)

    # --- Statistics Initialization ---
    stats = ParamDictType()
    stats["max_abs_u"] = Vector{Float64}(undef, num_steps)
    stats["l1_norm"] = Vector{Float64}(undef, num_steps) # Integral |u| dx approx

    # --- Generate Solution and Stats Over Time ---
    println("Generating 1D data for method '$method'...")
    _,sim_time,_... = @timed for i = 1:num_steps
        current_t = t[i]
        u_snapshot = Vector{Float64}(undef, num_points)

        # --- Method Definitions ---
        if method == "decaying_sine"
            A = params["amplitude"]
            rate = params["decay_rate"]
            k = params["k"] # Wavenumber

            amplitude_t = A * exp(-rate * current_t)
            u_snapshot .= amplitude_t .* sin.(k .* x_grid)

            if !haskey(stats, "amplitude"); stats["amplitude"] = zeros(num_steps); end
            stats["amplitude"][i] = amplitude_t

        elseif method == "moving_gaussian"
            A = params["amplitude"]
            v = params["velocity"]
            w_sq = params["width"]^2
            x0 = Lx / 4 # Initial position

            center_x_t = mod(x0 + v * current_t, Lx) # Center moves and wraps
            u_snapshot .= A .* exp.(-((x_grid .- center_x_t).^2) ./ w_sq)

            if !haskey(stats, "peak_position"); stats["peak_position"] = zeros(num_steps); end
            stats["peak_position"][i] = center_x_t
            
        elseif method == "diffusing_gaussian"
             A = params["amplitude"]
             D = params["diffusion_coeff"]
             w0_sq = params["width"]^2 # Initial width squared
             center_x = Lx / 2
             
             # Width squared increases linearly with time
             current_w_sq = w0_sq + 2 * D * current_t
             # Amplitude decreases to conserve mass (propto 1/sqrt(width))
             amplitude_t = A * sqrt(w0_sq / current_w_sq) 
             
             u_snapshot .= amplitude_t .* exp.(-((x_grid .- center_x).^2) ./ current_w_sq)

             if !haskey(stats, "gaussian_width"); stats["gaussian_width"] = zeros(num_steps); end
             stats["gaussian_width"][i] = sqrt(current_w_sq)

        else
            u_snapshot .= 0.0
            @warn "Unknown method '$method'. Returning zero solution."
        end

        # Store snapshot
        u_over_time[i] = u_snapshot

        # Calculate basic stats for this time step
        stats["max_abs_u"][i] = isempty(u_snapshot) ? 0.0 : maximum(abs.(u_snapshot))
        # Approximate L1 norm: sum(|u_i| * dx)
        stats["l1_norm"][i] = sum(abs.(u_snapshot)) * dx

    end # End loop over time steps
    println("Finished generating 1D data for method '$method'.")

    # Create SimData1D object
    sim_data = SimData1D(x_over_time, u_over_time, t, params)
    sim_data.stats = stats # Add the calculated stats
    sim_data.stats["time"] = sim_time
    return sim_data
end

# --- Define Parameters and Configuration for 1D ---

shared_params_1d = ParamDict(
    "dx" => 0.02,  # Spatial step
    "Lx" => 10.0,  # Domain length
    "T"  => 4.0,   # End time
    "dt" => 0.05,  # Time step
    "amplitude" => 1.5 # Default amplitude
)

methods_dict_1d = MethodDict(
    "decaying_sine" => ParamDict(
        "decay_rate" => 0.5,
        "k" => 2.0 * pi / 5.0 # Wavelength approx 5
    ),
    "moving_gaussian" => ParamDict(
        "velocity" => 2.0,
        "width" => 0.5
    ),
    "diffusing_gaussian" => ParamDict(
        "diffusion_coeff" => 0.2,
        "width" => 0.5 # Initial width
    )
)

# Create SimulationConfig
sim_config_1d = SimulationConfig(
    doTestSimulation1D, # Use the new 1D function
    methods_dict_1d,
    shared_params_1d,
    "moving_gaussian" # Default method to show initially
)

# --- Run the 1D Visualization ---
# Assuming show1DSolutionFig is defined and functional
println("Starting 1D Visualization...")
show1DSolutionFig(sim_config_1d) 
show1DSolutionFig(sim_config_1d) 
showDynamicDependence(sim_config_1d)
println("Visualization launched (call commented out).")

showConvergencePlot(sim_config_1d, "dx", 10 .^ (collect(-1:.2:1)))
showConvergencePlot(sim_config_1d, "dx", 10 .^ (collect(-1:.2:1)), "l1_norm")
showConvergencePlot(sim_config_1d, "dx", 10 .^ (collect(-1:.2:1)), "time", "l1_norm")

#2D Testing
"""
Generates SimData2D for testing visualization. Creates time-dependent
2D functions on a static grid based on the specified method.
"""
function doTestSimulation2D(params::ParamDictType)

    # --- Parameters ---
    nx = params["nx"]       # Number of points in x direction
    ny = params["ny"]       # Number of points in y direction
    Lx = params["Lx"]       # Domain size in x
    Ly = params["Ly"]       # Domain size in y
    T = params["T"]         # End time
    dt = params["dt"]       # Time step
    method = params["method"] # Simulation method name

    # --- Time Vector ---
    t = collect(0.0:dt:T)
    num_steps = length(t)

    # --- Spatial Grid (Static Cartesian Grid) ---
    x_coords = range(0, Lx, length=nx)
    y_coords = range(0, Ly, length=ny)
    # Create a vector of (x,y) tuples for all grid points (column-major order)
    x_grid_tuples = [NTuple{2,Float64}((xi, yj)) for yj in y_coords for xi in x_coords] 
    num_points = length(x_grid_tuples)
    
    # Replicate the static grid for each time step
    x_over_time = [x_grid_tuples for _ in 1:num_steps]

    # --- Solution Vector Initialization ---
    u_over_time = Vector{Vector{Float64}}(undef, num_steps)

    # --- Statistics Initialization ---
    stats = ParamDictType() 
    stats["max_u"] = Vector{Float64}(undef, num_steps)
    stats["l2_norm_sq"] = Vector{Float64}(undef, num_steps) # L2 norm squared

    # --- Generate Solution and Stats Over Time ---
    println("Generating data for method '$method'...")
    _,sim_time,_... = @timed for i = 1:num_steps
        current_t = t[i]
        u_snapshot = Vector{Float64}(undef, num_points)
        
        # Get coordinates for broadcasting convenience
        coords_x = [pt[1] for pt in x_grid_tuples]
        coords_y = [pt[2] for pt in x_grid_tuples]

        # --- Method Definitions ---
        if method == "gaussian_decay"
            A = params["amplitude"]
            rate = params["decay_rate"]
            width_sq = (Lx / 8)^2
            center_x = Lx / 2
            center_y = Ly / 2
            
            amplitude_t = A * exp(-rate * current_t)
            u_snapshot .= amplitude_t .* exp.(-((coords_x .- center_x).^2 .+ (coords_y .- center_y).^2) ./ width_sq)
            
            # Add method-specific stat
            if !haskey(stats, "decaying_amplitude")
                 stats["decaying_amplitude"] = Vector{Float64}(undef, num_steps)
            end
            stats["decaying_amplitude"][i] = amplitude_t

        elseif method == "wave_packet"
            A = params["amplitude"]
            k = params["k"]         # Wave number magnitude
            omega = params["omega"]   # Frequency
            vx = params["vx"]       # Velocity in x
            width_sq = (Lx / 10)^2
            center_y = Ly / 2
            center_x_t = mod(Lx / 4 + vx * current_t, Lx) # Center moves and wraps around

            envelope = A .* exp.(-((coords_x .- center_x_t).^2 .+ (coords_y .- center_y).^2) ./ width_sq)
            wave = cos.(k .* (coords_x .- center_x_t) .- omega .* current_t) # Simple plane wave within envelope
            u_snapshot .= envelope .* wave

            # Add method-specific stat
             if !haskey(stats, "packet_center_x")
                 stats["packet_center_x"] = Vector{Float64}(undef, num_steps)
             end
            stats["packet_center_x"][i] = center_x_t


        elseif method == "standing_wave"
            A = params["amplitude"]
            kx_mode = params["kx_mode"] # Integer mode number
            ky_mode = params["ky_mode"] # Integer mode number
            amp_freq = params["amp_freq"] # Frequency of amplitude oscillation

            spatial_part = sin.(kx_mode * pi .* coords_x ./ Lx) .* sin.(ky_mode * pi .* coords_y ./ Ly)
            temporal_part = cos(amp_freq * current_t)
            u_snapshot .= A .* spatial_part .* temporal_part
            
            # Add method-specific stat
            if !haskey(stats, "amplitude_oscillation")
                 stats["amplitude_oscillation"] = Vector{Float64}(undef, num_steps)
            end
            stats["amplitude_oscillation"][i] = A * temporal_part

        else
            # Default case: flat zero solution
            u_snapshot .= 0.0
            @warn "Unknown method '$method'. Returning zero solution."
        end

        # Store snapshot
        u_over_time[i] = u_snapshot

        # Calculate basic stats for this time step
        stats["max_u"][i] = isempty(u_snapshot) ? 0.0 : maximum(abs.(u_snapshot))
        # Approximate L2 norm squared: sum(u_i^2 * dA) -> sum(u_i^2)*(Lx/nx)*(Ly/ny)
        dA = (Lx / (nx - 1)) * (Ly / (ny - 1)) # Area element approximation
        stats["l2_norm_sq"][i] = sum(u_snapshot.^2) * dA

        
    end # End loop over time steps
    println("Finished generating data for method '$method'.")

    # Create SimData2D object
    sim_data = SimData2D(x_over_time, u_over_time, t, params)
    sim_data.stats = stats # Add the calculated stats
    stats["time"] = sim_time
    return sim_data
end

# --- Define Parameters and Configuration ---

shared_params = ParamDict(
    "nx" => 30,    # Resolution x
    "ny" => 30,    # Resolution y
    "Lx" => 10.0,  # Domain size x
    "Ly" => 10.0,  # Domain size y
    "T"  => 5.0,   # End time
    "dt" => 0.1,   # Time step
    "amplitude" => 2.0 # Default amplitude
)

methods_dict = MethodDict(
    "gaussian_decay" => ParamDict(
        "decay_rate" => 0.8
    ),
    "wave_packet" => ParamDict(
        "k" => 2.0 * pi / 2.0, # Wavenumber (wavelength approx 2)
        "omega" => 2.0 * pi / 1.0, # Frequency (period approx 1)
        "vx" => 2.5            # Speed in x direction
    ),
    "standing_wave" => ParamDict(
        "kx_mode" => 2,      # Mode number in x
        "ky_mode" => 3,      # Mode number in y
        "amp_freq" => 2.0 * pi / 2.5 # Amplitude oscillation period approx 2.5
    )
)

# Create SimulationConfig
sim_config_2d = SimulationConfig(
    doTestSimulation2D, 
    methods_dict, 
    shared_params, 
    "gaussian_decay" # Default method to show initially
)

# --- Run the Visualization ---
println("Starting 2D Visualization...")
show2DSolutionFig(sim_config_2d)
println("Visualization launched.")
showConvergencePlot(sim_config_2d, "dt", .05:.05:.2, "l2_norm_sq")
showConvergencePlot(sim_config_2d, "dt", .05:.05:.2, "time", "max_u")