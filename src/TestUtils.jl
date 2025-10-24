
"""
    check_for_nans(s::Any; range=nothing, counter=nothing)

Inspects all `AbstractVector{<:AbstractFloat}` fields within a given struct `s` for `NaN` values.

If NaNs are found, it throws a detailed error specifying which fields contain NaNs,
how many there are, and a list of the indices where they were found.

# Arguments
- `s::Any`: The struct to inspect.
- `range::Union{UnitRange, Nothing}=nothing`: An optional range of indices to check within each vector. If `nothing`, the entire vector is checked.
- `counter::Union{Base.Threads.Atomic{Int}, Nothing}=nothing`: An optional atomic counter to track calls.

# Example
```julia
debug_counter = Threads.Atomic{Int}(0)
# ... inside a loop
check_for_nans(my_struct, counter=debug_counter)
```
"""
function check_for_nans(s::Any; range::Union{UnitRange, Nothing}=nothing, counter::Union{Base.Threads.Atomic{Int}, Nothing}=nothing)
    # --- NEW: Atomic Counter Logic ---
    # If a counter is provided, increment it and optionally print a debug message.
    if !isnothing(counter)
        Threads.atomic_add!(counter, 1)
        current_count = counter[]

        # Example: Print a message every 1000 calls.
        if mod(current_count, 1000) == 0
            println("Running NaN check #$(current_count)...")
        end
    end

    # A dictionary to store the field name and a list of indices where NaNs are found.
    nan_locations = Dict{Symbol, Vector{Int}}()

    # Iterate over all property names (fields) of the struct.
    for field_name in propertynames(s)
        field_value = getproperty(s, field_name)

        # We only care about vectors of floating-point numbers.
        if !(field_value isa AbstractVector{<:AbstractFloat})
            continue
        end

        # Determine the actual range to iterate over.
        check_range = isnothing(range) ? eachindex(field_value) : range
        
        # Safety check to prevent BoundsError if the provided range is too large.
        if last(check_range) > length(field_value)
            println("Warning: Skipping field `:$field_name` in check_for_nans because the provided range is out of bounds.")
            continue
        end

        # Iterate and check for NaNs.
        for i in check_range
            if isnan(field_value[i])
                if !haskey(nan_locations, field_name)
                    nan_locations[field_name] = Int[]
                end
                push!(nan_locations[field_name], i)
            end
        end
    end

    # If the dictionary is not empty, it means we found NaNs.
    if !isempty(nan_locations)
        error_message = "NaNs detected!\n"
        if !isnothing(counter)
            error_message *= "(Check count: $(counter[]))\n"
        end
        
        # Build a detailed error message.
        for (field, indices) in nan_locations
            count = length(indices)
            error_message *= "Field `:$field`: Found $count NaN(s) at indices:\n"
            
            max_indices_to_show = 20
            indices_str = join(indices[1:min(count, max_indices_to_show)], ", ")
            if count > max_indices_to_show
                indices_str *= ", ..."
            end
            error_message *= "  [$indices_str]\n"
        end
        
        error(error_message)
    end

    return nothing
end