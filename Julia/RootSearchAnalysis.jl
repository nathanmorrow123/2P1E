using Roots
using Plots

function quartic(x_E::Float64, y_E::Float64)
    x_P = 1.1

    # Define coefficients
    a4 = 1
    a3 = -4
    a2 = 2 * (1 - x_E^2 - 3 * y_E^2 + x_P^2)
    a1 = 4 * (x_E^2 - y_E^2 - x_P^2 + 1)
    a0 = (x_E^2 + y_E^2 - x_P^2 + 1)^2 + 4 * (x_P^2 - 1) * y_E^2

    # Define the polynomial function
    poly_func(t) = a4 * t^4 + a3 * t^3 + a2 * t^2 + a1 * t + a0

    # Use Roots.jl to find real, positive roots in a specified range
    roots = find_zeros(poly_func, 0, 10)  # Adjust range as needed

    # Return roots if found; otherwise return nothing
    return  isempty(roots) ? nothing : minimum(roots)

end

function compute_roots(x_E::Float64, y_E::Float64)
    x_P = 1.1

    # Define coefficients
    a4 = 1
    a3 = -4
    a2 = 2 * (1 - x_E^2 - 3 * y_E^2 + x_P^2)
    a1 = 4 * (x_E^2 - y_E^2 - x_P^2 + 1)
    a0 = (x_E^2 + y_E^2 - x_P^2 + 1)^2 + 4 * (x_P^2 - 1) * y_E^2

    # Define the polynomial function
    poly_func(t) = a4 * t^4 + a3 * t^3 + a2 * t^2 + a1 * t + a0

    # Use Roots.jl to find real, positive roots in a specified range
    roots = find_zeros(poly_func, 0, 10)  # Adjust range as needed

    # Return roots if found; otherwise return nothing
    return isempty(roots) ? nothing : roots
end

# Define the range for x_E and y_E
x_E_range = range(0.0, stop=1.1, length=20)
y_E_range = range(0.0, stop=1.0, length=20)

# Dictionary to store the largest y_E for each x_E
largest_y_E_for_positive_root = Dict{Float64, Float64}()
# List to store all the roots for plotting
all_roots = []

# Iterate over values of x_E
for x_E in x_E_range
    max_y_E = nothing  # Placeholder to track the largest y_E with a positive real root

    # For each x_E, check for the largest y_E that yields a positive real root
    for y_E in y_E_range
        current_roots = compute_roots(x_E, y_E)
        # Check if there is at least one positive root
        if current_roots !== nothing
            max_y_E = y_E  # Update max_y_E to the current y_E
            min_root = minimum(current_roots)
            push!(all_roots, (x_E, y_E, min_root))  # Store the roots for plotting
        end
    end
    
    # Store the largest y_E found for this x_E in the dictionary
    if max_y_E !== nothing
        largest_y_E_for_positive_root[x_E] = max_y_E
    end
end


# Prepare data for 3D plotting
x_vals = Float64[]
y_vals = Float64[]
z_vals = Float64[]

# Collect the roots for plotting
for (x_E, y_E, roots) in all_roots
    for root in roots
        push!(x_vals, x_E)
        push!(y_vals, y_E)
        push!(z_vals, root)
    end
end

# Mirroring the data
mirror_x_vals = [-x for x in x_vals]
mirror_y_vals = [-y for y in y_vals]
x_vals_copy = copy(x_vals)
y_vals_copy = copy(y_vals)
z_vals_copy = copy(z_vals)

# Append mirrored data Quadrant II
x_vals = vcat(x_vals, mirror_x_vals)
y_vals = vcat(y_vals, y_vals_copy)
z_vals = vcat(z_vals, z_vals_copy)  # Keep z_vals the same for mirrored y

# Append mirrored data Quadrant III
x_vals = vcat(x_vals, mirror_x_vals)
y_vals = vcat(y_vals, mirror_y_vals)
z_vals = vcat(z_vals, z_vals_copy)  # Keep z_vals the same for mirrored y

# Append mirrored data Quadrant IV
x_vals = vcat(x_vals, x_vals_copy)
y_vals = vcat(y_vals, mirror_y_vals)
z_vals = vcat(z_vals, z_vals_copy)  # Keep z_vals the same for mirrored y

display(scatter3d(x_vals,y_vals,z_vals))


