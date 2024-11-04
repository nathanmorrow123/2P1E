using PlotlyJS
using Polynomials

function compute_roots(x_P::Float64, x_E::Float64, y_E::Float64)
    
    # Define coefficients
    a4 = 1
    a3 = -4
    a2 = 2 * (1 - x_E^2 - 3 * y_E^2 + x_P^2)
    a1 = 4 * (x_E^2 - y_E^2 - x_P^2 + 1)
    a0 = (x_E^2 + y_E^2 - x_P^2 + 1)^2 + 4 * (x_P^2 - 1) * y_E^2


    p = Polynomial([a0,a1,a2,a3,a4])
    r = roots(p)
    
    # Filter for real roots that are non-negative, have an imaginary part of zero, and are close to zero when evaluated by the polynomial 
    real_positive_roots = [x for x in r if isreal(x) && real(x) >= 0.0 && imag(x) ==0.0] 
    real_positive_roots = convert(Array{Float64},real_positive_roots)
    
    # There must be four postive real roots to be a sufficient solution to the quartic
    if isempty(real_positive_roots)
        return nothing
    else
        if length(real_positive_roots) == 4
            return minimum(real_positive_roots)
        else
            return nothing
        end
    end

end


function mirrorData(x_vals,y_vals,z_vals)
    
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

    return x_vals,y_vals,z_vals
end

# Define the range for x_E and y_E
x_P = 1.2
x_E_range = range(0.0, stop=x_P, length=100)
y_E_range = range(0.0, stop=1.0, length=100)


# List to store all the roots for plotting
all_roots = []

# Iterate over values of x_E
for x_E in x_E_range
    # For each x_E, check for the largest y_E that yields a positive real root
    for y_E in y_E_range
        if sqrt((x_E - x_P)^2 + y_E^2) >= 1  # Remove points within the capture circle
            min_root = compute_roots(x_P,x_E, y_E)
            # Check if there is at least one positive root
            if !isnothing(min_root)
                push!(all_roots, (x_E, y_E, min_root))  # Store the roots for plotting
            end
        end
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

x_vals,y_vals,z_vals = mirrorData(x_vals,y_vals,z_vals)

# Plot the data points
scatter_points = scatter(
    x=x_vals,
    y=y_vals,
    z=z_vals,
    mode="markers",
    marker=attr(
        size=2.5,
        color=z_vals,                # set color to an array/list of desired values
        colorscale="Viridis",   # choose a colorscale
        opacity=0.8
    ),
    type="scatter3d"
)

# Define capture circles for display
θ = range(0, stop=2π, length=100)
circle_x = x_P .+ cos.(θ)
circle_y = sin.(θ)
circle_z = zeros(length(θ))

# Plot the capture circles
capture_circle1 = scatter(
    x=circle_x,
    y=circle_y,
    z=circle_z,
    mode="lines",
    line=attr(
        color="red",
        width=2
    ),
    type="scatter3d"
)

circle_x = -x_P .+ cos.(θ)

capture_circle2 = scatter(
    x=circle_x,
    y=circle_y,
    z=circle_z,
    mode="lines",
    line=attr(
        color="red",
        width=2
    ),
    type="scatter3d"
)


plt = plot([scatter_points, capture_circle1, capture_circle2]) 
display(plt) # Display the plot
readline()
savefig(plt, "Results/surface_with_capture_circle.pdf")
