using PlotlyJS
using Polynomials
using LaTeXStrings

function compute_roots(x_P::Float64, x_E::Float64, y_E::Float64)
    
    # Define coefficients
    a4 = 1
    a3 = -4
    a2 = 2 * (1 - x_E^2 - 3 * y_E^2 + x_P^2)
    a1 = 4 * (x_E^2 - y_E^2 - x_P^2 + 1)
    a0 = (x_E^2 + y_E^2 - x_P^2 + 1)^2 + 4 * (x_P^2 - 1) * y_E^2


    p = Polynomial([a0,a1,a2,a3,a4])
    r = roots(p)
    
    # Filter for real roots that are non-negative, have an imaginary part of zero
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
x_P = 1.1
x_E_range = range(0.0, stop=x_P, length=1000)
y_E_range = range(0.0, stop=1.0, length=1000)


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

# Define the height of the cylinders as the maximum of the roots array
height = maximum(z_vals)

# Define capture circles for display
θ = range(0, stop=2π, length=50)

# Circle 1
circle_x1 = x_P .+ cos.(θ)
circle_y1 = sin.(θ)
circle_z1_bottom = zeros(length(θ))
circle_z1_top = fill(height, length(θ))

# Circle 2
circle_x2 = -x_P .+ cos.(θ)
circle_y2 = sin.(θ)
circle_z2_bottom = zeros(length(θ))
circle_z2_top = fill(height, length(θ))

# Function to create mesh for cylinders
function create_cylinder_mesh(circle_x, circle_y, circle_z_bottom, circle_z_top)
    vertices_x = vcat(circle_x, circle_x)
    vertices_y = vcat(circle_y, circle_y)
    vertices_z = vcat(circle_z_bottom, circle_z_top)

    n = length(circle_x)
    
    # Create faces for the cylinder sides
    faces_i, faces_j, faces_k = [], [], []
    for i in 1:(n-1)
        push!(faces_i, i); push!(faces_j, i+n); push!(faces_k, i+n+1)
        push!(faces_i, i); push!(faces_j, i+n+1); push!(faces_k, i+1)
    end
    # Last segment
    push!(faces_i, n); push!(faces_j, 2n); push!(faces_k, n+1)
    push!(faces_i, n); push!(faces_j, n+1); push!(faces_k, 1)

    return (vertices_x, vertices_y, vertices_z, faces_i, faces_j, faces_k)
end

# Create the mesh for both cylinders
(vertices_x1, vertices_y1, vertices_z1, faces_i1, faces_j1, faces_k1) = create_cylinder_mesh(circle_x1, circle_y1, circle_z1_bottom, circle_z1_top)
(vertices_x2, vertices_y2, vertices_z2, faces_i2, faces_j2, faces_k2) = create_cylinder_mesh(circle_x2, circle_y2, circle_z2_bottom, circle_z2_top)

# Create the mesh plot
cylinder1_mesh = mesh3d(
    x=vertices_x1,
    y=vertices_y1,
    z=vertices_z1,
    i=faces_i1 .- 1,  # Adjust for 0-based indexing
    j=faces_j1 .- 1,
    k=faces_k1 .- 1,
    color="blue",
    opacity=0.1
)

cylinder2_mesh = mesh3d(
    x=vertices_x2,
    y=vertices_y2,
    z=vertices_z2,
    i=faces_i2 .- 1,
    j=faces_j2 .- 1,
    k=faces_k2 .- 1,
    color="blue",
    opacity=0.1
)

layout3 = Layout( 
            width=1920, height=1080,
			scene = attr(
                    xaxis_title="xE",
                    yaxis_title="yE",
                    zaxis_title="Time to Capture",	
                    title="Barrier Surface"),
            title_x =0.5,
            titlefont_size="18",
            scene_aspectratio=attr(x=2, y=1, z=0.5))

plt = plot([scatter_points, cylinder1_mesh, cylinder2_mesh],layout3) 

display(plt) # Display the plot
savefig(plt, "Results/surface_with_capture_circle.pdf")
