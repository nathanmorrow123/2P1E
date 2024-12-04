using PlotlyJS
using Polynomials
using Serialization
using Base.Threads: @threads


function compute_min_root(x_P::Float64, x_E::Float64, y_E::Float64)
    
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
            sort(real_positive_roots)
            println(real_positive_roots)
            return real_positive_roots[2] # According to Pacther, t_2 is the optimal solution for all players, t_1 is for the evader is forced to pass between the pursuers
        else
            return nothing
        end
    end
end


function mirrorData(x_vals,y_vals,z_vals)
    
    # Remove NaNs from the arrays
    valid_indices = .!isnan.(x_vals) .& .!isnan.(y_vals) .& .!isnan.(z_vals)
    x_vals = x_vals[valid_indices]
    y_vals = y_vals[valid_indices]
    z_vals = z_vals[valid_indices]

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

x_E_range = range(0.0, stop=x_P, length=100)
y_E_range = range(0.0, stop=1.0, length=100)

# Prepare data for 3D plotting 
x_vals = Float64[]
y_vals = Float64[] 
z_vals = Float64[]

# Function to compute the roots and collect valid points
@threads for x_E in reverse(x_E_range)
    for y_E in y_E_range
        if sqrt((x_E - x_P)^2 + y_E^2) >= 1  # Remove points within the capture circle
            min_root = compute_min_root(x_P, x_E, y_E)
            if !isnothing(min_root)
                push!(x_vals, x_E)
                push!(y_vals, y_E)
                push!(z_vals, min_root) # This case the z axis is our root axis (Time to capture)
            else 
                y_E_range = range(start = 0,stop=y_E,length = 100)
                break
            end
        end
    end
end



x_vals,y_vals,z_vals = mirrorData(x_vals,y_vals,z_vals)
println("Length of surface array: ",length(x_vals))
# Plot the data points
scatter_points = scatter(
    x=x_vals,
    y=y_vals,
    z=z_vals,
    mode="markers",
    marker=attr(
        size=2.5,
        color=z_vals,
    ),
    type="scatter3d"
)



# Find the maximum and minimum values along the z-axis
max_z = maximum(z_vals)
min_z = minimum(z_vals)

# Find the indices of the maximum and minimum values
max_index = argmax(z_vals)
min_index = argmin(z_vals)

println(max_z," ",min_z)
println(max_index," ",min_index)

# Add annotations for the maximum and minimum points
annotations = [
    attr(
        x=x_vals[max_index],
        y=y_vals[max_index],
        z=z_vals[max_index],
        text="Max",
        showarrow=true,
        arrowhead=2
    ),
    attr(
        x=x_vals[min_index],
        y=y_vals[min_index],
        z=z_vals[min_index],
        text="Min",
        showarrow=true,
        arrowhead=2
    )
]

# Define the height of the cylinders as the maximum of the roots array
z_Ceil = maximum(z_vals)
z_Floor = minimum(z_vals)

"# Define capture circles for display Half Circles
θ1 = range(start = pi/2, stop = 3*pi/2, length = 20)
θ2 = range(start = pi/2, stop = -pi/2, length= 20)
"
θ1 = range(start = 0, stop = 2*pi, length = 60)
θ2 = range(start = 0, stop = 2*pi, length = 60)
# Circle 1
circle_x1 = x_P .+ cos.(θ1)
circle_y1 = sin.(θ1)
circle_z1_bottom = z_Floor*ones(length(θ1))
circle_z1_top = fill(z_Ceil, length(θ1))

# Circle 2
circle_x2 = -x_P .+ cos.(θ2)
circle_y2 = sin.(θ2)
circle_z2_bottom = z_Floor*ones(length(θ2))
circle_z2_top = fill(z_Ceil, length(θ2))

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
    opacity=0.3
)

cylinder2_mesh = mesh3d(
    x=vertices_x2,
    y=vertices_y2,
    z=vertices_z2,
    i=faces_i2 .- 1,
    j=faces_j2 .- 1,
    k=faces_k2 .- 1,
    color="blue",
    opacity=0.3
)

layout3 = Layout( 
            template = templates.plotly_dark,  
            #width=1920, height=1080,
			scene = attr(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Time to Capture",
                    title="2P1E Barrier Surface, μ = √2",
                    xaxis_range = [-1,1],
                    yaxis_range = [-1,1],
                    zaxis_range = [z_Floor,z_Ceil+0.25]),
            title_x =0.5,
            titlefont_size="18",
            #dpi = 300 #For saving the figure 
            #annotations = annotations # For labeling all the minimum and maximum points (Not working)
            )


plt = Plot([scatter_points, cylinder1_mesh, cylinder2_mesh],layout3) 
savefig(plt, "Results/surface_with_capture_circle.pdf")
display(plt) # Display the plot


# Saving the surface data into a bin file

surface_data = (x_vals,y_vals,z_vals)

open("Results/surface_data.bin","w") do file
    serialize(file,surface_data)
end

