using LinearAlgebra
using Plots
using Serialization
plotly()

# Load the surface data from the binary file
surface_data = open("Results/surface_data.bin", "r") do file
    deserialize(file)
end

# Extract the arrays from the loaded data
x_vals, y_vals, z_vals = surface_data

# Count the total number of lines in the surface data file
total_lines = length(x_vals)
println("Total number of lines in the surface data file: ", total_lines)

# Filter data to match the desired density
function filter_data(x_vals, y_vals, z_vals,desired_density)
    filtered_x = []
    filtered_y = []
    filtered_z = []

    unique_x_vals = unique(x_vals)
    num_x_indices = length(unique_x_vals)
    diff_x = maximum(unique_x_vals)-minimum(unique_x_vals)
    num_x_steps = Int(floor(num_x_indices/(desired_density*diff_x)))-1
    new_x_range = [unique_x_vals[i] for i in 1:num_x_steps:length(unique_x_vals)]

    for x in new_x_range
        # Find indices of current x_val
        indices = findall(i -> x_vals[i] == x, eachindex(x_vals))
        current_y_vals = y_vals[indices]

        # Calculate min and max of y_vals for the current x_val
        diff_y = maximum(current_y_vals) -minimum(current_y_vals)
        
        
        num_y_indices = length(current_y_vals)
        num_steps = Int(floor(num_y_indices/(desired_density*diff_y)))
        if(num_y_indices!=0&&num_steps!=0)
            selected_indices = [indices[1 + i * num_steps] for i in 0:(Int(floor(num_y_indices/num_steps))-1)]
            # Append the selected values to filtered arrays
            append!(filtered_x, x_vals[selected_indices])
            append!(filtered_y, y_vals[selected_indices])
            append!(filtered_z, z_vals[selected_indices])
        end
    end

    return filtered_x, filtered_y, filtered_z
end

# Apply the filter
desired_density = 25
filtered_x_vals, filtered_y_vals, filtered_z_vals = filter_data(x_vals, y_vals, z_vals,desired_density)
println("Total number of lines after filter: ", length(filtered_x_vals))

# Extract the arrays from the loaded data
x_vals, y_vals, z_vals = filtered_x_vals, filtered_y_vals, filtered_z_vals

# Define the pursuer position and speed ratio
x_P = 1.1  # Example value for x_P
μ = sqrt(2)  # Speed ratio

# Function to solve for the initial heading angles using the law of cosines
function initial_headings(x_E, y_E, t, x_P, μ)
    if y_E == 0
        y = sqrt(μ^2 * t^2 - x_E^2)
        y = sqrt((t + 1)^2 - x_P^2)
    else
        y = (x_E^2 + y_E^2 - x_P^2 + 1 + 2 * t - (μ^2 - 1) * t^2) / (2 * y_E)
    end

    println((x_E, y_E, t, x_P, μ))
    # Triangle one P1 - Origin - Intercept
    a = x_P  # (Origin to P1)
    b = t + 1  # (P1 to Intercept)
    c = abs(y)  # (Origin to Intercept)

    chi = acos((a^2 + b^2 - c^2) / (2 * a * b))
    chi = pi - chi
    if y < 0
        chi = -chi
    end

    # Triangle two Evader - EvaderYaxisProj - Intercept
    if x_E == 0
        if y > 0
            phi = π / 2
        else
            phi = -π / 2
        end
    else

        a_ev = x_E  # (Y axis to Evader)
        b_ev = μ * t  # (Evader to Intercept)
        c_ev = abs(y - y_E)  # (EvaderYaxisProj to Intercept)

        phi = acos((a_ev^2 + b_ev^2 - c_ev^2) / (2 * a_ev * b_ev))
        if y < 0 
            phi = -phi
        end

    end
    
    return chi, phi
end

# Function to propagate the players in time and store their trajectories
function propagate_capture_check(x_E, y_E, x_P, chi, phi, t, μ)
    dt = 0.0001
    capture_radius = 1.0

    # Initial positions
    x_ev, y_ev = x_E, y_E
    x_p1, y_p1 = x_P, 0.0
    x_p2, y_p2 = -x_P, 0.0

    # Store trajectories
    evader_trajectory = [(x_ev, y_ev)]
    pursuer1_trajectory = [(x_p1, y_p1)]
    pursuer2_trajectory = [(x_p2, y_p2)]

    # Propagate in time
    for _ in 1:Int(round(t/dt))
        x_ev += dt * cos(phi)  # Evader moves at speed 1
        y_ev += dt * sin(phi)

        x_p1 += dt * μ * cos(chi)  # Pursuer 1 moves at speed μ
        y_p1 += dt * μ * sin(chi)

        x_p2 += dt * μ * cos(pi - chi)  # Pursuer 2 mirrors pursuer 1
        y_p2 += dt * μ * sin(pi - chi)

        # Store positions
        push!(evader_trajectory, (x_ev, y_ev))
        push!(pursuer1_trajectory, (x_p1, y_p1))
        push!(pursuer2_trajectory, (x_p2, y_p2))

        # Check for capture
        if sqrt((x_p1 - x_ev)^2 + (y_p1 - y_ev)^2) <= capture_radius || sqrt((x_p2 - x_ev)^2 + (y_p2 - y_ev)^2) <= capture_radius
            println("Evader was captured")
            return evader_trajectory, pursuer1_trajectory, pursuer2_trajectory
        end
    end

    println("Evader was not captured")
    println("Final Distance Between Players: ")
    println((sqrt((x_p1 - x_ev)^2 + (y_p1 - y_ev)^2),sqrt((x_p2 - x_ev)^2 + (y_p2 - y_ev)^2)))
    return evader_trajectory, pursuer1_trajectory, pursuer2_trajectory
end

# Initialize plot
plt = plot(legend = false, size=(1920,1080),template = "plotly_dark")


# Verify intercept trajectories for all evader positions and plot trajectories
for (x_E, y_E, t) in zip(x_vals, y_vals, z_vals)
    chi, phi = initial_headings(x_E, y_E, t, x_P, μ) # Chi is Pursuer's heading and Phi is the evader's heading
    
    evader_trajectory, pursuer1_trajectory, pursuer2_trajectory = propagate_capture_check(x_E, y_E, x_P, chi, phi, t, μ)
    
    # Plot trajectories 
    scatter!(plt, (x_E,y_E), color=:maroon, alpha = 0.1)
    scatter!(plt, (-x_P,0), color=:darkBlue, alpha = 0.1)
    scatter!(plt, (x_P,0), color=:darkBlue, alpha = 0.1)
    #quiver([x_E],[y_E],[cos(phi)+x_E],[sin(phi)+y_E])
    plot!(plt, [p[1] for p in evader_trajectory], [p[2] for p in evader_trajectory], arrow = true, color=:red, linewidth=1,) 
    plot!(plt, [p[1] for p in pursuer1_trajectory], [p[2] for p in pursuer1_trajectory], arrow = false, color=:blue, linewidth=0.1) 
    plot!(plt, [p[1] for p in pursuer2_trajectory], [p[2] for p in pursuer2_trajectory], arrow = false, color=:blue, linewidth=0.1)
end


# Show plot with all trajectories
display(plt)
savefig(plt, "Results/OptimalFlowField.pdf")