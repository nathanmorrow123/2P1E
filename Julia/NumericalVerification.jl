using Serialization
using LinearAlgebra
using Plots

# Load the surface data from the binary file
surface_data = open("Results/surface_data.bin", "r") do file
    deserialize(file)
end

# Extract the arrays from the loaded data
x_vals, y_vals, z_vals = surface_data

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
function propagate_and_store(x_E, y_E, x_P, chi, phi, t, μ)
    dt = 0.1
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
            return evader_trajectory, pursuer1_trajectory, pursuer2_trajectory
        end
    end

    return evader_trajectory, pursuer1_trajectory, pursuer2_trajectory
end

# Initialize plot
plt = plot(legend = false)

# Verify intercept trajectories for all evader positions and plot trajectories
for (x_E, y_E, t) in zip(x_vals, y_vals, z_vals)
    angle1, angle2 = initial_headings(x_E, y_E, t, x_P, μ)
    evader_trajectory, pursuer1_trajectory, pursuer2_trajectory = propagate_and_store(x_E, y_E, x_P, angle1, angle2, t, μ)

    # Plot trajectories 
    plot!(plt, [p[1] for p in evader_trajectory], [p[2] for p in evader_trajectory], arrow = true, color=:red, linewidth=1) 
    plot!(plt, [p[1] for p in pursuer1_trajectory], [p[2] for p in pursuer1_trajectory], arrow = false, color=:blue, linewidth=0.1) 
    plot!(plt, [p[1] for p in pursuer2_trajectory], [p[2] for p in pursuer2_trajectory], arrow = false, color=:blue, linewidth=0.1)
    
    """    
    # Add arrows at the end of the trajectories using quiver 
    evader_end = evader_trajectory[end] 
    pursuer1_end = pursuer1_trajectory[end] 
    pursuer2_end = pursuer2_trajectory[end] 
    # Calculate direction vectors for arrows 
    evader_direction = (evader_end[1] - evader_trajectory[end-1][1], evader_end[2] - evader_trajectory[end-1][2]) 
    pursuer1_direction = (pursuer1_end[1] - pursuer1_trajectory[end-1][1], pursuer1_end[2] - pursuer1_trajectory[end-1][2]) 
    pursuer2_direction = (pursuer2_end[1] - pursuer2_trajectory[end-1][1], pursuer2_end[2] - pursuer2_trajectory[end-1][2]) 
    quiver!(plt, [evader_end[1]], [evader_end[2]], quiver=([evader_direction[1]], [evader_direction[2]]), color=:red,linewidth=0.1) 
    quiver!(plt, [pursuer1_end[1]], [pursuer1_end[2]], quiver=([pursuer1_direction[1]], [pursuer1_direction[2]]), color=:blue,linewidth=0.1) 
    quiver!(plt, [pursuer2_end[1]], [pursuer2_end[2]], quiver=([pursuer2_direction[1]], [pursuer2_direction[2]]), color=:blue,linewidth=0.1)
    """

end

# Show plot with all trajectories
display(plt)

