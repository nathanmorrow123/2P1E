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
    return  isempty(roots) ? nothing : min(roots)

end

display(surface(-1:1:100,-1:1:100,quartic))
