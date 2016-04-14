# TODO: Optimize for fg! calls
# TODO: Implement safeguards

function interpolating_linesearch!{T}(d::Union{DifferentiableFunction,
                                               TwiceDifferentiableFunction},
                                      x::Vector,
                                      p::Vector,
                                      x_new::Vector,
                                      gr_new::Vector,
                                      lsr::LineSearchResults{T},
                                      c::Real,
                                      mayterminate::Bool;
                                      c1::Real = 1e-4,
                                      c2::Real = 0.9,
                                      rho::Real = 2.0)

    # Parameter space
    n = length(x)

    # Count function calls
    f_calls, g_calls = 0, 0

    # Step-sizes
    a_0 = 0.0
    a_iminus1 = a_0
    a_i = 1.0
    a_max = 65536.0

    # phi(alpha) = f(x + alpha * p)
    phi_0 = d.f(x)
    f_calls += 1
    phi_a_iminus1 = phi_0
    phi_a_i = NaN

    # phi'(alpha) = vecdot(g(x + alpha * p), p)
    d.g!(x, gr_new)
    g_calls += 1
    phiprime_0 = vecdot(gr_new, p)
    phiprime_a_i = NaN

    # Iteration counter
    i = 1

    while a_i < a_max
        # Update x_new
        @simd for index in 1:n
            @inbounds x_new[index] = x[index] + a_i * p[index]
        end

        # Evaluate phi(a_i)
        phi_a_i = d.f(x_new)
        f_calls += 1

        # Test Wolfe conditions
        if (phi_a_i > phi_0 + c1 * a_i * phiprime_0) ||
             (phi_a_i >= phi_a_iminus1 && i > 1)
            a_star, f_up, g_up = zoom(a_iminus1, a_i,
                                      phiprime_0, phi_0,
                                      d.f, d.g!, x, p, x_new, gr_new)
            return a_star, f_calls + f_up, g_calls + g_up
        end

        # Evaluate phi'(a_i)
        d.g!(x_new, gr_new)
        g_calls += 1
        phiprime_a_i = vecdot(gr_new, p)

        # Check condition 2
        if abs(phiprime_a_i) <= -c2 * phiprime_0
            return a_i, f_calls, g_calls
        end

        # Check condition 3
        if phiprime_a_i >= 0.0
            a_star, f_up, g_up = zoom(a_i, a_iminus1,
                                      phiprime_0, phi_0,
                                      d.f, d.g!, x, p, x_new, gr_new)
            return a_star, f_calls + f_up, g_calls + g_up
        end

        # Choose a_iplus1 from the interval (a_i, a_max)
        a_iminus1 = a_i
        a_i *= rho

        # Update phi_a_iminus1
        phi_a_iminus1 = phi_a_i

        # Update iteration count
        i += 1
    end

    # Quasi-error response
    return a_max, f_calls, g_calls
end

function zoom(a_lo::Real,
              a_hi::Real,
              phiprime_0::Real,
              phi_0::Real,
              f::Function,
              g!::Function,
              x::Vector,
              p::Vector,
              x_new::Vector,
              gr_new::Vector;
              c1::Real = 1e-4,
              c2::Real = 0.9)

    # Parameter space
    n = length(x)

    # Count function and gradient calls
    f_calls, g_calls = 0, 0

    # Step-size
    a_j = NaN

    # Count iterations
    iteration = 0
    max_iterations = 10

    # Shrink bracket
    while iteration < max_iterations
        iteration += 1

        # Cache phi_a_lo
        @simd for index in 1:n
            @inbounds x_new[index] = x[index] + a_lo * p[index]
        end
        phi_a_lo = f(x_new)
        g!(x_new, gr_new)
        f_calls += 1
        g_calls += 1
        phiprime_a_lo = vecdot(gr_new, p)

        # Cache phi_a_hi
        @simd for index in 1:n
            @inbounds x_new[index] = x[index] + a_hi * p[index]
        end
        phi_a_hi = f(x_new)
        g!(x_new, gr_new)
        f_calls += 1
        g_calls += 1
        phiprime_a_hi = vecdot(gr_new, p)

        # Interpolate a_j
        if a_lo < a_hi
            a_j = interpolate(a_lo, a_hi,
                              phi_a_lo, phi_a_hi,
                              phiprime_a_lo, phiprime_a_hi)
        else
            # TODO: Check if this is needed
            a_j = interpolate(a_hi, a_lo,
                              phi_a_hi, phi_a_lo,
                              phiprime_a_hi, phiprime_a_lo)
        end

        # Update x_new
        @simd for index in 1:n
            @inbounds x_new[index] = x[index] + a_j * p[index]
        end

        # Evaluate phi(a_j)
        phi_a_j = f(x_new)
        f_calls += 1

        # Check Armijo
        if (phi_a_j > phi_0 + c1 * a_j * phiprime_0) ||
             (phi_a_j > phi_a_lo)
            a_hi = a_j
        else
            # Evaluate phiprime(a_j)
            g!(x_new, gr_new)
            g_calls += 1
            phiprime_a_j = vecdot(gr_new, p)

            if abs(phiprime_a_j) <= -c2 * phiprime_0
                return a_j, f_calls, g_calls
            end

            if phiprime_a_j * (a_hi - a_lo) >= 0.0
                a_hi = a_lo
            end

            a_lo = a_j
        end
    end

    # Quasi-error response
    return a_j, f_calls, g_calls
end

# a_lo = a_{i - 1}
# a_hi = a_{i}
function interpolate(a_i1::Real, a_i::Real,
                     phi_a_i1::Real, phi_a_i::Real,
                     phiprime_a_i1::Real, phiprime_a_i::Real)
    d1 = phiprime_a_i1 + phiprime_a_i -
           3.0 * (phi_a_i1 - phi_a_i) / (a_i1 - a_i)
    d2 = sqrt(d1 * d1 - phiprime_a_i1 * phiprime_a_i)
    return a_i - (a_i - a_i1) *
            ((phiprime_a_i + d2 - d1) /
             (phiprime_a_i - phiprime_a_i1 + 2.0 * d2))
end
