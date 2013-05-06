log_temperature(t::Real) = 1.0 / log(t)

constant_temperature(t::Real) = 1.0

function default_neighbor!(x::Vector, x_proposal::Vector)
    for i in 1:length(x)
        x_proposal[i] = x[i] + randn()
    end
    return
end

function simulated_annealing_trace!(tr::OptimizationTrace,
                                    x::Vector,
                                    f_x::Real,
                                    iteration::Integer,
                                    store_trace::Bool,
                                    show_trace::Bool)
    os = OptimizationState(copy(x), f_x, iteration)
    if store_trace
        push!(tr, os)
    end
    if show_trace
        println(os)
    end
end

function simulated_annealing{T}(cost::Function,
                                initial_x::Vector{T};
                                neighbor!::Function = default_neighbor!,
                                temperature::Function = log_temperature,
                                keep_best::Bool = true,
                                iterations::Integer = 100_000,
                                store_trace::Bool = false,
                                show_trace::Bool = false)

    # Maintain current state in x
    x = copy(initial_x)
    x_proposal = copy(initial_x)

    # Record the number of iterations we perform
    iteration = 0

    # Track calls to function and gradient
    f_calls = 0

    # Count number of parameters
    n = length(x)

    # Store f(x) in f_x
    f_x = cost(x)
    f_calls += 1

    # Store the history of function values
    f_values = Array(T, iterations + 1)
    fill!(f_values, nan(T))
    f_values[iteration + 1] = f_x

    # Store the best state ever visited
    best_x = copy(x)
    best_f_x = f_x

    # Trace the history of states visited
    tr = OptimizationTrace()
    tracing = store_trace || show_trace
    if tracing
        simulated_annealing_trace!(tr, x, f_x,
                                   iteration, store_trace, show_trace)
    end

    # We always perform a fixed number of iterations
    while iteration < iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        # Determine the temperature for current iteration
        t = temperature(iteration)

        # Randomly generate a neighbor of our current state
        neighbor!(x, x_proposal)

        # Evaluate the cost function at the proposed state
        f_proposal = cost(x_proposal)
        f_calls += 1

        if f_proposal <= f_x
            # If proposal is superior, we always move to it
            copy!(x, x_proposal)
            f_x = f_proposal

            # If the new state is the best state yet, keep a record of it
            if f_proposal < best_f_x
                best_f_x = f_proposal
                copy!(best_x, x_proposal)
            end
        else
            # If proposal is inferior, we move to it with probability p
            p = exp(-(f_proposal - f_x) / t)
            if rand() <= p
                copy!(x, x_proposal)
                f_x = f_proposal
            end
        end

        # Update history of function values
        f_values[iteration + 1] = f_x

        # Show trace
        if tracing
            simulated_annealing_trace!(tr, x, f_x,
                                       iteration, store_trace, show_trace)
        end
    end

    # Return the best state ever visited
    return OptimizationResults("Simulated Annealing",
                               initial_x,
                               best_x,
                               best_f_x,
                               iterations,
                               iteration == iterations,
                               false,
                               0.0,
                               false,
                               0.0,
                               false,
                               0.0,
                               tr,
                               f_calls,
                               0,
                               f_values[1:(iteration + 1)])
end
