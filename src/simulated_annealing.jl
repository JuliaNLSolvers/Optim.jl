log_temperature(t::Real) = 1.0 / log(t)

constant_temperature(t::Real) = 1.0

function default_neighbor!(x::Vector, storage::Vector)
    for i in 1:length(x)
        storage[i] = x[i] + randn()
    end
end

function simulated_annealing(cost::Function,
                             s0::Vector;
                             neighbor!::Function = default_neighbor!,
                             temperature::Function = log_temperature,
                             keep_best::Bool = true,
                             tolerance::Real = 1e-8,
                             iterations::Integer = 100_000,
                             store_trace::Bool = false,
                             show_trace::Bool = false)

    # Maintain a trace of the optimization algo's state
    tr = OptimizationTrace()

    # Set our current state to the specified intial state
    s = copy(s0)
    s_n = copy(s0)
    y = cost(s)

    # Track how many function calls we've made
    f_calls = 1

    # Set the best state we've seen to the intial state
    best_s = copy(s0)
    best_y = y

    # Record the number of iterations we perform
    iteration = 0

    # Update our trace information
    os = OptimizationState(copy(s), y, iteration)
    if store_trace
        push!(tr, os)
    end
    if show_trace
        println(os)
    end

    # We always perform a fixed number of iterations
    while iteration < iterations
        # Update the iteration counter
        iteration += 1

        # Call temperature to find the proper temperature at time i
        t = temperature(iteration)

        # Call neighbor to randomly generate a neighbor of our current state
        neighbor!(s, s_n)

        # Evaluate the cost function at the proposed state
        y_n = cost(s_n)
        f_calls += 1

        if y_n <= y
            # If the proposed new state is superior, we always move to it
            copy!(s, s_n)
            y = y_n

            # If the new state is the best state we have seen,
            #  keep a record of it
            if y_n < best_y
                best_y = y_n
                copy!(best_s, s_n)
            end
        else
            # If the proposed new state is inferior, we move to it with
            #  probability p
            p = exp(-(y_n - y) / t)
            if rand() <= p
                copy!(s, s_n)
                y = y_n
            end
        end

        # Print out the state of the system
        if store_trace || show_trace
            os = OptimizationState(copy(s), y, iteration)
            if store_trace
                push!(tr, os)
            end
            if show_trace
                println(os)
            end
        end
    end

    # If specified by the user, we return the best state we've seen
    # Otherwise, we return the last state we've seen
    if keep_best
        OptimizationResults("Simulated Annealing",
                            s0,
                            best_s,
                            best_y,
                            iterations,
                            false,
                            tr,
                            f_calls,
                            0)
    else
        OptimizationResults("Simulated Annealing",
                            s0,
                            s,
                            y,
                            iterations,
                            false,
                            tr,
                            f_calls,
                            0)
    end
end
