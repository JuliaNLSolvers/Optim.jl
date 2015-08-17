log_temperature(t::Real) = 1 / log(t)

constant_temperature(t::Real) = 1.0

function default_neighbor!(x::Array, x_proposal::Array)
    @assert size(x) == size(x_proposal)
    for i in 1:length(x)
        @inbounds x_proposal[i] = x[i] + randn()
    end
    return
end

macro satrace()
    quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(x)
            end
            grnorm = NaN
            update!(tr,
                    iteration,
                    f_x,
                    grnorm,
                    dt,
                    store_trace,
                    show_trace,
                    show_every,
                    callback)
        end
    end
end

function simulated_annealing{T}(cost::Function,
                                initial_x::Array{T};
                                neighbor!::Function = default_neighbor!,
                                temperature::Function = log_temperature,
                                keep_best::Bool = true,
                                iterations::Integer = 100_000,
                                store_trace::Bool = false,
                                show_trace::Bool = false,
                                callback = nothing,
                                show_every = 1,
                                extended_trace::Bool = false)
    # Maintain current and proposed state
    x, x_proposal = copy(initial_x), copy(initial_x)

    # Record the number of iterations we perform
    iteration = 0

    # Track calls to function and gradient
    f_calls = 0

    # Count number of parameters
    n = length(x)

    # Store f(x) in f_x
    f_x = cost(x)
    f_calls += 1

    # Store the best state ever visited
    best_x = copy(x)
    best_f_x = f_x

    # Trace the history of states visited
    tr = OptimizationTrace()
    tracing = store_trace || show_trace || extended_trace || callback != nothing
    @satrace

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

        @satrace
    end

    return MultivariateOptimizationResults("Simulated Annealing",
                                           initial_x,
                                           best_x,
                                           @compat(Float64(best_f_x)),
                                           iterations,
                                           iteration == iterations,
                                           false,
                                           NaN,
                                           false,
                                           NaN,
                                           false,
                                           NaN,
                                           tr,
                                           f_calls,
                                           0)
end
