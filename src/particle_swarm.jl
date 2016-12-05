immutable ParticleSwarm{T} <: Optimizer
    lower::Vector{T}
    upper::Vector{T}
    n_particles::Int
end

ParticleSwarm(; lower = [], upper = [], n_particles = 0) = ParticleSwarm(lower, upper, n_particles)

type ParticleSwarmState{T}
    @add_generic_fields()
    iteration::Int64
    lower
    upper
    c1::T # Weight variable; currently not exposed to users
    c2::T # Weight variable; currently not exposed to users
    w::T  # Weight variable; currently not exposed to users
    limit_search_space::Bool
    n_particles::Int64
    X
    V
    X_best
    score
    best_score
    x_learn
    current_state
    iterations
end

initial_state(method::ParticleSwarm, options, d, initial_x::Array) = initial_state(method, options, d.f, initial_x)

function initial_state{T}(method::ParticleSwarm, options, f::Function, initial_x::Array{T})

    #=
    Variable X represents the whole swarm of solutions with
    the columns being the individual particles (= solutions to
    the optimization problem.)
    In each iteration the cost function is evaluated for all
    particles. For the next iteration all particles "move"
    towards their own historically best and the global historically
    best solution. The weighing coefficients c1 and c2 define how much
    towards the global or individual best solution they are pulled.

    In each iteration there is a check for an additional special
    solution which consists of the historically global best solution
    where one randomly chosen parameter is modified. This helps
    the swarm jumping out of local minima.
    =#
    n = length(initial_x)
    lower = copy(method.lower)
    upper = copy(method.upper)

    # do some checks on input parameters
    @assert length(lower) == length(upper) "lower and upper must be of same length."
    if length(lower) > 0
        limit_search_space = true
        @assert length(lower) == length(initial_x) "limits must be of same length as x_initial."
        @assert all(upper .> lower) "upper must be greater than lower"
    else
        limit_search_space = false
    end

    if method.n_particles > 0
        if method.n_particles < 3
          warn("Number of particles is set to 3 (minimum required)")
          n_particles = 3
        else
          n_particles = method.n_particles
        end
    else
      # user did not define number of particles
       n_particles = maximum([3, length(initial_x)])
    end
    c1 = 2.0
    c2 = 2.0
    w = 1.0

    X = Array{T}(size(initial_x)..., n_particles)
    V = similar(X)
    X_best = similar(X)
    dx = zeros(X)
    score = zeros(T, n_particles)
    x = zeros(initial_x)
    best_score = zeros(T, n_particles)
    x_learn = zeros(initial_x)

    f_calls = 0
    current_state = 0
    f_x = f(initial_x)
    f_calls += 1

    # if search space is limited, spread the initial population
    # uniformly over the whole search space
    if limit_search_space
        for i in 1:n_particles
            for j in 1:n
                ww = upper[j] - lower[j]
                X[j, i] = lower[j] + ww * rand()
                X_best[j, i] = X[j, i]
                V[j, i] = ww * (rand() * 2.0 - 1.0) / 10.0
            end
        end
    else
        for i in 1:n_particles
            for j in 1:n
                if i == 1
                    if abs(initial_x[i]) > 0.0
                        dx[j] = abs(initial_x[i])
                    else
                        dx[j] = 1.0
                    end
                end
                X[j, i] = initial_x[j] + dx[j] * rand()
                X_best[j, i] = X[j, i]
                V[j, i] = abs(X[j, i]) * (rand() * 2.0 - 1.0)
            end
        end
    end

    for j in 1:n
        X[j, 1] = initial_x[j]
        X_best[j, 1] = initial_x[j]
    end
    ParticleSwarmState("Particle Swarm",
        n,
        x,
        f_x,
        f_calls, # f call
        0, # g calls
        0, # h calls
        0,
        lower,
        upper,
        c1,
        c2,
        w,
        limit_search_space,
        n_particles,
        X,
        V,
        X_best,
        score,
        best_score,
        x_learn,
        0,
        options.iterations)
end

update_state!(d, state::ParticleSwarmState, method::ParticleSwarm) = update_state!(d.f, state, method)
function update_state!{T}(f::Function, state::ParticleSwarmState{T}, method::ParticleSwarm)
    if state.limit_search_space
        limit_X!(state.X, state.lower, state.upper, state.n_particles, state.n)
    end
    compute_cost!(f, state.n_particles, state.X, state.score)
    state.f_calls += state.n_particles

    if state.iteration == 0
        copy!(state.best_score, state.score)
        state.f_x = Base.minimum(state.score)
    end
    state.f_x = housekeeping!(state.score,
                              state.best_score,
                              state.X,
                              state.X_best,
                              state.x,
                              state.f_x,
                              state.n_particles)
    # Elitist Learning:
    # find a new solution named 'x_learn' which is the current best
    # solution with one randomly picked variable being modified.
    # Replace the current worst solution in X with x_learn
    # if x_learn presents the new best solution.
    # In all other cases discard x_learn.
    # This helps jumping out of local minima.
    worst_score, i_worst = findmax(state.score)
    for k in 1:state.n
        state.x_learn[k] = state.x[k]
    end
    random_index = rand(1:state.n)
    random_value = randn()
    sigma_learn = 1 - (1 - 0.1) * state.iteration / state.iterations

    r3 = randn() * sigma_learn

    if state.limit_search_space
        state.x_learn[random_index] = state.x_learn[random_index] + (state.upper[random_index] - state.lower[random_index]) / 3.0 * r3
    else
        state.x_learn[random_index] = state.x_learn[random_index] + state.x_learn[random_index] * r3
    end

    if state.limit_search_space
        if state.x_learn[random_index] < state.lower[random_index]
            state.x_learn[random_index] = state.lower[random_index]
        elseif state.x_learn[random_index] > state.upper[random_index]
            state.x_learn[random_index] = state.upper[random_index]
        end
    end

    score_learn = f(state.x_learn)
    state.f_calls += 1
    if score_learn < state.f_x
        state.f_x = score_learn * 1.0
        for j in 1:state.n
            state.X_best[j, i_worst] = state.x_learn[j]
            state.X[j, i_worst] = state.x_learn[j]
            state.x[j] = state.x_learn[j]
        end
        state.score[i_worst] = score_learn
        state.best_score[i_worst] = score_learn
    end

    # TODO find a better name for _f (look inthe paper, it might be called f there)
    state.current_state, _f = get_swarm_state(state.X, state.score, state.x, state.current_state)
    state.w, state.c1, state.c2 = update_swarm_params!(state.c1, state.c2, state.w, state.current_state, _f)
    update_swarm!(state.X, state.X_best, state.x, state.n, state.n_particles, state.V, state.w, state.c1, state.c2)
    state.iteration += 1
    false
end


function update_swarm!(X, X_best, best_point, n, n_particles, V,
                       w, c1, c2)
  # compute new positions for the swarm particles
  for i in 1:n_particles
      for j in 1:n
          r1 = rand()
          r2 = rand()
          vx = X_best[j, i] - X[j, i]
          vg = best_point[j] - X[j, i]
          V[j, i] = V[j, i]*w + c1*r1*vx + c2*r2*vg
          X[j, i] = X[j, i] + V[j, i]
      end
    end
end

function get_mu_1(f)
    if 0 <= f <= 0.4
        return 0.0
    elseif 0.4 < f <= 0.6
        return 5. * f - 2.0
    elseif 0.6 < f <= 0.7
        return 1.0
    elseif 0.7 < f <= 0.8
        return -10. * f + 8.0
    else
        return 0.0
    end
end

function get_mu_2(f)
    if 0 <= f <= 0.2
        return 0.0
    elseif 0.2 < f <= 0.3
        return 10. * f - 2.0
    elseif 0.3 < f <= 0.4
        return 1.0
    elseif 0.4 < f <= 0.6
        return -5. * f + 3.0
    else
        return 0.0
    end
end

function get_mu_3(f)
    if 0 <= f <= 0.1
        return 1.0
    elseif 0.1 < f <= 0.3
        return -5. * f + 1.5
    else
        return 0.0
    end
end

function get_mu_4(f)
    if 0 <= f <= 0.7
        return 0.0
    elseif 0.7 < f <= 0.9
        return 5. * f - 3.5
    else
        return 1.0
    end
end

function get_swarm_state(X, score, best_point, previous_state)
    # swarm can be in 4 different states, depending on which
    # the weighing factors c1 and c2 are adapted.
    # New state is not only depending on the current swarm state,
    # but also from the previous state.
    n, n_particles = size(X)
    f_best, i_best = findmin(score)
    d = zeros(Float64, n_particles)
    for i in 1:n_particles
        dd = 0.0
        for k in 1:n_particles
            for dim in 1:n
                @inbounds ddd = (X[dim, i] - X[dim, k])
                dd += ddd * ddd
            end
        end
        d[i] = sqrt(dd)
    end
    dg = d[i_best]
    dmin = Base.minimum(d)
    dmax = Base.maximum(d)

    f = (dg - dmin) / (dmax - dmin)
    mu = zeros(Float64, 4)
    mu[1] = get_mu_1(f)
    mu[2] = get_mu_2(f)
    mu[3] = get_mu_3(f)
    mu[4] = get_mu_4(f)
    best_mu, i_best_mu = findmax(mu)
    current_state = 0

    if previous_state == 0
        current_state = i_best_mu
    elseif previous_state == 1
        if mu[1] > 0
            current_state = 1
        else
          if mu[2] > 0
              current_state = 2
          elseif mu[4] > 0
              current_state = 4
          else
              current_state = 3
          end
        end
    elseif previous_state == 2
        if mu[2] > 0
            current_state = 2
        else
          if mu[3] > 0
              current_state = 3
          elseif mu[1] > 0
              current_state = 1
          else
              current_state = 4
          end
        end
    elseif previous_state == 3
        if mu[3] > 0
            current_state = 3
        else
          if mu[4] > 0
              current_state = 4
          elseif mu[2] > 0
              current_state = 2
          else
              current_state = 1
          end
        end
    elseif previous_state == 4
        if mu[4] > 0
            current_state = 4
        else
            if mu[1] > 0
                current_state = 1
            elseif mu[2] > 0
                current_state = 2
            else
                current_state = 3
            end
        end
    end
    return current_state, f
end

function update_swarm_params!(c1, c2, w, current_state, f)

    delta_c1 = 0.05 + rand() / 20.
    delta_c2 = 0.05 + rand() / 20.

    if current_state == 1
        c1 += delta_c1
        c2 -= delta_c2
    elseif current_state == 2
        c1 += delta_c1 / 2.
        c2 -= delta_c2 / 2.
    elseif current_state == 3
        c1 += delta_c1 / 2.
        c2 += delta_c2 / 2.
    elseif current_state == 4
        c1 -= delta_c1
        c2 -= delta_c2
    end

    if c1 < 1.5
        c1 = 1.5
    elseif c1 > 2.5
        c1 = 2.5
    end

    if c2 < 1.5
        c2 = 2.5
    elseif c2 > 2.5
        c2 = 2.5
    end

    if c1 + c2 > 4.0
        c_total = c1 + c2
        c1 = c1 / c_total * 4
        c2 = c2 / c_total * 4
    end

    w = 1 / (1 + 1.5 * exp(-2.6 * f))
    return w, c1, c2
end

function housekeeping!(score, best_score, X, X_best, best_point,
                       f_x, n_particles)
    n = size(X, 1)
    for i in 1:n_particles
        if score[i] <= best_score[i]
            best_score[i] = score[i]
            for k in 1:n
                X_best[k, i] = X[k, i]
            end
            if score[i] <= f_x
                for k in 1:n
                  	best_point[k] = X[k, i]
                end
              	f_x = score[i]
            end
        end
    end
    return f_x
end

function limit_X!(X, lower, upper, n_particles, n)
    # limit X values to boundaries
    for i in 1:n_particles
        for j in 1:n
            if X[j, i] < lower[j]
              	X[j, i] = lower[j]
            elseif X[j, i] > upper[j]
              	X[j, i] = upper[j]
            end
        end
    end
    nothing
end

function compute_cost!(f::Function,
                       n_particles::Int,
                       X::Matrix,
                       score::Vector)

    for i in 1:n_particles
        score[i] = f(X[:, i])
    end
    nothing
end
