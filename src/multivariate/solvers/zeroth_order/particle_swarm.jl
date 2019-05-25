struct ParticleSwarm{T} <: ZerothOrderOptimizer
    lower::Vector{T}
    upper::Vector{T}
    n_particles::Int
end

"""
# Particle Swarm
## Constructor
```julia
ParticleSwarm(; lower = [],
                upper = [],
                n_particles = 0)
```

The constructor takes 3 keywords:
* `lower = []`, a vector of lower bounds, unbounded below if empty or `Inf`'s
* `upper = []`, a vector of upper bounds, unbounded above if empty or `Inf`'s
* `n_particles = 0`, the number of particles in the swarm, defaults to least three

## Description
The Particle Swarm implementation in Optim.jl is the so-called Adaptive Particle
Swarm algorithm in [1]. It attempts to improve global coverage and convergence by
switching between four evolutionary states: exploration, exploitation, convergence,
and jumping out. In the jumping out state it intentially tries to take the best
particle and move it away from its (potentially and probably) local optimum, to
improve the ability to find a global optimum. Of course, this comes a the cost
of slower convergence, but hopefully converges to the global optimum as a result.

## References
- [1] Zhan, Zhang, and Chung. Adaptive particle swarm optimization, IEEE Transactions on Systems, Man, and Cybernetics, Part B: CyberneticsVolume 39, Issue 6 (2009): 1362-1381
"""
ParticleSwarm(; lower = [], upper = [], n_particles = 0) = ParticleSwarm(lower, upper, n_particles)

Base.summary(::ParticleSwarm) = "Particle Swarm"

mutable struct ParticleSwarmState{Tx,T} <: ZerothOrderState
    x::Tx
    iteration::Int
    lower::Tx
    upper::Tx
    c1::T # Weight variable; currently not exposed to users
    c2::T # Weight variable; currently not exposed to users
    w::T  # Weight variable; currently not exposed to users
    limit_search_space::Bool
    n_particles::Int
    X
    V
    X_best
    score::Vector{T}
    best_score::Vector{T}
    x_learn
    current_state
    iterations::Int
end

function initial_state(method::ParticleSwarm, options, d, initial_x::AbstractArray{T}) where T
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
    # TODO do we even need a lower((n) upper    ) that is different from method.lower(upper)
    # do some checks on input parameters
    @assert length(method.lower) == length(method.upper) "lower and upper must be of same length."
    if length(method.lower) > 0
        lower = copyto!(similar(initial_x), copy(method.lower))
        upper = copyto!(similar(initial_x), copy(method.upper))
        limit_search_space = true
        @assert length(lower) == length(initial_x) "limits must be of same length as x_initial."
        @assert all(upper .> lower) "upper must be greater than lower"
    else
        lower = copy(initial_x)
        upper = copy(initial_x)
        limit_search_space = false
    end

    if method.n_particles > 0
        if method.n_particles < 3
          @warn("Number of particles is set to 3 (minimum required)")
          n_particles = 3
        else
          n_particles = method.n_particles
        end
    else
      # user did not define number of particles
       n_particles = maximum([3, length(initial_x)])
    end
    c1 = T(2)
    c2 = T(2)
    w = T(1)

    X = Array{T,2}(undef, n, n_particles)
    V = Array{T,2}(undef, n, n_particles)
    X_best = Array{T,2}(undef, n, n_particles)
    dx = zeros(T, n)
    score = zeros(T, n_particles)
    x = copy(initial_x)
    best_score = zeros(T, n_particles)
    x_learn = copy(initial_x)

    current_state = 0

    value!!(d, initial_x)

    # if search space is limited, spread the initial population
    # uniformly over the whole search space
    if limit_search_space
        for i in 1:n_particles
            for j in 1:n
                ww = upper[j] - lower[j]
                X[j, i] = lower[j] + ww * rand(T)
                X_best[j, i] = X[j, i]
                V[j, i] = ww * (rand(T) * T(2) - T(1)) / 10
            end
        end
    else
        for i in 1:n_particles
            for j in 1:n
                if i == 1
                    if abs(initial_x[i]) > T(0)
                        dx[j] = abs(initial_x[i])
                    else
                        dx[j] = T(1)
                    end
                end
                X[j, i] = initial_x[j] + dx[j] * rand(T)
                X_best[j, i] = X[j, i]
                V[j, i] = abs(X[j, i]) * (rand(T) * T(2) - T(1))
            end
        end
    end

    for j in 1:n
        X[j, 1] = initial_x[j]
        X_best[j, 1] = initial_x[j]
    end
    ParticleSwarmState(
        x,
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

function update_state!(f, state::ParticleSwarmState{T}, method::ParticleSwarm) where T
    n = length(state.x)
    if state.limit_search_space
        limit_X!(state.X, state.lower, state.upper, state.n_particles, n)
    end
    compute_cost!(f, state.n_particles, state.X, state.score)

    if state.iteration == 0
        copyto!(state.best_score, state.score)
        f.F = Base.minimum(state.score)
    end
    f.F = housekeeping!(state.score,
                              state.best_score,
                              state.X,
                              state.X_best,
                              state.x,
                              value(f),
                              state.n_particles)
    # Elitist Learning:
    # find a new solution named 'x_learn' which is the current best
    # solution with one randomly picked variable being modified.
    # Replace the current worst solution in X with x_learn
    # if x_learn presents the new best solution.
    # In all other cases discard x_learn.
    # This helps jumping out of local minima.
    worst_score, i_worst = findmax(state.score)
    for k in 1:n
        state.x_learn[k] = state.x[k]
    end
    random_index = rand(1:n)
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

    score_learn = value(f, state.x_learn)
    if score_learn < f.F
        f.F = score_learn * 1.0
        for j in 1:n
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
    update_swarm!(state.X, state.X_best, state.x, n, state.n_particles, state.V, state.w, state.c1, state.c2)
    state.iteration += 1
    false
end


function update_swarm!(X::AbstractArray{Tx}, X_best, best_point, n, n_particles, V,
                       w, c1, c2) where Tx
  # compute new positions for the swarm particles
  for i in 1:n_particles
      for j in 1:n
          r1 = rand(Tx)
          r2 = rand(Tx)
          vx = X_best[j, i] - X[j, i]
          vg = best_point[j] - X[j, i]
          V[j, i] = V[j, i]*w + c1*r1*vx + c2*r2*vg
          X[j, i] = X[j, i] + V[j, i]
      end
    end
end

function get_mu_1(f::Tx) where Tx
    if Tx(0) <= f <= Tx(4)/10
        return Tx(0)
    elseif Tx(4)/10 < f <= Tx(6)/10
        return Tx(5) * f - Tx(2)
    elseif Tx(6)/10 < f <= Tx(7)/10
        return Tx(1)
    elseif Tx(7)/10 < f <= Tx(8)/10
        return -Tx(10) * f + Tx(8)
    else
        return Tx(0)
    end
end

function get_mu_2(f::Tx) where Tx
    if Tx(0) <= f <= Tx(2)/10
        return Tx(0)
    elseif Tx(2)/10 < f <= Tx(3)/10
        return Tx(10) * f - Tx(2)
    elseif Tx(3)/10 < f <= Tx(4)/10
        return Tx(1)
    elseif Tx(4)/10 < f <= Tx(6)/10
        return -Tx(5) * f + Tx(3)
    else
        return Tx(0)
    end
end

function get_mu_3(f::Tx) where Tx
    if Tx(0) <= f <= Tx(1)/10
        return Tx(1)
    elseif Tx(1)/10 < f <= Tx(3)/10
        return -Tx(5) * f + Tx(3)/2
    else
        return Tx(0)
    end
end

function get_mu_4(f::Tx) where Tx
    if Tx(0) <= f <= Tx(7)/10
        return Tx(0)
    elseif Tx(7)/10 < f <= Tx(9)/10
        return Tx(5) * f - Tx(7)/2
    else
        return Tx(1)
    end
end

function get_swarm_state(X::AbstractArray{Tx}, score, best_point, previous_state) where Tx
    # swarm can be in 4 different states, depending on which
    # the weighing factors c1 and c2 are adapted.
    # New state is not only depending on the current swarm state,
    # but also from the previous state.
    n, n_particles = size(X)
    f_best, i_best = findmin(score)
    d = zeros(Tx, n_particles)
    for i in 1:n_particles
        dd = Tx(0.0)
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

    f = (dg - dmin) / max(dmax - dmin, sqrt(eps(Tx)))

    mu = zeros(Tx, 4)
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

function update_swarm_params!(c1, c2, w, current_state, f::T) where T

    delta_c1 = T(5)/100 + rand(T) / T(20)
    delta_c2 = T(5)/100 + rand(T) / T(20)

    if current_state == 1
        c1 += delta_c1
        c2 -= delta_c2
    elseif current_state == 2
        c1 += delta_c1 / 2
        c2 -= delta_c2 / 2
    elseif current_state == 3
        c1 += delta_c1 / 2
        c2 += delta_c2 / 2
    elseif current_state == 4
        c1 -= delta_c1
        c2 -= delta_c2
    end

    if c1 < T(3)/2
        c1 = T(3)/2
    elseif c1 > T(5)/2
        c1 = T(5)/2
    end

    if c2 < T(3)/2
        c2 = T(5)/2
    elseif c2 > T(5)/2
        c2 = T(5)/2
    end

    if c1 + c2 > T(4)
        c_total = c1 + c2
        c1 = c1 / c_total * 4
        c2 = c2 / c_total * 4
    end

    w = 1 / (1 + T(3)/2 * exp(-T(26)/10 * f))
    return w, c1, c2
end

function housekeeping!(score, best_score, X, X_best, best_point,
                       F, n_particles)
    n = size(X, 1)
    for i in 1:n_particles
        if score[i] <= best_score[i]
            best_score[i] = score[i]
            for k in 1:n
                X_best[k, i] = X[k, i]
            end
            if score[i] <= F
                for k in 1:n
                  	best_point[k] = X[k, i]
                end
              	F = score[i]
            end
        end
    end
    return F
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

function compute_cost!(f,
                       n_particles::Int,
                       X::Matrix,
                       score::Vector)

    for i in 1:n_particles
        score[i] = value(f, X[:, i])
    end
    nothing
end
