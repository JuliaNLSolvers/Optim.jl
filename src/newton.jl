immutable Newton <: Optimizer
    linesearch!::Function
end

Newton(; linesearch!::Function = hz_linesearch!) =
  Newton(linesearch!)

  type NewtonState{T}
      @add_generic_fields()
      x_previous::Array{T}
      g::Array{T}
      f_x_previous::T
      s::Array{T}
      x_ls::Array{T}
      g_ls::Array{T}
      H
      F
      Hd
      alpha::T
      mayterminate::Bool
      lsr
  end

function initialize_state{T}(method::Newton, options, d, initial_x::Array{T})
        n = length(initial_x)
      # Maintain current gradient in gr
      g = Array(T, n)
      s = Array(T, n)
      x_ls, g_ls = Array(T, n), Array(T, n)
      f_x_previous, f_x = NaN, d.fg!(initial_x, g)
      f_calls, g_calls = 1, 1
      H = Array(T, n, n)
      d.h!(initial_x, H)
      h_calls = 1

      NewtonState("Newton's Method",
                  length(initial_x),
                  copy(initial_x), # Maintain current state in state.x
                  f_x, # Store current f in state.f_x
                  f_calls, # Track f calls in state.f_calls
                  g_calls, # Track g calls in state.g_calls
                  h_calls,
                  0., # Elapsed time
                  copy(initial_x), # Maintain current state in state.x_previous
                  g, # Store current gradient in state.g
                  T(NaN), # Store previous f in state.f_x_previous
                  similar(initial_x), # Maintain current search direction in state.s
                  similar(initial_x), # Buffer of x for line search in state.x_ls
                  similar(initial_x), # Buffer of g for line search in state.g_ls
                  H,
                  copy(H),
                  copy(H),
                  alphainit(one(T), initial_x, g, f_x), # Keep track of step size in state.alpha
                  false, # state.mayterminate
                  LineSearchResults(T)) # Maintain a cache for line search results in state.lsr
  end

  function update!{T}(d, state::NewtonState{T}, method::Newton)

        # Search direction is always the negative gradient divided by
        # a matrix encoding the absolute values of the curvatures
        # represented by H. It deviates from the usual "add a scaled
        # identity matrix" version of the modified Newton method. More
        # information can be found in the discussion at issue #153.
        state.F, state.Hd = ldltfact!(Positive, state.H)
        state.s[:] = -(state.F\state.g)

        # Refresh the line search cache
        dphi0 = vecdot(state.g, state.s)
        clear!(state.lsr)
        push!(state.lsr, zero(T), state.f_x, dphi0)

        # Determine the distance of movement along the search line
        state.alpha, f_update, g_update =
          method.linesearch!(d, state.x, state.s, state.x_ls, state.g_ls, state.lsr, state.alpha, state.mayterminate)
        state.f_calls, state.g_calls = state.f_calls + f_update, state.g_calls + g_update

        # Maintain a record of previous position
        copy!(state.x_previous, state.x)

        # Update current position # x = x + alpha * s
        LinAlg.axpy!(state.alpha, state.s, state.x)

        # Update the function value and gradient
        state.f_x_previous, state.f_x = state.f_x, d.fg!(state.x, state.g)
        state.f_calls, state.g_calls = state.f_calls + 1, state.g_calls + 1

        # Update the Hessian
        d.h!(state.x, state.H)
        false
end
