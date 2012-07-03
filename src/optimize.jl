# Wrap other functions.
# Automatically wrap inputs and convert vectors appropriately.
function optimize(f::Function,
                  g::Function,
                  h::Function,
                  initial_x::Vector,
                  method::String,
                  tolerance::Float64,
                  minimize::Bool)
  error("Not yet implemented")
end

function optimize(f::Function,
                  g::Function,
                  h::Function,
                  initial_x::Vector,
                  method::String)
  if method == "nelder-mead"
    nelder_mead(f, initial_x)
  elseif method == "sa"
    simulated_annealing(f, initial_x)
  elseif method == "gradient_descent"
    gradient_descent(f, g, initial_x)
  elseif method == "newton"
    newton(f, g, h, initial_x)
  elseif method == "bfgs"
    bfgs(f, g, initial_x)
  elseif method == "l-bfgs"
    l_bfgs(f, g, initial_x)
  end
end

function optimize(f::Function,
                  g::Function,
                  h::Function,
                  initial_x::Vector)
  newton(f, g, h, initial_x)
end

function optimize(f::Function,
                  g::Function,
                  initial_x::Vector)
  l_bfgs(f, g, initial_x)
end

function optimize(f::Function, initial_x::Vector)
  nelder_mead(f, initial_x)
end
