using Base

function grid_search(f::Function, grid::Vector)
  min_value = Inf
  arg_min_value = Inf
  
  for el in grid
    if f(el) < min_value
      arg_min_value = el
    end
  end
  
  arg_min_value
end
