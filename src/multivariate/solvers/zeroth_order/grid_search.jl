function grid_search(f, grid)
    min_value = f(grid[1])
    arg_min_value = grid[1]

    for el in grid
        if f(el) < min_value
            arg_min_value = el
        end
    end

    return arg_min_value
end
