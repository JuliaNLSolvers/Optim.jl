function f_lm(x)
  [x[1], 2.0 - x[2]]
end
function g_lm(x)
  [1.0 0.0; 0.0 -1.0]
end

initial_x = [100.0, 100.0]

results = Optim.levenberg_marquardt(f_lm, g_lm, initial_x)
@assert norm(results.minimum - [0.0, 2.0]) < 0.01