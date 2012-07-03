##########################################################################
###
### Fletcher-Powell
###
##########################################################################

function fletcher_powell(x::Vector)
  function theta(x::Vector)
    if x[1] > 0
      atan(x[2] / x[1]) / (2.0 * pi)
    else
      (pi + atan(x[2] / x[1])) / (2.0 * pi)
    end
  end
  
  100.0 * (x[3] - 10.0 * theta(x[1], x[2]))^2 + (sqrt(x[1]^2 + x[2]^2) - 1.0)^2 + x[3]^2
end

function fletcher_powell_gradient(x::Vector)
  error("Not yet defined")
end

function fletcher_powell_hessian(x::Vector)
  error("Not yet defined")
end

##########################################################################
###
### Himmelbrau's Function
###
##########################################################################

function himmelbrau(x::Vector)
  (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
end

function himmelbrau_gradient(x::Vector)
  [4.0 * x[1]^3 + 4.0 * x[1] * x[2] + 44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0,
   2.0 * x[1]^2 + 2.0 * x[2] - 22.0 + 4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]]
end

function himmelbrau_hessian(x::Vector)
end

##########################################################################
###
### Parabola
###
##########################################################################

function parabola(x::Vector)
  (1.0 - x[1])^2 + 
  (2.0 - x[2])^2 +
  (3.0 - x[3])^2 +
  (5.0 - x[4])^2 +
  (8.0 - x[5])^2
end

function parabola_gradient(x::Vector)
  [-2.0 * (1.0 - x[1]),
   -2.0 * (2.0 - x[2]),
   -2.0 * (3.0 - x[3]),
   -2.0 * (5.0 - x[4]),
   -2.0 * (8.0 - x[5])]
end

function parabola_hessian(x::Vector)
  2.0 * eye(5)  
end

##########################################################################
###
### Powell
###
##########################################################################

function powell(x::Vector)
  (x[1] + 10.0 * x[2])^2 +
  5.0 * (x[3] - x[4])^2 + 
  (x[2] - 2.0 * x[3])^4 +
  10.0 * (x[1] - x[4])^4
end

function powell_gradient(x::Vector)
  [2.0 * (x[1] + 10.0 * x[2]) + 40.0 * (x[1] - x[4])^3,
   20.0 * (x[1] + 10.0 * x[2]) + 4.0 * (x[2] - 2.0 * x[3])^3,
   10.0 * (x[3] - x[4]) - 8.0 * (x[2] - 2.0 * x[3])^3,
   -10.0 * (x[3] - x[4]) - 40.0 * (x[1] - x[4])^3]
end

function powell_hessian(x::Vector)
  h = zeros(4, 4)
  # EDIT h!
  h
end

##########################################################################
###
### Rosenbrock
###
##########################################################################

function rosenbrock(x::Vector)
  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function rosenbrock_gradient(x::Vector)
  [2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1],
   200.0 * (x[2] - x[1]^2)]
end

function rosenbrock_hessian(x::Vector)
  h = zeros(2, 2)
  h[1, 1] = -2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
  h[1, 2] = -400.0 * x[1]
  h[2, 1] = -400.0 * x[1]
  h[2, 2] = 200
  h
end
