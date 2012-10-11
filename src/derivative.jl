# Functions for calculating numerical derivates. By exploiting Taylor series expansions higher order approximations of derivatives are calculated.

# The function "derivative" is the workhorse function for calculation fo derivatives. The function f must be of the form Float64->Float64 and hence x0 must also be a Float64. The argument h is the step size. Step sizes around 0.0001 seem to be optimal. The estimate can be either single sided or double sided where there latter is preferred but not always feasible.
function derivative(f::Function, x0::Float64, h::Float64, twoside::Bool)
	if twoside
		d = 4^5*(2^3*(f(x0+h) - f(x0-h)) - (f(x0+2h) - f(x0-2h))) - (2^3*(f(x0+4h) - f(x0-4h)) - (f(x0+8h) - f(x0-8h)))
		d /= (4^5*(2^4 - 2^2) - (2^6 - 2^4))*h
	else
		f0 	= f(x0)
		d 	= 2^3*(2^2*(f(x0+h) - f0) - (f(x0+2h) - f0)) - (2^2*(f(x0+2h) - f0) - (f(x0+4h) - f0))
		d 	/= 3*2^2*h
	end
	return d
end
derivative(f::Function, x0::Float64, h::Float64) = derivative(f, x0, h, true)
derivative(f::Function, x0::Float64) = derivative(f, x0, 0.0001)
derivative(f::Function) = x -> derivative(f, x)

# The function "dirderivative" calculates directional derivatives in the direction v. The function supplied must have the form Array{Float64, 1} -> Float64
dirderivative(f::Function, v::Array{Float64, 1}, x0::Array{Float64, 1}, h::Float64, twoside::Bool) = derivative(t::Float64 -> f(x0 + v*t) / norm(v), 0.0, h, twoside)
dirderivative(f::Function, v::Array{Float64, 1}, x0::Array{Float64, 1}, h::Float64) = dirderivative(f, v, x0, h, true)
dirderivative(f::Function, v::Array{Float64, 1}, x0::Array{Float64, 1}, ) = derivative(f, v, x0, 0.0001)
dirderivative(f::Function, v::Array{Float64, 1}) = x -> dirderivative(f, v, x)

# Function for calculation of second order derivatives. This seem not more precise than applying derivative twice but it is more efficient. Eventually single sided option but will wait to see how often problems arise.
function derivative2(f::Function, x0::Float64, h::Float64)
	f0 = f(x0)
	d = 4^6*(2^4*(f(x0+h) + f(x0-h) - 2f0) - (f(x0+2h) + f(x0-2h) - 2f0)) - (2^4*(f(x0+4h) + f(x0-4h) - 2f0) - (f(x0+8h) + f(x0-8h) - 2f0))
	return d / (3*2^6*(2^8-1)*h^2)
end 

# Function for calculation of a gradient. The function supplied must be of the form Array{Float64, 1} -> Float64
function gradient(f::Function, x::Array{Float64, 1}, h::Float64, twoside::Bool)
	k = length(x)
	ans = Array(Float64, k)
	for i = 1:k
		v 		= zeros(k)
		v[i] 	= 1.0
		ans[i] 	= dirderivative(f, v, x, h, twoside)
	end
	ans
end
gradient(f::Function, x::Array{Float64, 1}, h::Float64) = gradient(f, x, h, true)
gradient(f::Function, x::Array{Float64, 1}) = gradient(f, x, 0.0001)
gradient(f::Function, x::Array{Int64, 1}) = gradient(f, float(x))
gradient(f::Function) = x::Array -> gradient(f, x)

# Function for calculation of Jacobians. One method for functions of the form Float64 -> Array{Float64, 1} and one method for function of the form Array{Float64, 1} -> Array{Float64, 1}.
function jacobian(f::Function, x::Float64, h::Float64)
	l = length(f(x))
	ans = Array(Float64, l)
	for i = 1:l
		ans[i] = derivative(y -> f(y)[i], x, h)
	end
	ans
end
function jacobian(f::Function, x::Array{Float64, 1}, h::Float64)
	k = length(x)
	l = length(f(x))
	ans = Array(Float64, (k,l))
	for i = 1:l
		ans[:,i] = gradient(y->f(y)[i], x, h)
	end
	ans'
end
jacobian(f::Function, x) = jacobian(f, x, 0.0001)

# Function for calculation of the Hessian. Function argument must be of the form Array{Float64, 1} -> Float64
hessian(f::Function, x, h::Float64) = jacobian(gradient(f), x, h)
hessian(f::Function, x) = hessian(f, x, 0.0001)
hessian(f::Function) = x -> hessian(f, x)