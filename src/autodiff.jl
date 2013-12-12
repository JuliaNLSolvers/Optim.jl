using DualNumbers

function autodiff{T <: Real}(f,
                             x::Vector{T},
                             gradient_output,
                             dualvec)
    # Assume f doesn't modify the input
    # otherwise we need to make a copy
    for i in 1:length(x)
        dualvec[i] = Dual(x[i], zero(T))
    end
    for i in 1:length(x)
        dualvec[i] = Dual(real(dualvec[i]), one(T))
        result = f(dualvec)
        gradient_output[i] = epsilon(result)
        dualvec[i] = Dual(real(dualvec[i]), zero(T))
    end

end

# generates a function that computes the gradient of f(x)
# assuming that f takes a Vector{T} of length n
function autodiff(f,T,n)
    dualvec = Array(Dual{T},n)
    function g!(x, gradient_output)
        autodiff(f,x, gradient_output, dualvec)
    end
    return DifferentiableFunction(f,g!)
end
