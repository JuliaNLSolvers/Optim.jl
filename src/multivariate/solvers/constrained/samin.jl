# """
# History: Based on Octave code samin.cc, by Michael Creel,
# which was originally based on Gauss code by E.G. Tsionas. A source
# for the Gauss code is http://web.stanford.edu/~doubleh/otherpapers/sa.txt
# The original Fortran code by W. Goffe is at
# http://www.degruyter.com/view/j/snde.1996.1.3/snde.1996.1.3.1020/snde.1996.1.3.1020.xml?format=INT
# Tsionas and Goffe agreed to MIT licensing of samin.jl in email
# messages to Creel.
#
# This Julia code uses the same names for control variables,
# for the most part. A notable difference is that the initial
# temperature can be found automatically to ensure that the active
# bounds when the temperature begins to reduce cover the entire
# parameter space (defined as a n-dimensional rectangle that is the
# Cartesian product of the(lb_i, ub_i), i = 1,2,..n. The code also
# allows for parameters to be restricted, by setting lb_i = ub_i,
# for the appropriate i.

"""
# SAMIN
## Constructor
```julia
SAMIN(; nt::Int = 5     # reduce temperature every nt*ns*dim(x_init) evaluations
        ns::Int = 5     # adjust bounds every ns*dim(x_init) evaluations
        rt::T = 0.9     # geometric temperature reduction factor: when temp changes, new temp is t=rt*t
        neps::Int = 5   # number of previous best values the final result is compared to
        f_tol::T = 1e-12 # the required tolerance level for function value comparisons
        x_tol::T = 1e-6 # the required tolerance level for x
        coverage_ok::Bool = false, # if false, increase temperature until initial parameter space is covered
        verbosity::Int = 1) # scalar: 0, 1, 2 or 3 (default = 1).
```
## Description
The `SAMIN` method implements the Simulated Annealing algorithm for problems with
bounds constrains a described in Goffe et. al. (1994) and Goffe (1996). The
algorithm

## References
 - Goffe, et. al. (1994) "Global Optimization of Statistical Functions with Simulated Annealing", Journal of Econometrics, V. 60, N. 1/2.
 - Goffe, William L. (1996) "SIMANN: A Global Optimization Algorithm using Simulated Annealing " Studies in Nonlinear Dynamics & Econometrics, Oct96, Vol. 1 Issue 3.
"""
@with_kw struct SAMIN{T}<:ZerothOrderOptimizer
    nt::Int = 5 # reduce temperature every nt*ns*dim(x_init) evaluations
    ns::Int = 5 # adjust bounds every ns*dim(x_init) evaluations
    rt::T = 0.9 # geometric temperature reduction factor: when temp changes, new temp is t=rt*t
    neps::Int = 5 # number of previous best values the final result is compared to
    f_tol::T = 1e-12 # the required tolerance level for function value comparisons
    x_tol::T = 1e-6 # the required tolerance level for x
    coverage_ok::Bool = false # if false, increase temperature until initial parameter space is covered
    verbosity::Int = 1 # scalar: 0, 1, 2 or 3 (default = 1: see final results).
end
# * verbosity: scalar: 0, 1, 2 or 3 (default = 1).
#     * 0 = no screen output
#     * 1 = only final results to screen
#     * 2 = summary every temperature change, without param values
#     * 3 = summary every temperature change, with param values
#         covered by the trial values. 1: start decreasing temperature immediately
Base.summary(::SAMIN) = "SAMIN"

function optimize(obj_fn, lb::AbstractArray, ub::AbstractArray, x::AbstractArray{Tx}, method::SAMIN, options::Options = Options()) where Tx

    t0 = time() # Initial time stamp used to control early stopping by options.time_limit

    hline = "="^80
    d = NonDifferentiable(obj_fn, x)

    tr = OptimizationTrace{typeof(value(d)), typeof(method)}()
    tracing = options.store_trace || options.show_trace || options.extended_trace || options.callback != nothing

    @unpack nt, ns, rt, neps, f_tol, x_tol, coverage_ok, verbosity = method
    verbose = verbosity > 0

    x0 = copy(x)
    n = size(x,1) # dimension of parameter
    #  Set initial values
    nacc = 0 # total accepted trials
    t = 2.0 # temperature - will initially rise or fall to cover parameter space. Then it will fall
    converge = 0 # convergence indicator 0 (failure), 1 (normal success), or 2 (convergence but near bounds)
    # most recent values, to compare to when checking convergend
    fstar = typemax(Float64)*ones(neps)
    # Initial obj_value
    xopt = copy(x)
    f_old = value!(d, x)
    fopt = copy(f_old) # give it something to compare to
    details = [f_calls(d) t fopt xopt']
    bounds = ub - lb
    # check for out-of-bounds starting values
    for i = 1:n
        if(( x[i] > ub[i]) || (x[i] < lb[i]))
            error("samin: initial parameter %d out of bounds\n", i)
        end
    end

    options.show_trace && print_header(method)
    iteration = 0
    trace!(tr, d, (x=xopt, iteration=iteration), iteration, method, options, time()-t0)

    # main loop, first increase temperature until parameter space covered, then reduce until convergence
    while (converge==0)
        # statistics to report at each temp change, set back to zero
        nup = 0
        nrej = 0
        nnew = 0
        ndown = 0
        lnobds = 0
        # repeat nt times then adjust temperature
        for m = 1:nt
            # repeat ns times, then adjust bounds
            nacp = zeros(n)
            for j = 1:ns
                # generate new point by taking last and adding a random value
                # to each of elements, in turn
                for h = 1:n
                    iteration += 1
                    # new Sept 2011, if bounds are same, skip the search for that vbl.
                    # Allows restrictions without complicated programming
                    if (lb[h] != ub[h])
                        xp = copy(x)
                        xp[h] += (Tx(2.0) * rand(Tx) - Tx(1.0)) * bounds[h]
                        if (xp[h] < lb[h]) || (xp[h] > ub[h])
                            xp[h] = lb[h] + (ub[h] - lb[h]) * rand(Tx)
                            lnobds += 1
                        end
                        # Evaluate function at new point
                        f_proposal = value(d, xp)
                        #  Accept the new point if the function value decreases
                        if (f_proposal <= f_old)
                            x = copy(xp)
                            f_old = f_proposal
                            nacc += 1 # total number of acceptances
                            nacp[h] += 1 # acceptances for this parameter
                            nup += 1
                            #  If lower than any other point, record as new optimum
                            if f_proposal < fopt
                                xopt = copy(xp)
                                fopt = f_proposal
                                d.F = f_proposal
                                nnew +=1
                                details = [details; [f_calls(d) t f_proposal xp']]
                            end
                        # If the point is higher, use the Metropolis criteria to decide on
                        # acceptance or rejection.
                        else
                            p = exp(-(f_proposal - f_old) / t)
                            if rand(Tx) < p
                                x = copy(xp)
                                f_old = copy(f_proposal)
                                d.F = f_proposal
                                nacc += 1
                                nacp[h] += 1
                                ndown += 1
                            else
                                nrej += 1
                            end
                        end
                    end

                    if tracing
                        # update trace; callbacks can stop routine early by returning true
                        stopped_by_callback =  trace!(tr, d, (x=xopt,iteration=iteration), iteration, method, options, time()-t0)
                    end

                    # If options.iterations exceeded, terminate the algorithm
                    if f_calls(d) >= options.iterations

                        if verbose
                            println(hline)
                            println("SAMIN results")
                            println("NO CONVERGENCE: MAXEVALS exceeded")
                            @printf("\n     Obj. value:  %16.5f\n\n", fopt)
                            println("       parameter      search width")
                            for i=1:n
                                @printf("%16.5f  %16.5f \n", xopt[i], bounds[i])
                            end
                            println(hline)
                        end
                        converge = 0

                        return MultivariateOptimizationResults(method,
                                                                x0,# initial_x,
                                                                xopt, #pick_best_x(f_incr_pick, state),
                                                                fopt, # pick_best_f(f_incr_pick, state, d),
                                                                f_calls(d), #iteration,
                                                                f_calls(d) >= options.iterations, #iteration == options.iterations,
                                                                false, # x_converged,
                                                                0.0,#T(options.x_tol),
                                                                0.0,#T(options.x_tol),
                                                                NaN,# x_abschange(state),
                                                                NaN,# x_abschange(state),
                                                                false,# f_converged,
                                                                0.0,#T(options.f_tol),
                                                                0.0,#T(options.f_tol),
                                                                NaN,#f_abschange(d, state),
                                                                NaN,#f_abschange(d, state),
                                                                false,#g_converged,
                                                                0.0,#T(options.g_tol),
                                                                NaN,#g_residual(d),
                                                                false, #f_increased,
                                                                tr,
                                                                f_calls(d),
                                                                g_calls(d),
                                                                h_calls(d),
                                                                true)
                    end
                end
            end
            #  Adjust bounds so that approximately half of all evaluations are accepted
            test = 0
            for i = 1:n
                if (lb[i] != ub[i])
                    ratio = nacp[i] / ns
                    if(ratio > 0.6) bounds[i] = bounds[i] * (1.0 + 2.0 * (ratio - 0.6) / 0.4) end
                    if(ratio < .4) bounds[i] = bounds[i] / (1.0 + 2.0 * ((0.4 - ratio) / 0.4)) end
                    # keep within initial bounds
                    if(bounds[i] > (ub[i] - lb[i]))
                        bounds[i] = ub[i] - lb[i]
                        test += 1
                    end
                else
                    test += 1 # make sure coverage check passes for the fixed parameters
                end
            end
            nacp = 0 # set back to zero
            # check if we cover parameter space, if we have yet to do so
            if !coverage_ok
                coverage_ok = (test == n)
            end
        end

        # intermediate output, if desired
        if verbosity > 1
            println(hline)
            println("samin: intermediate results before next temperature change")
            println("temperature: ", round(t, digits=5))
            println("current best function value: ", round(fopt, digits=5))
            println("total evaluations so far: ", f_calls(d))
            println("total moves since last temperature reduction: ", nup + ndown + nrej)
            println("downhill: ", nup)
            println("accepted uphill: ", ndown)
            println("rejected uphill: ", nrej)
            println("out of bounds trials: ", lnobds)
            println("new minima this temperature: ", nnew)
            println()
            println("       parameter      search width")
            for i=1:n
                @printf("%16.5f  %16.5f \n", xopt[i], bounds[i])
            end
            println(hline*"\n")
        end
        # Check for convergence, if we have covered the parameter space
        if coverage_ok
            # last value close enough to last neps values?
            fstar[1] = f_old
            test = 0
            for i=1:neps
                test += (abs(f_old - fstar[i]) > f_tol)
            end
            test = (test > 0) # if different from zero, function conv. has failed
            # last value close enough to overall best?
            if (((fopt - f_old) <= f_tol) && (!test))
                # check for bound narrow enough for parameter convergence
                for i = 1:n
                    if (bounds[i] > x_tol)
                        converge = 0 # no conv. if bounds too wide
                        break
                    else
                        converge = 1
                    end
                end
            end
            # check if optimal point is near boundary of parameter space, and change message if so
            if (converge == 1) && (lnobds > 0)
                converge = 2
            end
            # Like to see the final results?
            if (converge > 0)
                if verbose
                    println(hline)
                    println("SAMIN results")
                    if (converge == 1)
                        println("==> Normal convergence <==")
                    end
                    if (converge == 2)
                        printstyled("==> WARNING <==\n", color=:red)
                        println("Last point satisfies convergence criteria, but is near")
                        println("boundary of parameter space.")
                        println(lnobds, " out of  ", (nup+ndown+nrej), " evaluations were out of bounds in the last round.")
                        println("Expand bounds and re-run, unless this is a constrained minimization.")
                    end
                    println("total number of objective function evaluations: ", f_calls(d))
                    @printf("\n     Obj. value:  %16.10f\n\n", fopt)
                    println("       parameter      search width")
                    for i=1:n
                        @printf("%16.5f  %16.5f \n", xopt[i], bounds[i])
                    end
                    println(hline*"\n")
                end
            end
            # Reduce temperature, record current function value in the
            # list of last "neps" values, and loop again
            t *= rt
            for i = neps:-1:2
                fstar[i] = fstar[i-1]
            end
            f_old = copy(fopt)
            x = copy(xopt)
        else  # coverage not ok - increase temperature quickly to expand search area
            t *= 10.0
            for i = neps:-1:2
                fstar[i] = fstar[i-1]
            end
            f_old = fopt
            x = xopt
        end
    end
    return MultivariateOptimizationResults(method,
                                            x0,# initial_x,
                                            xopt, #pick_best_x(f_incr_pick, state),
                                            fopt, # pick_best_f(f_incr_pick, state, d),
                                            f_calls(d), #iteration,
                                            f_calls(d) >= options.iterations, #iteration == options.iterations,
                                            false, # x_converged,
                                            0.0,#T(options.x_tol),
                                            0.0,#T(options.x_tol),
                                            NaN,# x_abschange(state),
                                            NaN,# x_abschange(state),
                                            false,# f_converged,
                                            0.0,#T(options.f_tol),
                                            0.0,#T(options.f_tol),
                                            NaN,#f_abschange(d, state),
                                            NaN,#f_abschange(d, state),
                                            false,#g_converged,
                                            0.0,#T(options.g_tol),
                                            NaN,#g_residual(d),
                                            false, #f_increased,
                                            tr,
                                            f_calls(d),
                                            g_calls(d),
                                            h_calls(d),
                                            true)

end

# TODO
# Handle traces
# * details: a px3 matrix. p is the number of times improvements were found.
#            The columns record information at the time an improvement was found
#            * first: cumulative number of function evaluations
#            * second: temperature
#            * third: function value
#
# Add doc entry
