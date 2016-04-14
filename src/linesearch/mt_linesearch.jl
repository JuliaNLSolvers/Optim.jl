#   Translation of Matlab version by John Myles White
#   Translation of minpack subroutine cvsrch
#   Dianne O'Leary   July 1991
#
#     **********
#
#     Subroutine cvsrch
#
#     The purpose of cvsrch is to find a step which satisfies
#     a sufficient decrease condition and a curvature condition.
#     The user must provide a subroutine which calculates the
#     function and the gradient.
#
#     At each stage the subroutine updates an interval of
#     uncertainty with endpoints stx and sty. The interval of
#     uncertainty is initially chosen so that it contains a
#     minimizer of the modified function
#
#          f(x + stp * s) - f(x) - ftol * stp * (gradf(x)' * s).
#
#     If a step is obtained for which the modified function
#     has a nonpositive function value and nonnegative derivative,
#     then the interval of uncertainty is chosen so that it
#     contains a minimizer of f(x + stp * s).
#
#     The algorithm is designed to find a step which satisfies
#     the sufficient decrease condition
#
#           f(x + stp * s) <= f(x) + ftol * stp * (gradf(x)' * s),
#
#     and the curvature condition
#
#           abs(gradf(x + stp * s)' * s)) <= gtol * abs(gradf(x)' * s).
#
#     If ftol is less than gtol and if, for example, the function
#     is bounded below, then there is always a step which satisfies
#     both conditions. If no step can be found which satisfies both
#     conditions, then the algorithm usually stops when rounding
#     errors prevent further progress. In this case stp only
#     satisfies the sufficient decrease condition.
#
#     The subroutine statement is
#
#        subroutine cvsrch(fcn,n,x,f,g,s,stp,ftol,gtol,xtol,
#                          stpmin,stpmax,maxfev,info,nfev,wa)
#
#     where
#
#	fcn is the name of the user-supplied subroutine which
#    calculates the function and the gradient.  fcn must
#    be declared in an external statement in the user
#    calling program, and should be written as follows.
#
#    function [f,g] = fcn(n,x) (Matlab)
#                     (10/2010 change in documentation)
#                     (derived from Fortran subroutine fcn(n,x,f,g))
#    integer n
#    f
#    x(n),g(n)
#
#    Calculate the function at x and
#    return this value in the variable f.
#    Calculate the gradient at x and
#    return this vector in g.
#
#  n is a positive integer input variable set to the number
#	  of variables.
#
#	x is an array of length n. On input it must contain the
#	  base point for the line search. On output it contains
#    x + stp * s.
#
#	f is a variable. On input it must contain the value of f
#    at x. On output it contains the value of f at x + stp * s.
#
#	g is an array of length n. On input it must contain the
#    gradient of f at x. On output it contains the gradient
#    of f at x + stp * s.
#
#	s is an input array of length n which specifies the
#    search direction.
#
#	stp is a nonnegative variable. On input stp contains an
#    initial estimate of a satisfactory step. On output
#    stp contains the final estimate.
#
#  ftol and gtol are nonnegative input variables. Termination
#    occurs when the sufficient decrease condition and the
#    directional derivative condition are satisfied.
#
#	xtol is a nonnegative input variable. Termination occurs
#    when the relative width of the interval of uncertainty
#	  is at most xtol.
#
#	stpmin and stpmax are nonnegative input variables which
#	  specify lower and upper bounds for the step.
#
#	maxfev is a positive integer input variable. Termination
#    occurs when the number of calls to fcn is at least
#    maxfev by the end of an iteration.
#
#	info is an integer output variable set as follows:
#
#	  info = 0  Improper input parameters.
#
#	  info = 1  The sufficient decrease condition and the
#              directional derivative condition hold.
#
#	  info = 2  Relative width of the interval of uncertainty
#		         is at most xtol.
#
#	  info = 3  Number of calls to fcn has reached maxfev.
#
#	  info = 4  The step is at the lower bound stpmin.
#
#	  info = 5  The step is at the upper bound stpmax.
#
#	  info = 6  Rounding errors prevent further progress.
#              There may not be a step which satisfies the
#              sufficient decrease and curvature conditions.
#              Tolerances may be too small.
#
#    nfev is an integer output variable set to the number of
#         calls to fcn.
#
#     Argonne National Laboratory. MINPACK Project. June 1983
#     Jorge J. More', David J. Thuente
#
#     **********

# Returns x, f, g, stp, info, nfev
# TODO: Decide whether to update x, f, g and info
#       or just return step and nfev and let existing code do its job

function mt_linesearch!{T}(fcn::Union{DifferentiableFunction,
                                      TwiceDifferentiableFunction},
                         x::Vector,
                         s::Vector,
                         new_x::Vector,
                         g::Vector,
                         lsr::LineSearchResults{T},
                         c::Real,
                         mayterminate::Bool;
                         n::Integer = length(x),
                         stp::Real = 1.0,
                         ftol::Real = 1e-4,
                         gtol::Real = 0.9,
                         xtol::Real = 1e-8,
                         stpmin::Real = 1e-16,
                         stpmax::Real = 65536.0,
                         maxfev::Integer = 100)

   info = 0
   info_cstep = 1 # Info from step

   # Count function and gradient calls
   f_calls = 0
   g_calls = 0

   f = fcn.fg!(x, g)
   f_calls += 1
   g_calls += 1

   #
   # Check the input parameters for errors.
   #

   if n <= 0 || stp <= 0.0 || ftol < 0.0 || gtol < 0.0 ||
      xtol < 0.0 || stpmin < 0.0 || stpmax < stpmin || maxfev <= 0
      throw(ArgumentError("Invalid parameters to mure_thuente_line_search"))
   end

   #
   # Compute the initial gradient in the search direction
   # and check that s is a descent direction.
   #

   dginit = vecdot(g, s)
   if dginit >= 0.0
      throw(ArgumentError("Search direction is not a direction of descent"))
   end

   #
   # Initialize local variables.
   #

   bracketed = false
   stage1 = true
   nfev = 0
   finit = f
   dgtest = ftol * dginit
   width = stpmax - stpmin
   width1 = 2 * width
   copy!(new_x, x)
   # Keep this across calls
   # Replace with new_x?

   #
   # The variables stx, fx, dgx contain the values of the step,
   # function, and directional derivative at the best step.
   # The variables sty, fy, dgy contain the value of the step,
   # function, and derivative at the other endpoint of
   # the interval of uncertainty.
   # The variables stp, f, dg contain the values of the step,
   # function, and derivative at the current step.
   #

   stx = 0.0
   fx = finit
   dgx = dginit
   sty = 0.0
   fy = finit
   dgy = dginit

   while true
      #
      # Set the minimum and maximum steps to correspond
      # to the present interval of uncertainty.
      #

      if bracketed
         stmin = min(stx, sty)
         stmax = max(stx, sty)
      else
         stmin = stx
         stmax = stp + 4 * (stp - stx) # Why 4?
      end

      #
      # Force the step to be within the bounds stpmax and stpmin
      #

      stp = max(stp, stpmin)
      stp = min(stp, stpmax)

      #
      # If an unusual termination is to occur then let
      # stp be the lowest point obtained so far.
      #

      if (bracketed && (stp <= stmin || stp >= stmax)) ||
           nfev >= maxfev-1 || info_cstep == 0 ||
           (bracketed && stmax - stmin <= xtol * stmax)
         stp = stx
      end

      #
      # Evaluate the function and gradient at stp
      # and compute the directional derivative.
      #

      for i in 1:n
         new_x[i] = x[i] + stp * s[i] # TODO: Use x_new here
      end
      f = fcn.fg!(new_x, g)
      f_calls += 1
      g_calls += 1
      nfev += 1 # This includes calls to f() and g!()
      dg = vecdot(g, s)
      ftest1 = finit + stp * dgtest

      #
      # Test for convergence.
      #

      # What's does info_cstep stand for?

      if (bracketed && (stp <= stmin || stp >= stmax)) || info_cstep == 0
         info = 6
      end
      if stp == stpmax && f <= ftest1 && dg <= dgtest
         info = 5
      end
      if stp == stpmin && (f > ftest1 || dg >= dgtest)
         info = 4
      end
      if nfev >= maxfev
         info = 3
      end
      if bracketed && stmax - stmin <= xtol * stmax
         info = 2
      end
      if f <= ftest1 && abs(dg) <= -gtol * dginit
         info = 1
      end

      #
      # Check for termination.
      #

      if info != 0
         return stp, f_calls, g_calls
      end

      #
      # In the first stage we seek a step for which the modified
      # function has a nonpositive value and nonnegative derivative.
      #

      if stage1 && f <= ftest1 && dg >= min(ftol, gtol) * dginit
         stage1 = false
      end

      #
      # A modified function is used to predict the step only if
      # we have not obtained a step for which the modified
      # function has a nonpositive function value and nonnegative
      # derivative, and if a lower function value has been
      # obtained but the decrease is not sufficient.
      #

      if stage1 && f <= fx && f > ftest1
         #
         # Define the modified function and derivative values.
         #
         fm = f - stp * dgtest
         fxm = fx - stx * dgtest
         fym = fy - sty * dgtest
         dgm = dg - dgtest
         dgxm = dgx - dgtest
         dgym = dgy - dgtest
         #
         # Call cstep to update the interval of uncertainty
         # and to compute the new step.
         #
         stx, fxm, dgxm,
         sty, fym, dgym,
         stp, fm, dgm,
         bracketed, info_cstep =
           cstep(stx, fxm, dgxm, sty, fym, dgym,
                 stp, fm, dgm, bracketed, stmin, stmax)
         #
         # Reset the function and gradient values for f.
         #
         fx = fxm + stx * dgtest
         fy = fym + sty * dgtest
         dgx = dgxm + dgtest
         dgy = dgym + dgtest
      else
         #
         # Call cstep to update the interval of uncertainty
         # and to compute the new step.
         #
         stx, fx, dgx,
         sty, fy, dgy,
         stp, f, dg,
         bracketed, info_cstep =
           cstep(stx, fx, dgx, sty, fy, dgy,
                 stp, f, dg, bracketed, stmin, stmax)
      end

      #
      # Force a sufficient decrease in the size of the
      # interval of uncertainty.
      #

      if bracketed
         if abs(sty - stx) >= 0.66 * width1
            stp = stx + 0.5 * (sty - stx)
         end
         width1 = width
         width = abs(sty - stx)
      end
   end # while
end # function
