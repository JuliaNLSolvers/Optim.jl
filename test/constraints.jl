using Optim, Base.Test

b = @inferred(Optim.ConstraintBounds([0.0, 0.5, 2.0], [1.0, 1.0, 2.0], [5.0, 3.8], [5.0, 4.0]))
@test b.eqx == [3]
@test b.valx == [2.0]
@test b.ineqx == [1,2,2]
@test b.σx == [-1,1,-1]
@test b.bx == [1.0,0.5,1.0]
@test b.iz == [1]
@test b.σz == [1]
@test b.eqc == [1]
@test b.valc == [5]
@test b.ineqc == [2,2]
@test b.σc == [1,-1]
@test b.bc == [3.8,4.0]

b = @inferred(Optim.ConstraintBounds(Float64[], Float64[], [5.0, 3.8], [5.0, 4.0]))
for fn in (:eqx, :valx, :ineqx, :σx, :bx, :iz, :σz)
    @test isempty(getfield(b, fn))
end
@test b.eqc == [1]
@test b.valc == [5]
@test b.ineqc == [2,2]
@test b.σc == [1,-1]
@test b.bc == [3.8,4.0]

ba = Optim.ConstraintBounds([], [], [5.0, 3.8], [5.0, 4.0])
@test eltype(ba) == Float64

@test_throws ArgumentError Optim.ConstraintBounds([0.0, 0.5, 2.0], [1.0, 1.0, 2.0], [5.0, 4.8], [5.0, 4.0])
@test_throws DimensionMismatch Optim.ConstraintBounds([0.0, 0.5, 2.0], [1.0, 1.0], [5.0, 4.8], [5.0, 4.0])

nothing
