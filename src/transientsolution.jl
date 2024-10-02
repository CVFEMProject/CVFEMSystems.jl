mutable struct TransientSolution{T, N, A, B} <: AbstractDiffEqArray{T, N, A}
    u::A
    t::B
end

(sol::TransientSolution)(t) = _interpolate(sol, t)

function _interpolate(sol, t)
    if isapprox(t, sol.t[1]; atol = 1.0e-10 * abs(sol.t[2] - sol.t[1]))
        return sol[1]
    end
    idx = searchsortedfirst(sol.t, t)
    if idx == 1 || idx > length(sol)
        return nothing
    end
    if t == sol.t[idx - 1]
        return sol[idx - 1]
    else
        retval = similar(sol.u[idx])
        dt = sol.t[idx] - sol.t[idx - 1]
        a = (sol.t[idx] - t) / dt
        b = (t - sol.t[idx - 1]) / dt
        retval .= a * sol.u[idx - 1] + b * sol.u[idx]
    end
end

function TransientSolution(vec::AbstractVector{T}, ts, ::NTuple{N}) where {T, N}
    TransientSolution{eltype(T), N, typeof(vec), typeof(ts)}(vec, ts)
end

TransientSolution(vec::AbstractVector, ts::AbstractVector) = TransientSolution(vec, ts, (size(vec[1])..., length(vec)))
Base.append!(s::TransientSolution, t::Real, sol::AbstractArray) = push!(s.t, t), push!(s.u, sol)
