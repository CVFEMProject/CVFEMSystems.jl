"""
    struct CellData

Data structure holding information for user cell assembly.
"""
mutable struct CellData{Tgc, Tgn, Ten, Tcm, Tsg, Ttm, Tuold}
    icell::Int64
    region::Int64
    globalcoord::Tgc
    globalnodes::Tgn
    edgenodes::Ten
    factdim::Int64
    coordmatrix::Tcm
    shapegradients::Tsg
    volume::Float64
    transmission::Ttm
    uold::Tuold
    time::Float64
    tstep::Float64
end

"""
    CellData(coord,edgenodes)

Constructor for cell data from coordinate array and nodes per edge matrix.
"""
function CellData(coord, time, tstep)
    spacedim = size(coord, 1)
    globalcoord = reinterpret(reshape, SVector{spacedim, Float64}, coord)
    celldim = spacedim + 1
    icell = 0
    region = 0
    edgenodes=local_edgenodes[spacedim]
    xedgenodes = @MMatrix zeros(Int64, 2, size(edgenodes, 2))
    xedgenodes .= edgenodes
    globalnodes = @MVector zeros(Int64, celldim)
    factdim = prod(1:spacedim)
    coordmatrix = @MMatrix zeros(spacedim, spacedim)
    shapegradients = @MMatrix zeros(celldim, spacedim)
    transmission = @MVector zeros(size(edgenodes, 2)) # Transmission coefficients
    uold= @MVector zeros(celldim)
    CellData(icell,
             region,
             globalcoord,
             globalnodes,
             xedgenodes,
             factdim,
             coordmatrix,
             shapegradients,
             0.0,
             transmission,
             uold,
             time,
             tstep)
end

"""
    nnodes(celldata)

Number of nodes of cell. To be used in user callback.
"""
nnodes(celldata::CellData) = size(celldata.shapegradients, 1)

"""
    edges(celldata)

Number of nodes of cell. To be used in user callback.
"""
nedges(celldata::CellData) = size(celldata.edgenodes, 2)

"""
    spacedim(celldata)

Space dimension. To be used in user callback.
"""
spacedim(celldata::CellData) = size(celldata.shapegradients, 2)

"""
    coord(celldata,i)

Coordinates of i-th node. To be used in user callback.
"""
coord(celldata::CellData, il) = celldata.globalcoord[celldata.globalnodes[il]]

"""
   edgenode(celldata,inode,iedge)

Local number of node of edge.
"""
edgenode(celldata::CellData, i, ie) = celldata.edgenodes[i, ie]

"""
    globalnode(celldata, i)

Global (grid) node number of cell node.
"""
globalnode(celldata::CellData, il) = celldata.globalnodes[il]

"""
    volume(celldata)

Return volume (volume,area, length) of cell.
"""
volume(celldata::CellData) = celldata.volume

"""
        transmission(celldata, Λ)

Calculate and edge transmission coefficients in an array of length equal
to number of cell edges.
"""
function transmission(celldata, Λ)
    femfactors!(celldata.transmission, celldata.shapegradients, Λ, celldata.edgenodes)
    for ie = 1:nedges(celldata)
        celldata.transmission[ie] *= celldata.volume
    end
    celldata.transmission
end

"""
    update!(celldata,coord, cellnodes, icell)

Update cell data for current cell.
"""
function update!(celldata::CellData, coord, cellnodes, icell)
    celldata.icell = icell
    for i = 1:nnodes(celldata)
        celldata.globalnodes[i] = cellnodes[i, icell]
    end
    coordmatrix!(celldata.coordmatrix, coord, cellnodes, icell)
    celldata.volume = abs(det(celldata.coordmatrix)) / celldata.factdim
    femgrad!(celldata.shapegradients, celldata.coordmatrix)
end

"""
    struct BNodeData

Data structure holding information for user boundary condition assembly.
"""
mutable struct BNodeData{Tgc}
    ibface::Int64
    region::Int64
    globalcoord::Tgc
    globalnode::Int64
    volume::Float64
end

"""
    BNodeData(coord)

Constructor for BNodeData
"""
function BNodeData(coord)
    spacedim = size(coord, 1)
    globalcoord = reinterpret(reshape, SVector{spacedim, Float64}, coord)
    ibface = 0
    region = 0
    globalnode = 0
    BNodeData(ibface,
              region,
              globalcoord,
              globalnode,
              0.0)
end

"""
    volume(bnodedata)


Volume (area, length, 1)  corresponding to bondary node.
"""
volume(bnodedata::BNodeData) = bnodedata.nodevolume

"""
    coord(bonodedata)

Coordinates of boundary node.
"""
coord(bnodedata::BNodeData) = bnodedata.globalcoord[bnodedata.globalnode]

"""
    globalnode(bnodedata)

Global node number of boundary node.
"""
globalnode(bnodedata::BNodeData) = bnodedata.globalnode

function update!(bnodedata::BNodeData, coordinates, bfacenodes, ibface, inode, volume)
    bnodedata.ibface = ibface
    bnodedata.globalnode = bfacenodes[inode, ibface]
    bnodedata.volume = volume
end

"""
    dirichlet!(bnodedata,y,u,value)

Set Dirichlet value for bnodedata.
"""
function dirichlet!(bnodedata, y, u, val)
    y[1] = Dirichlet() * (u[1] - val)
end

"""
    struct CVFEMSystem

Struct which contains grid, cell evaluator, bnode evaluator and usertdata.
"""
struct CVFEMSystem{G<: ExtendableGrid,FC<:Function,FB<:Function, UD}
    grid::G
    celleval::FC
    bnodeeval::FB
    userdata::UD
end

"""
    zeros(sys)

Return a zero solution vector of system.
"""
Base.zeros(sys)=zeros(num_nodes(sys.grid))


"""
    eval_res_jac!(sys,U,Uold,Res,Jac,time, tstep)

Evaluate global residual and jacobian for one time step of system.
"""
function eval_res_jac!(sys::CVFEMSystem{G,FC,FB,UD}, U, Uold, Res, Jac, time, tstep) where {G,FC,FB,UD}
    (;grid, celleval, bnodeeval, userdata)=sys

    coordinates = grid[Coordinates]
    cellnodes = grid[CellNodes]
    bfacenodes = grid[BFaceNodes]

    spacedim = size(coordinates, 1)
    celldim = spacedim + 1
    celldata = CellData(coordinates, time, tstep)
    ncells = size(cellnodes, 2)

    bnodedata = BNodeData(coordinates)
    nbfaces = size(bfacenodes, 2)

    wrap_celleval(y, u) = celleval(y, u, celldata, userdata)
    wrap_bnodeeval(y, u) = bnodeeval(y, u, bnodedata, userdata)
    
    uclocal = zeros(celldim)
    yclocal = zeros(celldim)
    cresult = DiffResults.JacobianResult(yclocal, uclocal)
    cconfig = ForwardDiff.JacobianConfig(wrap_celleval, yclocal, uclocal)

    
    ublocal = zeros(1)
    yblocal = zeros(1)
    bresult = DiffResults.JacobianResult(yblocal, ublocal)
    bconfig = ForwardDiff.JacobianConfig(wrap_bnodeeval, yblocal, ublocal)

    for icell = 1:ncells
        update!(celldata, coordinates, cellnodes, icell)
        # Copy solution to ulocal
        for il = 1:celldim
            uclocal[il] = U[cellnodes[il, icell]]
            celldata.uold[il] = Uold[cellnodes[il, icell]]
        end
        
        # AD residual and jacobian
        yclocal.=0.0
        ForwardDiff.vector_mode_jacobian!(cresult, wrap_celleval, yclocal, uclocal, cconfig)
        cres = DiffResults.value(cresult)
        cjac = DiffResults.jacobian(cresult)

        # Assemble into global data
        for il = 1:celldim
            ig = cellnodes[il, icell]
            Res[ig] += cres[il]
            for jl = 1:celldim
                jg = cellnodes[jl, icell]
                Jac[ig, jg] += cjac[il, jl]
            end
        end
    end # icell
    
    for ibface = 1:nbfaces
        vol = bfacevolume(coordinates, bfacenodes, ibface)
        for il = 1:spacedim
            update!(bnodedata, coordinates, bfacenodes, ibface, il, vol)
            ig = bfacenodes[il, ibface]
            ublocal[1] = U[ig]
            yblocal.=0.0
            ForwardDiff.vector_mode_jacobian!(bresult, wrap_bnodeeval, yblocal, ublocal, bconfig)
            bres = DiffResults.value(bresult)
            bjac = DiffResults.jacobian(bresult)
            Res[ig] += bres[1]
            Jac[ig, ig] += bjac[1, 1]
        end
    end
end

"""
    solve_step!(system, U,Uold, Res, Jac, time, tstep; kwargs...)
    
Solve one implicit Euler timestep of system, or stationary problem if tstep=Inf.
"""
function solve_step!(sys::CVFEMSystem{G,FC,FB,UD},U,Uold,Res,Jac,time,tstep;
                    tol = 1.0e-10,
                    maxiter = 20,
                    damp_initial=1.0,
                    damp_growth=1.2,
                    log=false)  where {G,FC,FB,UD};
    it = 1
    nalloc = 0
    damp=damp_initial
    while it < maxiter
        nonzeros(Jac) .= 0.0
        Res .= 0.0
        eval_res_jac!(sys,U,Uold, Res, Jac, time, tstep)
        update=Jac\Res
        U.-=damp.*update
        nm = norm(update, Inf)
        if log
            @info it, nm
        end
        if nm < tol
            if nalloc > 0
                @warn "allocations in assembly loop"
            end
            return U
        end
        damp=min(damp*damp_growth,1.0)
        it += 1
    end  # while it
    error("no convergence after maxiter=$(maxiter) iterations")
end

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



"""
    solve(sys; kwargs...)

Solve system.
Keyword arguments and their defaults
- `times=nothing`: AbstractVector of time discretization points. If not given solve stationary problem.
- `inival=zeros(sys)`: initial value
- `tol = 1.0e-10`: stopping tolerance for Newton iteration. Compared with infinity norm of Newton update.
- `maxiter = 20` Maximum number of Newton iterations
- `damp_initial=1.0`: Initial value of damping for Newton
- `damp_growth=1.2`: Newton damp growth factor
- `log=false`: Trigger logging
"""
function SciMLBase.solve(sys::CVFEMSystem{G,FC,FB,UD};
                         times=nothing,
                         inival=zeros(sys), kwargs...) where {G,FC,FB,UD}
    (;grid)=sys
    N = num_nodes(grid)
    Jac = ExtendableSparseMatrix(N, N)
    Res = zeros(N)
    U = copy(inival)
    Uold = copy(inival)
    if isnothing(times)
        solve_step!(sys,U,Uold,Res,Jac,0.0,Inf; kwargs...)
    else
        tsol=TransientSolution([copy(Uold)],[times[1]])
        for itime=2:length(times)
            tstep=times[itime]-times[itime-1]
            Uold.=U
            solve_step!(sys,U,Uold,Res,Jac,times[itime],tstep; kwargs...)
            append!(tsol, times[itime], copy(U))
        end
        tsol
    end
end


