"""
    struct CellData

Data structure holding information for user cell assembly.
"""
mutable struct CellData{Tgc, Tgn, Ten, Tcm, Tsg, Ttm}
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
end

"""
    CellData(coord,edgenodes)

Constructor for cell data from coordinate array and nodes per edge matrix.
"""
function CellData(coord, edgenodes)
    spacedim = size(coord, 1)
    globalcoord = reinterpret(reshape, SVector{spacedim, Float64}, coord)
    celldim = spacedim + 1
    icell = 0
    region = 0
    xedgenodes = @MMatrix zeros(Int64, 2, size(edgenodes, 2))
    xedgenodes .= edgenodes
    globalnodes = @MVector zeros(Int64, celldim)
    factdim = prod(1:spacedim)
    coordmatrix = @MMatrix zeros(spacedim, spacedim)
    shapegradients = @MMatrix zeros(celldim, spacedim)
    transmission = @MVector zeros(size(edgenodes, 2)) # Transmission coefficients
    CellData(icell,
             region,
             globalcoord,
             globalnodes,
             xedgenodes,
             factdim,
             coordmatrix,
             shapegradients,
             0.0,
             transmission)
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
    fvmsolve(grid,celleval, bnodeval; userdata, tol, maxiter)

Solve finite volume discretization described by celleval! and bnodeeval! using
Newton's method with automatic differentiation and AMG preconditioned bicgstab solver.
"""
function fvmsolve(grid::ExtendableGrid,
                  celleval!::Tc,
                  bnodeeval!::Tb;
                  userdata = nothing,
                  tol = 1.0e-10,
                  maxiter = 10) where {Tc, Tb}
    coordinates = grid[Coordinates]
    cellnodes = grid[CellNodes]
    bfacenodes = grid[BFaceNodes]

    N = size(coordinates, 2)
    Jac = ExtendableSparseMatrix(N, N)
    Res = zeros(N)
    U = zeros(N)
    update = zeros(N)

    spacedim = size(coordinates, 1)
    celldim = spacedim + 1
    celldata = CellData(coordinates, local_edgenodes[spacedim])
    ncells = size(cellnodes, 2)

    bnodedata = BNodeData(coordinates)
    precon = nothing
    # Define wrapper function with closure in order
    # to fit the format uses by ForwardDiff
    wrap_celleval!(y, u) = celleval!(y, u, celldata, userdata)
    wrap_bnodeeval!(y, u) = bnodeeval!(y, u, bnodedata, userdata)

    ulocal = zeros(celldim)
    ylocal = zeros(celldim)
    result = DiffResults.JacobianResult(ylocal, ulocal)
    config = ForwardDiff.JacobianConfig(wrap_celleval!, ylocal, ulocal)

    nbfaces = size(bfacenodes, 2)

    # Evaluate at boundary
    xcoord = reinterpret(reshape, SVector{spacedim, Float64}, coordinates)

    ublocal = zeros(1)
    yblocal = zeros(1)
    bresult = DiffResults.JacobianResult(yblocal, ublocal)
    bconfig = ForwardDiff.JacobianConfig(wrap_bnodeeval!, yblocal, ublocal)

    it = 1
    nalloc = 0
    while it < maxiter
        # Set jacobian and residual to 0
        nonzeros(Jac) .= 0.0
        Res .= 0.0
        # Evaluate residual and assemble Jacobian
        for icell = 1:ncells
            update!(celldata, coordinates, cellnodes, icell)
            # Copy solution to ulocal
            for il = 1:celldim
                ulocal[il] = U[cellnodes[il, icell]]
            end

            # AD residual and jacobian 
            ForwardDiff.vector_mode_jacobian!(result, wrap_celleval!, ylocal, ulocal, config)

            res = DiffResults.value(result)
            jac = DiffResults.jacobian(result)

            # Assemble into global data
            for il = 1:celldim
                ig = cellnodes[il, icell]
                Res[ig] += res[il]
                for jl = 1:celldim
                    jg = cellnodes[jl, icell]
                    Jac[ig, jg] += jac[il, jl]
                end
            end
        end # icell

        for ibface = 1:nbfaces
            vol = bfacevolume(coordinates, bfacenodes, ibface)
            for il = 1:spacedim
                update!(bnodedata, coordinates, bfacenodes, ibface, il, vol)
                ig = bfacenodes[il, ibface]
                ublocal[1] = U[ig]
                # AD residual and jacobian 
                ForwardDiff.vector_mode_jacobian!(bresult, wrap_bnodeeval!, yblocal, ublocal, bconfig)
                bres = DiffResults.value(bresult)
                bjac = DiffResults.jacobian(bresult)
                Res[ig] += bres[1]
                Jac[ig, ig] += bjac[1, 1]
            end
        end

        # Solve residual system
        #solver=AMGSolver(Jac, param= (solver=(type="bicgstab",),))
        #update=solver\Res
        if isnothing(precon)
            precon = aspreconditioner(smoothed_aggregation(SparseMatrixCSC(Jac)))
            #precon=AMGPreconditioner(Jac)
        end
        update, stats = bicgstab(Jac, Res, update; M = precon, ldiv = true)
        #update=Jac\Res
        # update solution
        U .-= update

        nm = norm(update, Inf)
        @info it, nm
        if nm < tol
            if nalloc > 0
                @warn "allocations in assembly loop"
            end
            return U
        end
        it += 1
    end  # while it
    error("no convergence after maxiter=$(maxiter) iterations")
end
