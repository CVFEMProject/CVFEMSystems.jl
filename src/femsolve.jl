
function femassemble!(A_h,
                      F_h,
                      coord,
                      cellnodes,
                      bfacenodes,
                      Λ,
                      f::Tf,
                      β::Tβ,
                      ::Type{Val{spacedim}}) where {Tf, Tβ, spacedim}
    celldim = spacedim + 1
    S = @MMatrix zeros(celldim, celldim) # local stiffness matrix
    X = @MMatrix zeros(celldim, celldim) # local stiffness matrix
    C = @MMatrix zeros(spacedim, spacedim)# local coordinate matrix
    G = @MMatrix zeros(celldim, spacedim)  # shape function grdients
    BC = @MMatrix zeros(spacedim, spacedim)     # local boundary coordinate matrix
    ncells = size(cellnodes, 2)
    factdim = prod(1:spacedim)
    xcoord = reinterpret(reshape, SVector{spacedim, Float64}, coord)
    for icell = 1:ncells
        coordmatrix!(C, coord, cellnodes, icell)
        vol = abs(det(C)) / factdim
        femgrad!(G, C)
        femstiffness!(S, G, Λ)
        for il = 1:celldim
            ig = cellnodes[il, icell]
            for jl = 1:celldim
                jg = cellnodes[jl, icell]
                A_h[ig, jg] += S[il, jl] * vol
            end
            F_h[ig] += f(xcoord[ig]) * vol / celldim
        end
    end

    # Boundary part with penalty method
    nbfaces = size(bfacenodes, 2)
    for ibface = 1:nbfaces
        for idim = 1:spacedim
            i1 = bfacenodes[idim, ibface]
            A_h[i1, i1] += Dirichlet()
            F_h[i1] += Dirichlet() * β(xcoord[i1])
        end
    end
end

"""
    femassemble!(A_h,F_h,grid,Λ,f,β)

Assemble P1 FEM matrix and right hand side for the `-∇Λ∇ u = f` with  dirichlet
boundary conditions  `u=β`.

- A_h: `nnodes x nnodes` AbstractMatrix
- F_h: Vector of length `nnodes`
- grid: ExtendableGrid instance
- Λ: `dim x dim` matrix
- f(X): scalar function on `dim´-vectors
- β(X): scalar function on `dim´-vectors
"""
function femassemble!(A_h,
                      F_h,
                      grid::ExtendableGrid,
                      Λ,
                      f::Tf,
                      β::Tβ) where {Tf, Tβ}
    coord = grid[Coordinates]
    cellnodes = grid[CellNodes]
    bfacenodes = grid[BFaceNodes]
    femassemble!(A_h, F_h, coord, cellnodes, bfacenodes, Λ, f, β, Val{dim_space(grid)})
end

"""
    femsolve (grid,Λ,f,β)

Solve `-∇Λ∇ u = f` with  dirichlet boundary conditions  `u=β`. 
Uses P1 FEM and AMG preconditioned CG as solver.

- grid: ExtendableGrid instance
- Λ: `dim x dim` matrix
- f(X): scalar function on `dim´-vectors
- β(X): scalar function on `dim´-vectors
"""
function femsolve(grid, Λ, f, β)
    # Initialize sparse matrix and right hand side
    n = num_nodes(grid)
    matrix = ExtendableSparseMatrix(n, n)
    rhs = zeros(n)
    # Call the assemble function.

    femassemble!(matrix, rhs, grid, Λ, f, β)
    precon = aspreconditioner(smoothed_aggregation(SparseMatrixCSC(matrix)))
    u0 = zeros(n)
    u, stats = cg(matrix, rhs, u0; M = precon, ldiv = true)
    u
end
