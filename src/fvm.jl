"""
    femfactors!(T,G,Λ,enodes)

Calculate edge transmission coefficients for CVFE into T

- `G` is a `(dim+1) x dim` matrix
- `Λ` is a `dim x dim` matrix
- `T` is a `nedges` vector
- `enodes` is a `2 x nedges` edge ajacency matrix
"""
function femfactors!(T, G, Λ, enodes)
    ne = length(T)
    spacedim = size(Λ, 1)
    for ie = 1:ne
        i1 = enodes[1, ie]
        i2 = enodes[2, ie]
        x = 0.0
        for k = 1:spacedim
            for l = 1:spacedim
                x += G[i1, k] * Λ[k, l] * G[i2, l]
            end
        end
        T[ie] = -x
    end
    T
end

function fvmassemble!(A_h, # Global stiffness matrix
                      F_h, # Right hand side of FVM problem
                      coord,
                      cellnodes,
                      bfacenodes,
                      Λ,
                      f::Tf,
                      β::Tβ, # Boundary function
                      ::Type{Val{spacedim}},
                      ::Type{Val{nedges}}) where {Tf, Tβ, spacedim, nedges}
    celldim = spacedim + 1
    enodes = @MMatrix zeros(Int, 2, nedges)
    enodes .= local_edgenodes[spacedim]
    E = @MVector zeros(nedges) # local stiffness matrix
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
        femfactors!(E, G, Λ, enodes)
        for ie = 1:nedges
            i1 = cellnodes[enodes[1, ie], icell]
            i2 = cellnodes[enodes[2, ie], icell]
            efac = E[ie] * vol
            A_h[i1, i1] += efac
            A_h[i1, i2] -= efac
            A_h[i2, i1] -= efac
            A_h[i2, i2] += efac
        end
        for il = 1:celldim
            ig = cellnodes[il, icell]
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
    fvmassemble!(A_h,F_h,grid,Λ,f,β)

Assemble CVFE matrix and right hand side for the `-∇Λ∇ u = f` with  dirichlet
boundary conditions  `u=β`.

- A_h: `nnodes x nnodes` AbstractMatrix
- F_h: Vector of length `nnodes`
- grid: ExtendableGrid instance
- Λ: `dim x dim` matrix
- f(X): scalar function on `dim´-vectors
- β(X): scalar function on `dim´-vectors
"""
function fvmassemble!(A_h, # Global stiffness matrix
                      F_h, # Right hand side of FVM problem
                      grid::ExtendableGrid,
                      Λ,
                      f::Tf,
                      β::Tβ) where {Tf, Tβ}
    coord = grid[Coordinates]
    cellnodes = grid[CellNodes]
    bfacenodes = grid[BFaceNodes]
    fvmassemble!(A_h, F_h, coord, cellnodes, bfacenodes, Λ, f, β, Val{dim_space(grid)},
                 Val{size(local_edgenodes[dim_space(grid)], 2)})
end

"""
    fvmsolve (grid,Λ,f,β)

Solve `-∇Λ∇ u = f` with  dirichlet boundary conditions  `u=β`. 
Uses CVFEM and AMG preconditioned CG as solver.

- grid: ExtendableGrid instance
- Λ: `dim x dim` matrix
- f(X): scalar function on `dim´-vectors
- β(X): scalar function on `dim´-vectors
"""
function fvmsolve(grid, Λ, f, β)
    # Initialize sparse matrix and right hand side
    n = num_nodes(grid)
    matrix = ExtendableSparseMatrix(n, n)
    rhs = zeros(n)
    # Call the assemble function.

    fvmassemble!(matrix, rhs, grid, Λ, f, β)
    precon = aspreconditioner(smoothed_aggregation(SparseMatrixCSC(matrix)))
    u0 = zeros(n)
    u, stats = cg(matrix, rhs, u0; M = precon, ldiv = true)
    u
end
