
"""
    Const local edgenodes

Edge-node ajacency, taken from ExtendableGrids.jl.
Probably we can define this independently.
"""
const local_edgenodes = [
    local_celledgenodes(Edge1D),
    local_celledgenodes(Triangle2D),
    local_celledgenodes(Tetrahedron3D)
]

const local_massmatrix1d = [2.0 1.0; 1.0 2.0] / 6
const local_massmatrix2d = [2.0 1.0 1.0; 1.0 2.0 1.0; 1.0 1.0 2.0] / 12
const local_massmatrix3d = [2.0 1.0 1.0 1.0; 1.0 2.0 1.0 1.0; 1.0 1.0 2.0 1.0; 1.0 1.0 1.0 2.0] / 20

"""
    const local_massmatrix

Mass matrix templates for 1D/2D/3D
"""
const local_massmatrix = [local_massmatrix1d,
    local_massmatrix2d,
    local_massmatrix3d]

"""
    coordmatrix!(C,coord, cellnodes,icell)

Create reduced coordinates in `C`.
- `coord is the global  `dim x nnodes` matrix coordinate
- `cellnodes` is the `(dim+1) x ncells` connectivity matrix
- `C` is a `dim x dim` matrix
"""
function coordmatrix!(C, coord, cellnodes, icell)
    spacedim = size(coord, 1)
    celldim = size(cellnodes, 1)
    for jj = 1:spacedim
        for ii = 1:spacedim
            C[ii, jj] = coord[ii, cellnodes[jj, icell]] - coord[ii, cellnodes[celldim, icell]]
        end
    end
end

"""
    femgrad!(G,C)

Create shape function gradients from reduced coordinate matrix.
- `C` is a `dim x dim` matrix
- `G` is a `(dim+1) x dim` matrix
"""
function femgrad!(G::Tg, C::Tc) where {Tg, Tc}
    spacedim = size(C, 1)
    celldim = spacedim + 1
    G[1:spacedim, 1:spacedim] .= C \ I
    for i = 1:spacedim
        G[celldim, i] = 0.0
        for j = 1:spacedim
            G[celldim, i] -= G[j, i]
        end
    end
end

"""
    femstiffness!(S,G)

Create isotropic P1 FEM stiffness matrix `S` from shape function gradients `G`.
- `G` is a `(dim+1) x dim` matrix
- `S` is a `(dim+1) x (dim+1)` matrix
"""
function femstiffness!(S, G)
    @einsum S[il, jl] = G[il, k] * G[jl, k]
end

"""
    femstiffness!(S,G,Λ)

Create anisotropic P1 FEM stiffness matrix S from shape function gradients G
and permeability tensor Λ.
- `G` is a `(dim+1) x dim` matrix
- `S` is a `(dim+1) x (dim+1)` matrix
- `Λ` is a `dim x dim` matrix
"""
function femstiffness!(S, G, Λ)
    @einsum S[il, jl] = G[il, k] * Λ[k, m] * G[jl, m]
end

# Boundary face volume calculation by Heron's formula.
# We can do better here.

function dist2(coord, ig1, ig2)
    x = coord[1, ig1] - coord[1, ig2]
    y = coord[2, ig2] - coord[2, ig2]
    sqrt(x^2 + y^2)
end

function dist3(coord, ig1, ig2)
    x = coord[1, ig1] - coord[1, ig2]
    y = coord[2, ig1] - coord[2, ig2]
    z = coord[3, ig1] - coord[3, ig2]
    sqrt(x^2 + y^2 + z^2)
end

bfacevolume(coord, bfacenodes, ibface, ::Type{Val{1}}) = 1

bfacevolume(coord, bfacenodes, ibface, ::Type{Val{2}}) = dist2(coord, bfacenodes[1, ibface], bfacenodes[2, ibface])

function bfacevolume(coord, bfacenodes, ibface, ::Type{Val{3}})
    a = dist3(coord, bfacenodes[1, ibface], bfacenodes[2, ibface])
    b = dist3(coord, bfacenodes[1, ibface], bfacenodes[3, ibface])
    c = dist3(coord, bfacenodes[2, ibface], bfacenodes[3, ibface])
    s = 0.5 * (a + b + c)
    sqrt(s * (s - a) * (s - b) * (s - c))
end

"""
    bfacevolume(coord, bfacenodes, ibface)

Return volume (area, length) of boundary face. 
- `coord is the global  `dim x nnodes` matrix coordinate
- `bfacenodes` is the `dim x nbfaces` boundary connectivity matrix
"""
bfacevolume(coord, bfacenodes, ibface) = bfacevolume(coord, bfacenodes, ibface, Val{size(coord, 1)})

"""
    femnorms(coord,cellnodes,u)

Return L2 norm and H1 seminorm of P1 grid function u.
- `coord is the global  `dim x nnodes` matrix coordinate
- `cellnodes` is the `(dim+1) x ncells` connectivity matrix
"""
function femnorms(coord, cellnodes, u)
    l2norm = 0.0
    h1norm = 0.0
    spacedim = size(coord, 1)
    celldim = size(cellnodes, 1)
    S = zeros(celldim, celldim)  # local stiffness matrix
    C = zeros(spacedim, spacedim) # local coordinate matrix
    G = zeros(celldim, spacedim) # shape function gradients
    M = local_massmatrix[spacedim]
    ncells = size(cellnodes, 2)
    factdim = prod(1:spacedim)
    for icell = 1:ncells
        coordmatrix!(C, coord, cellnodes, icell)
        vol = abs(det(C)) / factdim
        femgrad!(G, C)
        femstiffness!(S, G)
        for il = 1:celldim
            for jl = 1:celldim
                uij = u[cellnodes[jl, icell]] * u[cellnodes[il, icell]] * vol
                l2norm += uij * M[jl, il]
                h1norm += uij * S[jl, il]
            end
        end
    end
    return (sqrt(l2norm), sqrt(abs(h1norm)))
end

"""
    femnorms(grid,u)

Return L2 norm and H1 seminorm of P1 grid function u.
"""
femnorms(g::ExtendableGrid, u) = femnorms(g[Coordinates], g[CellNodes], u)

"""
    Dirichlet()

Return  Dirichlet penalty constant.
"""
Dirichlet() = 1.0e30

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
