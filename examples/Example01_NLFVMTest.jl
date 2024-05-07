module Example01_NLFVMTest

using LinearAlgebra: Diagonal
using AnisotropicFVMProject: ∇Λ∇, finitebell, randgrid, rectgrid, fvmsolve
using AnisotropicFVMProject: coord, transmission, nnodes, nedges, volume, edgenode, dirichlet!
using GridVisualize: GridVisualizer, gridplot!, scalarplot!, reveal

"""
    main(;dim=2,n=100,tol = 1.0e-8, Plotter=nothing, Λ = Diagonal(ones(dim)))

Test problem on cuboid dim-dimensional grid with homogeneous Dirichlet BC    
"""
function main(; dim = 2, n = 100, tol = 1.0e-8, Plotter = nothing, Λ = Diagonal(ones(dim)))
    grid = randgrid(dim, n)
    f(X) = -∇Λ∇(finitebell, X, Λ)
    β(X) = 0.0
    η(u) = 1 + u^2

    # Evaluate local residuum
    # y: vector of local resiudal values; result
    # u: vector of local unknown values; input
    # celldata: information about current cell
    # userdata: optional user data; currently ignored
    function celleval!(y, u, celldata, userdata)
        y .= zero(eltype(y))
        ηavg = 0.0
        ω = volume(celldata) / nnodes(celldata)

        # Assemble right hand side, calculate η average
        for il = 1:nnodes(celldata)
            y[il] -= f(coord(celldata, il)) * ω
            ηavg += η(u[il])
        end
        ηavg /= nnodes(celldata)

        # Anisotropic transmission coefficients
        ΛKL = transmission(celldata, Λ)

        # Calculate fluxes, update node values
        for ie = 1:nedges(celldata)
            i1 = edgenode(celldata, 1, ie)
            i2 = edgenode(celldata, 2, ie)
            g = ηavg * ΛKL[ie] * (u[i1] - u[i2])
            y[i1] += g
            y[i2] -= g
        end
    end

    # Evaluate boundary condition in given node
    function bnodeeval!(y, u, bnodedata, userdata)
        dirichlet!(bnodedata, y, u, β(coord(bnodedata)))
    end

    # Solve problem
    sol = fvmsolve(grid, celleval!, bnodeeval!; tol)

    # Visualize grid and solution
    vis = GridVisualizer(; Plotter, size = (600, 300), layout = (1, 2))
    gridplot!(vis[1, 1], grid)
    scalarplot!(vis[1, 2], grid, sol)
    reveal(vis)

    # Return solution
    sol
end

end
