using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using LinearAlgebra: Diagonal
using AnisotropicFVMProject: ∇Λ∇, finitebell, randgrid, rectgrid, fvmsolve
using AnisotropicFVMProject: coord, transmission, nnodes, nedges, volume, edgenode, dirichlet!, CVFEMSystem, solve
using ExtendableGrids: dim_space, writeVTK

function nlfvmtest(; grid=randgrid(2,100000), tol = 1.0e-10)
    f(X) = -∇Λ∇(finitebell, X)
    β(X) = 0.0
    η(u) = 1 + u^2
    Λ = Diagonal(ones(dim_space(grid)))

    # Evaluate local residuum 
    function celleval!(y, u, celldata, userdata)
        y .= zero(eltype(y))
        ηavg = 0.0
        ω = volume(celldata) / nnodes(celldata)
        for il = 1:nnodes(celldata)
            y[il] -= f(coord(celldata, il)) * ω
            ηavg += η(u[il]) / nnodes(celldata)
        end
        ΛKL = transmission(celldata, Λ)
        for ie = 1:nedges(celldata)
            i1 = edgenode(celldata, 1, ie)
            i2 = edgenode(celldata, 2, ie)
            g = ηavg * ΛKL[ie] * (u[i1] - u[i2])
            y[i1] += g
            y[i2] -= g
        end
    end

    function bnodeeval!(y, u, bnodedata, userdata)
        dirichlet!(bnodedata, y, u, β(coord(bnodedata)))
    end

    sys=CVFEMSystem(grid,celleval!,bnodeeval!, nothing, 1)

    solution=solve(sys;tol)
    writeVTK("nlfvmtest.vtu",grid;solution)
end

nlfvmtest()
