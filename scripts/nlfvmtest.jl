using Pkg
Pkg.activate(joinpath(@__DIR__,".."))

using LinearAlgebra: Diagonal
using AnisotropicFVMProject: ∇Λ∇, finitebell, randgrid,rectgrid,fvmsolve, celldim,coord,transmission,nedges
using ExtendableGrids: dim_space

function nlfvmtest(grid;tol=1.0e-10)
    f(X)=-∇Λ∇(finitebell,X);
    β(X)=0.0
    η(u)=1+u^2
    Λ=Diagonal(ones(dim_space(grid)));
    
    # Evaluate local residuum 
    function celleval!(y,u,celldata, userdata)
        y.=zero(eltype(y))
        ηavg=0.0
        for il=1:celldim(celldata)
    	    y[il]-=f(coord(celldata,il))*celldata.vol/celldim(celldata)
            ηavg+=η(u[il])/celldim(celldata)
        end
        ΛKL=transmission(celldata,Λ)
        for ie=1:nedges(celldata)
            i1=celldata.edgenodes[1,ie]
	    i2=celldata.edgenodes[2,ie]
            g=ηavg*ΛKL[ie]*(u[i1]-u[i2])
            y[i1]+=g
            y[i2]-=g
        end
    end
    fvmsolve(grid, celleval!,nothing,β;tol)
end
