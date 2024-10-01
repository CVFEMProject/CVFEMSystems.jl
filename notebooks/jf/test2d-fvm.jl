### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 784b4c3e-bb2a-4940-a83a-ed5e5898dfd4
begin
	using Pkg

    # Activate the project environment
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
	using Revise
	    using LinearAlgebra
    using AnisotropicFVMProject
    using AnisotropicFVMProject: femstiffness!, local_massmatrix
    using AnisotropicFVMProject: randgrid, rectgrid, CVFEMSystem,solve,spacedim
    using AnisotropicFVMProject: finitebell, d1finitebell, d2finitebell, ∇ηΛ∇, hminmax, ΛMatrix, ScalarTestData, paraprod,hdirichlet, hneumann,udirichlet
    using AnisotropicFVMProject: coord, transmission, nnodes, nedges, volume, edgenode, dirichlet!, minplot, fourplots, runconvergence
	using StaticArrays: @MMatrix
	using GridVisualize: scalarplot, default_plotter!
	import CairoMakie, PlutoVista
	default_plotter!(CairoMakie)

end

# ╔═╡ c72ee588-f1f2-46a0-a49f-845d108752c8
function celleval(y, u, celldata, userdata)
    (;Λ,f)=userdata
    (;uold,tstep)=celldata
    ω = volume(celldata) / nnodes(celldata)
    for il = 1:nnodes(celldata)
        y[1,il] = ((u[1,il] - uold[1,il])/tstep - f(coord(celldata, il))) * ω
    end
    ΛKL = transmission(celldata, Λ)
    for ie = 1:nedges(celldata)
        i1 = edgenode(celldata, 1, ie)
        i2 = edgenode(celldata, 2, ie)
        g =  ΛKL[ie] * (u[1,i1] - u[1,i2])
        y[1,i1] += g
        y[1,i2] -= g
    end
end

# ╔═╡ e6400a50-ae58-4e5c-ac32-32f4b00903d5
data=ScalarTestData(Λ=ΛMatrix(100,0*π/4), u=X->paraprod(X))

# ╔═╡ 69f8af2f-af08-41c0-beae-b60b9737aa92
grid=randgrid(2,10000; X=(-1,1))

# ╔═╡ cfb631fb-7aa8-4e09-9d0b-63a5ffa58bd7
sys=CVFEMSystem(grid,celleval,hdirichlet,data,1)

# ╔═╡ 43d74112-72eb-4f28-91ca-6fb981b19d8b
sol=solve(sys)

# ╔═╡ 5acf6a5e-88ff-4831-9446-312aeacd70b2
scalarplot(grid,sol[1,:])

# ╔═╡ e9426664-1e35-4b5f-b522-3a898b1f4d6e
runconvergence(1:7,2,randgrid; celleval, data)

# ╔═╡ f06e702d-c4b6-4d2c-b965-d1a50fe99e29
tgrid=randgrid(2,10000; X=(-5,5))

# ╔═╡ 15bea5a1-b931-4e23-80a5-de2a55699f65
tdata=ScalarTestData(Λ=ΛMatrix(100,π/4), f=X->0)

# ╔═╡ 537bcfe6-7c40-41fe-a8a8-06f49f4cf65e
tsys=CVFEMSystem(tgrid,celleval,hneumann,tdata,1)

# ╔═╡ dfacf3f5-49d8-4f9c-acc0-c29f2573e2b1
tsol=solve(tsys; inival=finitebell, times=0:0.001:0.1)

# ╔═╡ 21db39f1-e617-4aa7-a33f-726c56067d0b
minplot(tsol)

# ╔═╡ 6c905898-aa4b-4c65-8af5-e726378a3405
fourplots(tgrid,tsol)

# ╔═╡ Cell order:
# ╠═784b4c3e-bb2a-4940-a83a-ed5e5898dfd4
# ╠═c72ee588-f1f2-46a0-a49f-845d108752c8
# ╠═e6400a50-ae58-4e5c-ac32-32f4b00903d5
# ╠═69f8af2f-af08-41c0-beae-b60b9737aa92
# ╠═cfb631fb-7aa8-4e09-9d0b-63a5ffa58bd7
# ╠═43d74112-72eb-4f28-91ca-6fb981b19d8b
# ╠═5acf6a5e-88ff-4831-9446-312aeacd70b2
# ╠═e9426664-1e35-4b5f-b522-3a898b1f4d6e
# ╠═f06e702d-c4b6-4d2c-b965-d1a50fe99e29
# ╠═15bea5a1-b931-4e23-80a5-de2a55699f65
# ╠═537bcfe6-7c40-41fe-a8a8-06f49f4cf65e
# ╠═dfacf3f5-49d8-4f9c-acc0-c29f2573e2b1
# ╠═21db39f1-e617-4aa7-a33f-726c56067d0b
# ╠═6c905898-aa4b-4c65-8af5-e726378a3405
