### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 784b4c3e-bb2a-4940-a83a-ed5e5898dfd4
begin
	using Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
	using Revise
	    using LinearAlgebra
    using CVFEMSystems
    using CVFEMSystems: femstiffness!, local_massmatrix
    using CVFEMSystems: randgrid, rectgrid, CVFEMSystem,solve,spacedim
    using CVFEMSystems: finitebell, d1finitebell, d2finitebell, ∇ηΛ∇, hminmax, ΛMatrix, ScalarTestData, paraprod,hdirichlet, hneumann,udirichlet
    using CVFEMSystems: coord, transmission, nnodes, nedges, nodevolume, edgenode, dirichlet!, minplot, fourplots, runconvergence
	using ExtendableGrids: simplexgrid, bfacemask!
	
	using GridVisualize: scalarplot, default_plotter!, scalarplot!, GridVisualizer, reveal, gridplot
	import CairoMakie, PlutoVista
	default_plotter!(CairoMakie)
	using PlutoUI
	using VoronoiFVM

end

# ╔═╡ 9d5c39da-4461-4296-80ff-31bd3206d847
md"""
# The XT system
"""

# ╔═╡ f62010ad-6531-470e-8cad-c4ee682a0eac
md"""
```math
\begin{align}
∂_t X - ∇ ⋅  \left(a(x) ∇ X\right) &=0\\
∂_t T - ∇ ⋅ \left(φ(T) ∇ T + T ∇ X\right) &=0
\end{align}
```
"""

# ╔═╡ ce013ee5-4b2b-4982-b285-7c28c7b160b6
begin
	const iX=1
	const iT=2
	const Γ_in=1
	const Γ_out=2
end;

# ╔═╡ cb601c88-b3f2-47ef-9c20-81c0ecd8a207
@kwdef struct XTData{Ta, Tφ}
	a::Ta=x->1
	φ::Tφ=T->1
	Λ::Matrix{Float64}=[1 0; 0 1.0]
end

# ╔═╡ c72ee588-f1f2-46a0-a49f-845d108752c8
function celleval(y, u, celldata, userdata)
    (;Λ,a,φ)=userdata
    (;uold,tstep)=celldata
    ω = nodevolume(celldata)
	aavg=0.0
	φavg=zero(eltype(u))
    for il = 1:nnodes(celldata)
        y[iX,il] += ((u[iX,il] - uold[iX,il])/tstep) * ω
        y[iT,il] += ((u[iT,il] - uold[iT,il])/tstep) * ω
		aavg+=a(coord(celldata, il))
		φavg+=φ(u[iT,il])
    end
	aavg/=nnodes(celldata)
	φavg/=nnodes(celldata)
    ΛKL = transmission(celldata, Λ)
    for ie = 1:nedges(celldata)
        i1 = edgenode(celldata, 1, ie)
        i2 = edgenode(celldata, 2, ie)
	
		dX= ΛKL[ie] * (u[iX,i1] - u[iX,i2])
        gX = aavg * dX
        y[iX,i1] += gX
        y[iX,i2] -= gX

	    Tup=  dX > 0 ? u[iT,i1] : u[iT,i2] 
		gT= ΛKL[ie] * φavg * (u[iT,i1] - u[iT,i2]) + Tup * dX
        y[iT,i1] += gT
        y[iT,i2] -= gT
    end
end

# ╔═╡ 71a22381-b926-474d-91a5-fbcca08a3e1f
function bnodeeval(y,u,bnodedata, userdata)
	(;region)= bnodedata
	if region==Γ_in
		dirichlet!(bnodedata,y,u, 1.0; ispec=iX)
		dirichlet!(bnodedata,y,u, 1.0; ispec=iT)
	end
	if region==Γ_out
		dirichlet!(bnodedata,y,u, 0.0; ispec=iX)
		dirichlet!(bnodedata,y,u, 0.0; ispec=iT)
	end
end

# ╔═╡ 431df177-6984-4132-91c8-11915246867d
φ(T)=0.01+0.01*T^2

# ╔═╡ 307164ca-3dbe-48a5-9bab-2bb14adea8a5
ax(x)=1

# ╔═╡ e6400a50-ae58-4e5c-ac32-32f4b00903d5
data1d=XTData(;Λ=ones(1,1), φ, a=ax)

# ╔═╡ 69f8af2f-af08-41c0-beae-b60b9737aa92
grid1d=simplexgrid(range(0,1,length=201))

# ╔═╡ cfb631fb-7aa8-4e09-9d0b-63a5ffa58bd7
sys1d=CVFEMSystem(grid1d,celleval,bnodeeval,data1d,2)

# ╔═╡ 7ff179f2-301c-4ebb-b1bc-d5d9a34e1ff9
solve(sys1d)

# ╔═╡ 83659238-ec5b-4210-95ce-0fd8ca7ea61f
function flux(y,u,edge,data)
		aavg=1.0
		φavg=0.5*(φ(u[iT,1])+φ(u[iT,2]))
		dX= (u[iX,1] - u[iX,2])
        gX = aavg * dX
        y[iX] += gX

	    Tup=  dX > 0 ? u[iT,1] : u[iT,2] 
		gT= φavg * (u[iT,1] - u[iT,2]) + Tup * dX
        y[iT] += gT
end

# ╔═╡ 0775fe8e-d08c-4907-91fa-a1998808f455
storage(y,u,node,data)= y.=u

# ╔═╡ 0e33ea6f-602d-4ccb-bfca-c07e76013733
function bcondition(y,u,bnode,data)
	boundary_dirichlet!(y,u,bnode;species=iX,region=Γ_in,value=1)
	boundary_dirichlet!(y,u,bnode;species=iT,region=Γ_in,value=1)
	boundary_dirichlet!(y,u,bnode;species=iX,region=Γ_out,value=0)
	boundary_dirichlet!(y,u,bnode;species=iT,region=Γ_out,value=0)
end

# ╔═╡ af64f64f-3f8d-4e4f-b07f-c95517b3465c
vfvmsys1d=VoronoiFVM.System(grid1d; flux,storage,bcondition, species=[iX,iT])

# ╔═╡ 90447221-fddb-4749-9453-0751f96825d0
Δt=0.01

# ╔═╡ 04764f80-cf1e-488c-b278-eddd7b4ca32e
tsol1d=solve(sys1d; times=0:Δt:1 )

# ╔═╡ 2f0c16bf-1f89-4947-aa89-e5ba35479a95
begin
	control=SolverControl()
	VoronoiFVM.fixed_timesteps!(control, Δt)
	vtsol1d=solve(vfvmsys1d; times=(0,1), control)
end

# ╔═╡ cb88bf20-48f6-4c45-9009-45be09ee4e68
vtsol1d.t≈tsol1d.t

# ╔═╡ 08c64a57-3514-49eb-95d9-18296881273a
@bind it1d  PlutoUI.Slider(1:length(tsol1d.t),show_value=true, default=10)

# ╔═╡ 1058d158-5684-484b-b4c0-6fa857500395
let
	vis=GridVisualizer(size=(700,300),legend=:rt,flimits=(-0.1,2), layout=(2,1))
	sol=tsol1d[it1d]
    scalarplot!(vis[1,1],grid1d, sol[iX,:], color=:green, label="X", title="CVFEMSystems")
    scalarplot!(vis[1,1],grid1d, sol[iT,:], color=:red, label="T", clear=false)
	vsol=vtsol1d[it1d]
    scalarplot!(vis[2,1],grid1d, vsol[iX,:], color=:green, label="X", title="VoronoiFVM, diff=$(norm(vsol-sol,Inf))")
    scalarplot!(vis[2,1],grid1d, vsol[iT,:], color=:red, label="T", clear=false)
		reveal(vis)
end

# ╔═╡ 9fc8702c-1fe7-404b-a61d-39d23eaf6c78
begin
	grid2d=simplexgrid(range(0,1,length=201), (0:0.05:0.2))
	bfacemask!(grid2d,(0,0), (1,0), 3)
    bfacemask!(grid2d,(0,0.2), (1,0.2), 3)
	bfacemask!(grid2d,(0,0), (0,1), Γ_in)
	bfacemask!(grid2d,(1,0), (1,1), Γ_out)
end

# ╔═╡ f1cfa9f7-542c-43bc-9a35-3e67e23e4833
gridplot(grid2d, size=(700,300), linewidth=0.1)

# ╔═╡ d6eec6ca-a5aa-4edc-8f7b-1a9797e816a3
data2d=XTData(Λ=Matrix(Diagonal(ones(2))), φ=T->0.01)

# ╔═╡ f30503ea-6262-423c-870a-e6c2afed56e4
sys2d=CVFEMSystem(grid2d,celleval,bnodeeval,data2d,2)

# ╔═╡ 43d74112-72eb-4f28-91ca-6fb981b19d8b
sol2d=solve(sys2d)

# ╔═╡ 87b03221-ce70-43d3-803e-f12a6fd7b281
tsol2d=solve(sys2d; times=0:0.01:1 )

# ╔═╡ 00c3db40-11f4-4f4a-ba98-ecc949d02448
@bind it2d  PlutoUI.Slider(1:length(tsol2d.t),show_value=true)

# ╔═╡ e6afb952-be58-4113-bba5-fba39747d7a9
let
	vis=GridVisualizer(size=(700,200),layout=(1,2),legend=:rt,flimits=(-0.1,2))
	sol=tsol2d[it2d]
    scalarplot!(vis[1,1],grid2d, sol[iX,:], title="X")
    scalarplot!(vis[1,2],grid2d, sol[iT,:], title="T")
	reveal(vis)
end

# ╔═╡ 1c63e831-ff08-46db-874e-6b2214b7a1c5
TableOfContents()

# ╔═╡ Cell order:
# ╠═784b4c3e-bb2a-4940-a83a-ed5e5898dfd4
# ╟─9d5c39da-4461-4296-80ff-31bd3206d847
# ╟─f62010ad-6531-470e-8cad-c4ee682a0eac
# ╠═ce013ee5-4b2b-4982-b285-7c28c7b160b6
# ╠═cb601c88-b3f2-47ef-9c20-81c0ecd8a207
# ╠═c72ee588-f1f2-46a0-a49f-845d108752c8
# ╠═71a22381-b926-474d-91a5-fbcca08a3e1f
# ╠═431df177-6984-4132-91c8-11915246867d
# ╠═307164ca-3dbe-48a5-9bab-2bb14adea8a5
# ╠═e6400a50-ae58-4e5c-ac32-32f4b00903d5
# ╠═69f8af2f-af08-41c0-beae-b60b9737aa92
# ╠═cfb631fb-7aa8-4e09-9d0b-63a5ffa58bd7
# ╠═7ff179f2-301c-4ebb-b1bc-d5d9a34e1ff9
# ╠═83659238-ec5b-4210-95ce-0fd8ca7ea61f
# ╠═0775fe8e-d08c-4907-91fa-a1998808f455
# ╠═0e33ea6f-602d-4ccb-bfca-c07e76013733
# ╠═af64f64f-3f8d-4e4f-b07f-c95517b3465c
# ╠═90447221-fddb-4749-9453-0751f96825d0
# ╠═04764f80-cf1e-488c-b278-eddd7b4ca32e
# ╠═2f0c16bf-1f89-4947-aa89-e5ba35479a95
# ╠═cb88bf20-48f6-4c45-9009-45be09ee4e68
# ╠═08c64a57-3514-49eb-95d9-18296881273a
# ╠═1058d158-5684-484b-b4c0-6fa857500395
# ╠═9fc8702c-1fe7-404b-a61d-39d23eaf6c78
# ╠═f1cfa9f7-542c-43bc-9a35-3e67e23e4833
# ╠═d6eec6ca-a5aa-4edc-8f7b-1a9797e816a3
# ╠═f30503ea-6262-423c-870a-e6c2afed56e4
# ╠═43d74112-72eb-4f28-91ca-6fb981b19d8b
# ╠═87b03221-ce70-43d3-803e-f12a6fd7b281
# ╠═00c3db40-11f4-4f4a-ba98-ecc949d02448
# ╠═e6afb952-be58-4113-bba5-fba39747d7a9
# ╠═1c63e831-ff08-46db-874e-6b2214b7a1c5
