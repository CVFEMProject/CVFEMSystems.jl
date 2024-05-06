### A Pluto.jl notebook ###
# v0.19.41

using Markdown
using InteractiveUtils

# ╔═╡ 784b4c3e-bb2a-4940-a83a-ed5e5898dfd4
begin
    using Pkg
	
    Pkg.activate(joinpath(@__DIR__,".."))
    using Revise
	using LinearAlgebra
    using AnisotropicFVMProject
    using AnisotropicFVMProject: randgrid,rectgrid
	using AnisotropicFVMProject: finitebell, d1finitebell,d2finitebell, ∇Λ∇, hminmax,ΛMatrix
    using SimplexGridFactory,ExtendableGrids
   using StaticArrays 
	   import Triangulate, TetGen, CairoMakie, PlutoVista
    using GridVisualize
	default_plotter!(CairoMakie)
	import PlutoUI
end

# ╔═╡ 3866057c-8a2b-481b-8399-6168e7a9f20b
PlutoUI.TableOfContents()

# ╔═╡ 2471465b-64b7-4f25-97b0-75957940d68d
md"""
# Finite element tests
"""

# ╔═╡ 10e998d3-1bb3-4591-b340-f9b93e027738
md"""
## Finitebell
"""

# ╔═╡ 64a7b893-8add-4c52-9725-c8b413c59e40
X=range(-2,2,length=1000)

# ╔═╡ 53cf08d6-f3d8-4a39-ad22-b395da8cda6a
let
	vis=GridVisualizer(size=(600,200), legend=:rt)
	scalarplot!(vis, X, finitebell.(X),color=:red, label="finitebell")
	scalarplot!(vis, X, d1finitebell.(X),color=:green, label="d1finitebell",clear=false)
	scalarplot!(vis, X, d2finitebell.(X),color=:blue, label="d2finitebell",clear=false)

	reveal(vis)
end

# ╔═╡ 5aa8a153-1ffa-4820-bcd2-83393170636a
md"""
## Grids 
"""

# ╔═╡ e2747647-ab45-4424-bb6f-89d9e965bad0
md"""
### Grid1d
"""

# ╔═╡ 4492bf1e-117b-4c26-bc6a-44b252332aeb
grid1d=randgrid(1,100)


# ╔═╡ f558c158-406e-4612-91d6-bbb52f29c3af
gridplot(grid1d,Plotter=CairoMakie,size=(600,100))

# ╔═╡ 2b695f50-07a6-4655-8876-5c5abf13af35
scalarplot(grid1d,map(finitebell, grid1d), size=(600,200))

# ╔═╡ 817b63cc-4c5a-446d-b22b-d52fc91ee35c
scalarplot(grid1d,map(x->∇Λ∇(finitebell,x), grid1d), size=(600,200))

# ╔═╡ f5bad455-7b9a-4473-8b39-80aa19e17514
md"""
### Grid2D
"""

# ╔═╡ 53908721-deec-4d31-a121-5075fc03f3e3
grid2d=rectgrid(2,160)

# ╔═╡ 99705ed4-546a-4739-afc7-91120f33723e
gridplot(grid2d,Plotter=CairoMakie, size=(400,300))

# ╔═╡ 76c28a15-4135-4b6e-9f6a-cd2d54d93eaf
scalarplot(grid2d,map(finitebell, grid2d), size=(400,300))

# ╔═╡ 0f0becb1-6588-4e67-9d6c-bf9458ace643
scalarplot(grid2d,map((x,y)->∇Λ∇(finitebell,(x,y)), grid2d), 
	size=(400,300),limits=(-10,10))

# ╔═╡ 299ed24c-c5d6-4c49-b407-cda278173477
md"""
### Grid3d
"""

# ╔═╡ 4ceb53f9-7a2e-4ccb-a2fc-9fb45848fb85
grid3d=randgrid(3,10000)

# ╔═╡ 0aeedbf2-de4b-4845-bf85-3e831e48c14b
gridplot(grid3d, Plotter=PlutoVista, xplanes=[0.0])

# ╔═╡ 8bac06c2-1649-4060-8335-691518e8805c
scalarplot(grid3d,map(finitebell, grid3d), size=(400,400), Plotter=PlutoVista)

# ╔═╡ 4f2ceb25-176f-4952-b2b8-fb7ba2f7ed90
scalarplot(grid3d,map((x,y,z)->∇Λ∇(finitebell,(x,y,z)), grid3d), 
	size=(400,400),limits=(-10,10), Plotter=PlutoVista)

# ╔═╡ bb44eba7-ca38-4280-928e-d61f55691539
md"""
## FEM solutions
"""

# ╔═╡ ea5ffa60-e0f1-4f9e-891a-6bf4aeffe46e
β(X)=0.0

# ╔═╡ 7fffecda-956c-43f7-a3b4-d78396def0c7
md"""
### 1D
"""

# ╔═╡ a1c9cef9-bd7d-40d5-be50-05c9355b2f2b
f1d(X)=-∇Λ∇(finitebell,X)

# ╔═╡ 41b0e5b7-590c-431e-99ff-1247aeac24a7
sol1d=femsolve(grid1d, [1.0;;], f1d, β)

# ╔═╡ d9738a22-db04-47a2-bded-c4de3417cab4
function plot1dresults(grid,sol)
	vis=GridVisualizer(size=(600,200))
	scalarplot!(vis,grid1d,sol1d,color=:red)
	scalarplot!(vis,grid1d,finitebell,color=:green,clear=false)
	reveal(vis)
end

# ╔═╡ 510addc3-deec-45a5-b15b-1587d985e227
plot1dresults(grid1d,sol1d)

# ╔═╡ 00349f7c-6010-4def-bd9d-f285ab695f2d
function runconvergence(Λ,ref,gengrid)
	dim=size(Λ,1)
	h1norms=[]
	l2norms=[]
	h=[]
	for n in ref
		grid=gengrid(dim,n)
		hmin,hmax=hminmax(grid)
		push!(h,hmax)
		u=map(finitebell,grid)
		sol=femsolve(grid,Λ,X->-∇Λ∇(finitebell,X,Λ),X->0.0)
		l2,h1=femnorms(grid,u-sol)
		@info num_nodes(grid), l2, h1
		push!(h1norms,h1)
		push!(l2norms,l2)
	end
    
	vis=GridVisualizer(size=(600,300),xscale=:log,yscale=:log, legend=:lt)
	scalarplot!(vis,h,l2norms, color=:red, label="l2")
	scalarplot!(vis,h,h.^2, color=:red, label="O(h^2)",linestyle=:dot,clear=false)
	scalarplot!(vis,h,h1norms, color=:blue, label="h1",linestyle=:solid,clear=false)
	scalarplot!(vis,h,h, color=:blue, label="O(h)",linestyle=:dot,clear=false)
	reveal(vis)
end

# ╔═╡ 59fcedaa-a78f-43a9-8dda-6ebfd67805bf
runconvergence([1.0;;],[10*2^k for k=1:10],rectgrid)

# ╔═╡ 2567c910-c04a-4d7b-be97-1a0013b1913a
md"""
### 2D isotropic
"""

# ╔═╡ 5084ef28-0194-4050-a24a-b60139b41000
const Λ2d=Diagonal(ones(2))

# ╔═╡ f19f0fa6-956b-4c5d-8842-4a8492890017
f2d(X)=-∇Λ∇(finitebell,X,Λ2d)

# ╔═╡ d11d26f8-8273-4321-a8bb-f4d46307d895
sol2d=femsolve(grid2d, Λ2d, f2d, β)

# ╔═╡ a8297522-61fb-4a59-ac18-9a113698c649
function plot2dresults(grid2d,sol2d)
	vis=GridVisualizer(size=(600,200),layout=(1,2))
	scalarplot!(vis[1,1],grid2d,sol2d, title="approx")
	scalarplot!(vis[1,2],grid2d,map(finitebell,grid2d), title="exact")
	reveal(vis)
end


# ╔═╡ 348126b6-76d7-4f97-a867-83c684e05c67
plot2dresults(grid2d,sol2d)

# ╔═╡ 48997cd2-4d81-49e4-aea9-d7e18f15f592
runconvergence(Λ2d,[10*4^k for k=1:7], rectgrid)

# ╔═╡ 2bc3ce81-cccd-400a-959a-d8980aff5de9
runconvergence(Λ2d,[10*4^k for k=1:7], randgrid)

# ╔═╡ b35cffd9-1901-467c-8fdd-6bd31de8572d
md"""
### 3D isotropic
"""

# ╔═╡ c5b2f3cb-456f-4341-9cd3-eb353e1ba8fc
const Λ3d=Diagonal(ones(3))

# ╔═╡ c958970a-eb33-4bb6-9368-02f4358abd96
f3d(X)=-∇Λ∇(finitebell,X,Λ3d)

# ╔═╡ 0109c34c-dca4-4d62-80b8-9c466e8c9359
sol3d=femsolve(grid3d, Λ3d, f3d, β)

# ╔═╡ 655019af-13a8-4a40-b313-fe2d1c3f0d6c
function plot3dresults(grid3d,sol3d)
	vis=GridVisualizer(size=(600,300),layout=(1,2), Plotter=PlutoVista)
	scalarplot!(vis[1,1],grid3d,sol3d, title="approx")
	scalarplot!(vis[1,2],grid3d,map(finitebell,grid3d), title="exact")
	reveal(vis)
end


# ╔═╡ 277ee78a-edbc-48ea-99d2-7de7c8b9720c
plot3dresults(grid3d,sol3d)

# ╔═╡ c3df0646-9f40-4dd2-9a47-44409b5b179b
runconvergence(Λ3d,[10*8^k for k=1:5], randgrid)

# ╔═╡ a2296968-0fdf-40bb-a254-3a45ed3650db
runconvergence(Λ3d,[10*8^k for k=1:5], rectgrid)

# ╔═╡ df8f3f9c-df90-4441-92a4-2cda52f3121a
md"""
### 2D Anisotropic
"""

# ╔═╡ 65e207b5-1180-4d8b-88cc-c315a0860165
Λ2d_a=ΛMatrix(100,π/4)

# ╔═╡ 0c2941bd-ab00-4aa2-bf34-8951efea1d01
grid2d_a=randgrid(2,10000)

# ╔═╡ 4072a4d3-b31b-430f-b2b2-1284c345fa7a
sol2d_a=femsolve(grid2d_a, Λ2d_a, X->-∇Λ∇(finitebell,X,Λ2d_a), β)

# ╔═╡ d409f9e4-f6a3-48b8-b073-704f50cdcad7
plot2dresults(grid2d_a,sol2d_a)

# ╔═╡ 2e74407d-7e87-45ff-aba5-1a76fd5ad685
runconvergence(Λ2d_a,[10*4^k for k=1:7], randgrid)

# ╔═╡ 578343a2-9050-4d76-9847-8cda124f7504
runconvergence(Λ2d_a,[10*4^k for k=1:7], rectgrid)

# ╔═╡ 9f642f6d-adeb-40a2-8c1c-1a5628183f99
md"""
### 3D Anisotropic
"""

# ╔═╡ 23540662-e08b-4513-bcee-0ba02d94c327
Λ3d_a=ΛMatrix(1000,100,π/4)

# ╔═╡ 4394fb2c-15fb-4999-ae0b-2d4be938f7af
grid3d_a=randgrid(3,100000)

# ╔═╡ 5143dfb5-5137-4bb1-803e-44fe40131b3d
sol3d_a=femsolve(grid3d_a, Λ3d_a, X->-∇Λ∇(finitebell,X,Λ3d_a), β)

# ╔═╡ 3d6716b6-c25e-4f3c-abbc-ba7803e4d1c5
plot3dresults(grid3d_a,sol3d_a)

# ╔═╡ 3309f370-9d6e-414a-8c8f-35a9c0bfa658
runconvergence(Λ3d_a,[10*8^k for k=1:5], randgrid)

# ╔═╡ 458eaec7-95eb-4168-b667-383e517f898b
runconvergence(Λ3d_a,[10*8^k for k=1:5], rectgrid)

# ╔═╡ Cell order:
# ╠═784b4c3e-bb2a-4940-a83a-ed5e5898dfd4
# ╠═3866057c-8a2b-481b-8399-6168e7a9f20b
# ╟─2471465b-64b7-4f25-97b0-75957940d68d
# ╟─10e998d3-1bb3-4591-b340-f9b93e027738
# ╠═64a7b893-8add-4c52-9725-c8b413c59e40
# ╠═53cf08d6-f3d8-4a39-ad22-b395da8cda6a
# ╟─5aa8a153-1ffa-4820-bcd2-83393170636a
# ╟─e2747647-ab45-4424-bb6f-89d9e965bad0
# ╠═4492bf1e-117b-4c26-bc6a-44b252332aeb
# ╠═f558c158-406e-4612-91d6-bbb52f29c3af
# ╠═2b695f50-07a6-4655-8876-5c5abf13af35
# ╠═817b63cc-4c5a-446d-b22b-d52fc91ee35c
# ╟─f5bad455-7b9a-4473-8b39-80aa19e17514
# ╠═53908721-deec-4d31-a121-5075fc03f3e3
# ╠═99705ed4-546a-4739-afc7-91120f33723e
# ╠═76c28a15-4135-4b6e-9f6a-cd2d54d93eaf
# ╠═0f0becb1-6588-4e67-9d6c-bf9458ace643
# ╟─299ed24c-c5d6-4c49-b407-cda278173477
# ╠═4ceb53f9-7a2e-4ccb-a2fc-9fb45848fb85
# ╠═0aeedbf2-de4b-4845-bf85-3e831e48c14b
# ╠═8bac06c2-1649-4060-8335-691518e8805c
# ╠═4f2ceb25-176f-4952-b2b8-fb7ba2f7ed90
# ╟─bb44eba7-ca38-4280-928e-d61f55691539
# ╠═ea5ffa60-e0f1-4f9e-891a-6bf4aeffe46e
# ╟─7fffecda-956c-43f7-a3b4-d78396def0c7
# ╠═a1c9cef9-bd7d-40d5-be50-05c9355b2f2b
# ╠═41b0e5b7-590c-431e-99ff-1247aeac24a7
# ╠═d9738a22-db04-47a2-bded-c4de3417cab4
# ╠═510addc3-deec-45a5-b15b-1587d985e227
# ╠═00349f7c-6010-4def-bd9d-f285ab695f2d
# ╠═59fcedaa-a78f-43a9-8dda-6ebfd67805bf
# ╟─2567c910-c04a-4d7b-be97-1a0013b1913a
# ╠═5084ef28-0194-4050-a24a-b60139b41000
# ╠═f19f0fa6-956b-4c5d-8842-4a8492890017
# ╠═d11d26f8-8273-4321-a8bb-f4d46307d895
# ╠═a8297522-61fb-4a59-ac18-9a113698c649
# ╠═348126b6-76d7-4f97-a867-83c684e05c67
# ╠═48997cd2-4d81-49e4-aea9-d7e18f15f592
# ╠═2bc3ce81-cccd-400a-959a-d8980aff5de9
# ╟─b35cffd9-1901-467c-8fdd-6bd31de8572d
# ╠═c5b2f3cb-456f-4341-9cd3-eb353e1ba8fc
# ╠═c958970a-eb33-4bb6-9368-02f4358abd96
# ╠═0109c34c-dca4-4d62-80b8-9c466e8c9359
# ╠═655019af-13a8-4a40-b313-fe2d1c3f0d6c
# ╠═277ee78a-edbc-48ea-99d2-7de7c8b9720c
# ╠═c3df0646-9f40-4dd2-9a47-44409b5b179b
# ╠═a2296968-0fdf-40bb-a254-3a45ed3650db
# ╟─df8f3f9c-df90-4441-92a4-2cda52f3121a
# ╠═65e207b5-1180-4d8b-88cc-c315a0860165
# ╠═0c2941bd-ab00-4aa2-bf34-8951efea1d01
# ╠═4072a4d3-b31b-430f-b2b2-1284c345fa7a
# ╠═d409f9e4-f6a3-48b8-b073-704f50cdcad7
# ╠═2e74407d-7e87-45ff-aba5-1a76fd5ad685
# ╠═578343a2-9050-4d76-9847-8cda124f7504
# ╟─9f642f6d-adeb-40a2-8c1c-1a5628183f99
# ╠═23540662-e08b-4513-bcee-0ba02d94c327
# ╠═4394fb2c-15fb-4999-ae0b-2d4be938f7af
# ╠═5143dfb5-5137-4bb1-803e-44fe40131b3d
# ╠═3d6716b6-c25e-4f3c-abbc-ba7803e4d1c5
# ╠═3309f370-9d6e-414a-8c8f-35a9c0bfa658
# ╠═458eaec7-95eb-4168-b667-383e517f898b
