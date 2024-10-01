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
    using AnisotropicFVMProject: randgrid, rectgrid
    using AnisotropicFVMProject: finitebell, d1finitebell, d2finitebell, ∇ηΛ∇, hminmax, ΛMatrix
    using AnisotropicFVMProject: coord, transmission, nnodes, nedges, volume, edgenode, dirichlet!, solve, CVFEMSystem
    using ExtendableGrids: dim_space

    using SimplexGridFactory, ExtendableGrids
    using StaticArrays
    import Triangulate, TetGen, CairoMakie, PlutoVista
    using GridVisualize
    default_plotter!(CairoMakie)
    import PlutoUI
end

# ╔═╡ 3866057c-8a2b-481b-8399-6168e7a9f20b
PlutoUI.TableOfContents()

# ╔═╡ 235b02b5-37bf-4f9d-afc0-e69bae8c720c
md"""
# Nonlinear finite volume tests

We just test the linear case, as we know the solutions there.
"""

# ╔═╡ 5aa8a153-1ffa-4820-bcd2-83393170636a
md"""
## Grids 
"""

# ╔═╡ e2747647-ab45-4424-bb6f-89d9e965bad0
md"""
### Grid1d
"""

# ╔═╡ 4492bf1e-117b-4c26-bc6a-44b252332aeb
grid1d = randgrid(1, 100)

# ╔═╡ f558c158-406e-4612-91d6-bbb52f29c3af
gridplot(grid1d; Plotter = CairoMakie, size = (600, 100))

# ╔═╡ f5bad455-7b9a-4473-8b39-80aa19e17514
md"""
### Grid2D
"""

# ╔═╡ 53908721-deec-4d31-a121-5075fc03f3e3
grid2d = rectgrid(2, 160)

# ╔═╡ 99705ed4-546a-4739-afc7-91120f33723e
gridplot(grid2d; Plotter = CairoMakie, size = (400, 300))

# ╔═╡ 0c2941bd-ab00-4aa2-bf34-8951efea1d01
grid2d_a = randgrid(2, 1000)

# ╔═╡ be947e11-c1c0-4173-b682-e87b3523be7d
gridplot(grid2d_a)

# ╔═╡ 6f265307-3a25-4801-a930-1a0109570bba
md"""
### Enable 3D
"""

# ╔═╡ 72526e34-0bbc-4af1-b91d-e23fe59b7168
do3d=true

# ╔═╡ 69732afb-5e74-40fa-88fa-108e284e57cb
md"""
## Beta function
"""

# ╔═╡ 9b1d5d85-c422-4538-a72f-af22f25859fd
β(a,h)= 1-exp(-max(a,0)^2/(2h^2))

# ╔═╡ 77f07c4f-1952-4e58-855f-99f612367f1f
X=-3:0.01:3

# ╔═╡ 97f25501-14d1-49af-a869-941b2df66b54
let
	vis=GridVisualizer(size=(600,300),legend=:lt)
	scalarplot!(vis,X,β.(X,1), label="h=1", color=:darkred)
	scalarplot!(vis,X,β.(X,0.5), label="h=0.5", color=:darkgreen, clear=false)
	scalarplot!(vis,X,β.(X,0.1), label="h=0.1", color=:darkblue, clear=false)
	reveal(vis)
end

# ╔═╡ bb44eba7-ca38-4280-928e-d61f55691539
md"""
## FVM solutions
"""

# ╔═╡ ea5ffa60-e0f1-4f9e-891a-6bf4aeffe46e
β_D(X) = 0.0

# ╔═╡ 7bc7440a-a8b8-41b8-a6a0-77e79136844b
function nlfvmtest(grid; Λ = Diagonal(ones(dim_space(grid))), kwargs...)
    η(u) = 1 + u^2
	f(X) = -∇ηΛ∇(finitebell, X, η, Λ)
    g=zeros(num_nodes(grid))
  	hmin,hmax=hminmax(grid)
    hmax=0.1
    # Evaluate local residuum 
    function celleval!(y, u, celldata, userdata)
        y .= zero(eltype(y))
        ω = volume(celldata) / nnodes(celldata)
	    ηavg = 0.0
        for il = 1:nnodes(celldata)
            y[il] -= f(coord(celldata, il)) * ω
            ηavg += η(u[il]) / nnodes(celldata)
        end
        ΛKL = transmission(celldata, Λ)
        for ie = 1:nedges(celldata)
            i1 = edgenode(celldata, 1, ie)
            i2 = edgenode(celldata, 2, ie)
			if ΛKL[ie]≥0
				b=1.0
			else
				b=β(u[i1]-0,hmax)*β(u[i2]-0,hmax)
			end
            g = ηavg*ΛKL[ie] * (u[i1] - u[i2])
            y[i1] += g
            y[i2] -= g
        end
    end


    function bfaceeval!(y, u, bnodedata, userdata)
        dirichlet!(bnodedata, y, u, β_D(coord(bnodedata)))
    end

	sys=CVFEMSystem(grid,celleval!,bfaceeval!,nothing,1)
    solve(sys; kwargs...)		
end

# ╔═╡ 7fffecda-956c-43f7-a3b4-d78396def0c7
md"""
### 1D comparison plot
"""

# ╔═╡ 41b0e5b7-590c-431e-99ff-1247aeac24a7
sol1d = nlfvmtest(grid1d)

# ╔═╡ d9738a22-db04-47a2-bded-c4de3417cab4
function plot1dresults(grid, sol)
    vis = GridVisualizer(; size = (600, 200))
    scalarplot!(vis, grid1d, sol1d; color = :red)
    scalarplot!(vis, grid1d, finitebell; color = :green, clear = false)
    reveal(vis)
end

# ╔═╡ 510addc3-deec-45a5-b15b-1587d985e227
plot1dresults(grid1d, sol1d)

# ╔═╡ 00349f7c-6010-4def-bd9d-f285ab695f2d
function runconvergence(ref, dim, gengrid; Λ = Diagonal(ones(dim)), tol = 1.0e-8)
    h1norms = []
    l2norms = []
	mins=[]
    h = []
    for n in ref
        grid = gengrid(dim, n)
        hmin, hmax = hminmax(grid)
        push!(h, hmax)
        u = map(finitebell, grid)
        sol = nlfvmtest(grid; Λ, tol)
        l2, h1 = femnorms(grid, u - sol[1,:])
		solmin=minimum(sol) 
        @info "--------------", num_nodes(grid), solmin, l2, h1
        push!(h1norms, h1)
        push!(l2norms, l2)
        push!(mins, solmin)
    end

    vis = GridVisualizer(; layout=(1,2), size = (700, 300))
    scalarplot!(vis[1,1], h, l2norms; color = :red, label = "l2", xscale = :log, yscale = :log, legend = :lt)
    scalarplot!(vis[1,1], h, h .^ 2; color = :red, label = "O(h^2)", linestyle = :dot, clear = false)
    scalarplot!(vis[1,1], h, h1norms; color = :blue, label = "h1", linestyle = :solid, clear = false)
    scalarplot!(vis[1,1], h, h; color = :blue, label = "O(h)", linestyle = :dot, clear = false)
	scalarplot!(vis[1,2], h, mins, xscale=:log, label="min", legend=:lt)
    reveal(vis)
end

# ╔═╡ 59fcedaa-a78f-43a9-8dda-6ebfd67805bf
runconvergence([10 * 2^k for k = 1:10], 1, rectgrid; tol = 1.0e-8)

# ╔═╡ 2567c910-c04a-4d7b-be97-1a0013b1913a
md"""
### 2D isotropic
"""

# ╔═╡ d11d26f8-8273-4321-a8bb-f4d46307d895
sol2d = nlfvmtest(grid2d; tol = 10e-8)

# ╔═╡ a8297522-61fb-4a59-ac18-9a113698c649
function plot2dresults(grid2d, sol2d)
    vis = GridVisualizer(; size = (600, 200), layout = (1, 2))
    scalarplot!(vis[1, 1], grid2d, sol2d[1,:]; title = "approx")
    scalarplot!(vis[1, 2], grid2d, map(finitebell, grid2d); title = "exact")
    reveal(vis)
end

# ╔═╡ 348126b6-76d7-4f97-a867-83c684e05c67
plot2dresults(grid2d, sol2d)

# ╔═╡ 48997cd2-4d81-49e4-aea9-d7e18f15f592
runconvergence([10 * 4^k for k = 1:7], 2, rectgrid; tol = 1.0e-7)

# ╔═╡ 2bc3ce81-cccd-400a-959a-d8980aff5de9
runconvergence([10 * 4^k for k = 1:7], 2, randgrid; tol = 1.0e-8)

# ╔═╡ b35cffd9-1901-467c-8fdd-6bd31de8572d
md"""
### 3D isotropic
"""

# ╔═╡ c958970a-eb33-4bb6-9368-02f4358abd96
f3d(X) = -∇Λ∇(finitebell, X, Λ3d)

# ╔═╡ 4ceb53f9-7a2e-4ccb-a2fc-9fb45848fb85
begin
	do3d
	grid3d = randgrid(3, 10000)
end

# ╔═╡ 0aeedbf2-de4b-4845-bf85-3e831e48c14b
gridplot(grid3d; Plotter = PlutoVista, xplanes = [0.0])

# ╔═╡ 0109c34c-dca4-4d62-80b8-9c466e8c9359
sol3d = nlfvmtest(grid3d; tol = 1.0e-8)

# ╔═╡ 655019af-13a8-4a40-b313-fe2d1c3f0d6c
function plot3dresults(grid3d, sol3d)
    vis = GridVisualizer(; size = (600, 300), layout = (1, 2), Plotter = PlutoVista)
    scalarplot!(vis[1, 1], grid3d, sol3d[1,:]; title = "approx")
    scalarplot!(vis[1, 2], grid3d, map(finitebell, grid3d); title = "exact")
    reveal(vis)
end

# ╔═╡ 277ee78a-edbc-48ea-99d2-7de7c8b9720c
plot3dresults(grid3d, sol3d)

# ╔═╡ c3df0646-9f40-4dd2-9a47-44409b5b179b
begin
	do3d
runconvergence([10 * 8^k for k = 1:4], 3, randgrid)
end

# ╔═╡ a2296968-0fdf-40bb-a254-3a45ed3650db
begin
	do3d
runconvergence([10 * 8^k for k = 1:4], 3, rectgrid)
end

# ╔═╡ df8f3f9c-df90-4441-92a4-2cda52f3121a
md"""
### 2D Anisotropic
"""

# ╔═╡ 65e207b5-1180-4d8b-88cc-c315a0860165
Λ2d_a = ΛMatrix(100, π / 4)

# ╔═╡ 4072a4d3-b31b-430f-b2b2-1284c345fa7a
sol2d_a = nlfvmtest(grid2d_a; Λ = Λ2d_a)

# ╔═╡ d409f9e4-f6a3-48b8-b073-704f50cdcad7
plot2dresults(grid2d_a, sol2d_a)

# ╔═╡ 2e74407d-7e87-45ff-aba5-1a76fd5ad685
runconvergence([10 * 4^k for k = 1:7], 2, randgrid; Λ = Λ2d_a, tol=1.0e-7)

# ╔═╡ 578343a2-9050-4d76-9847-8cda124f7504
runconvergence([10 * 4^k for k = 1:7], 2, rectgrid; Λ = Λ2d_a,)

# ╔═╡ 9f642f6d-adeb-40a2-8c1c-1a5628183f99
md"""
### 3D Anisotropic
"""

# ╔═╡ 23540662-e08b-4513-bcee-0ba02d94c327
Λ3d_a = ΛMatrix(1000, 100, π / 4)

# ╔═╡ 4394fb2c-15fb-4999-ae0b-2d4be938f7af
begin
do3d
grid3d_a = randgrid(3, 100000)

end

# ╔═╡ 5143dfb5-5137-4bb1-803e-44fe40131b3d
sol3d_a = nlfvmtest(grid3d_a; Λ = Λ3d_a)

# ╔═╡ 3d6716b6-c25e-4f3c-abbc-ba7803e4d1c5
plot3dresults(grid3d_a, sol3d_a)

# ╔═╡ 3309f370-9d6e-414a-8c8f-35a9c0bfa658
begin
	do3d
runconvergence([10 * 8^k for k = 1:4], 3, randgrid; Λ = Λ3d_a)

end

# ╔═╡ 458eaec7-95eb-4168-b667-383e517f898b
begin
do3d
runconvergence([10 * 8^k for k = 1:4], 3, rectgrid; Λ = Λ3d_a)
end

# ╔═╡ Cell order:
# ╠═784b4c3e-bb2a-4940-a83a-ed5e5898dfd4
# ╠═3866057c-8a2b-481b-8399-6168e7a9f20b
# ╟─235b02b5-37bf-4f9d-afc0-e69bae8c720c
# ╟─5aa8a153-1ffa-4820-bcd2-83393170636a
# ╟─e2747647-ab45-4424-bb6f-89d9e965bad0
# ╠═4492bf1e-117b-4c26-bc6a-44b252332aeb
# ╠═f558c158-406e-4612-91d6-bbb52f29c3af
# ╟─f5bad455-7b9a-4473-8b39-80aa19e17514
# ╠═53908721-deec-4d31-a121-5075fc03f3e3
# ╠═99705ed4-546a-4739-afc7-91120f33723e
# ╠═0c2941bd-ab00-4aa2-bf34-8951efea1d01
# ╠═be947e11-c1c0-4173-b682-e87b3523be7d
# ╟─6f265307-3a25-4801-a930-1a0109570bba
# ╠═72526e34-0bbc-4af1-b91d-e23fe59b7168
# ╟─69732afb-5e74-40fa-88fa-108e284e57cb
# ╠═9b1d5d85-c422-4538-a72f-af22f25859fd
# ╠═77f07c4f-1952-4e58-855f-99f612367f1f
# ╠═97f25501-14d1-49af-a869-941b2df66b54
# ╟─bb44eba7-ca38-4280-928e-d61f55691539
# ╠═7bc7440a-a8b8-41b8-a6a0-77e79136844b
# ╠═ea5ffa60-e0f1-4f9e-891a-6bf4aeffe46e
# ╟─7fffecda-956c-43f7-a3b4-d78396def0c7
# ╠═41b0e5b7-590c-431e-99ff-1247aeac24a7
# ╠═d9738a22-db04-47a2-bded-c4de3417cab4
# ╠═510addc3-deec-45a5-b15b-1587d985e227
# ╠═00349f7c-6010-4def-bd9d-f285ab695f2d
# ╠═59fcedaa-a78f-43a9-8dda-6ebfd67805bf
# ╟─2567c910-c04a-4d7b-be97-1a0013b1913a
# ╠═d11d26f8-8273-4321-a8bb-f4d46307d895
# ╠═a8297522-61fb-4a59-ac18-9a113698c649
# ╠═348126b6-76d7-4f97-a867-83c684e05c67
# ╠═48997cd2-4d81-49e4-aea9-d7e18f15f592
# ╠═2bc3ce81-cccd-400a-959a-d8980aff5de9
# ╟─b35cffd9-1901-467c-8fdd-6bd31de8572d
# ╠═c958970a-eb33-4bb6-9368-02f4358abd96
# ╠═4ceb53f9-7a2e-4ccb-a2fc-9fb45848fb85
# ╠═0aeedbf2-de4b-4845-bf85-3e831e48c14b
# ╠═0109c34c-dca4-4d62-80b8-9c466e8c9359
# ╠═655019af-13a8-4a40-b313-fe2d1c3f0d6c
# ╠═277ee78a-edbc-48ea-99d2-7de7c8b9720c
# ╠═c3df0646-9f40-4dd2-9a47-44409b5b179b
# ╠═a2296968-0fdf-40bb-a254-3a45ed3650db
# ╟─df8f3f9c-df90-4441-92a4-2cda52f3121a
# ╠═65e207b5-1180-4d8b-88cc-c315a0860165
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
