function femgrad(coord, cellnodes, icell)
    spacedim = size(coord, 1)
    celldim = size(cellnodes, 1)
    C = zeros(spacedim, spacedim)
    G = zeros(celldim, spacedim)
    coordmatrix!(C, coord, cellnodes, 1)
    femgrad!(G, C)
    G
end

function extended_femgrad(C)
    celldim = size(C, 1)
    spacedim = celldim - 1
    vol = abs(det(C)) / prod(1:spacedim)
    G = view(C \ I, :, 2:celldim)
    G
end

function extended_coordmatrix!(C, coord, cellnodes, icell)
    spacedim = size(coord, 1)
    celldim = size(cellnodes, 1)
    for jj = 1:celldim
        C[1, jj] = 1
        for ii = 1:spacedim
            C[ii + 1, jj] = coord[ii, cellnodes[jj, icell]]
        end
    end
end

function extended_femgrad(coord, cellnodes, icell)
    spacedim = size(coord, 1)
    celldim = size(cellnodes, 1)
    C = zeros(celldim, celldim)
    extended_coordmatrix!(C, coord, cellnodes, 1)
    extended_femgrad(C)
end

function femfactors(Λ, coord, cellnodes, icell)
    spacedim = size(coord, 1)
    celldim = size(cellnodes, 1)
    C = zeros(spacedim, spacedim)
    G = zeros(celldim, spacedim)
    ω = zeros(celldim)
    e = zeros(size(local_edgenodes[spacedim], 2))
    coordmatrix!(C, coord, cellnodes, 1)
    femgrad!(G, C)
    vol = abs(det(C)) / prod(1:spacedim)
    femfactors!(e, G, Λ, local_edgenodes[spacedim])
    ω .= vol / celldim
    ω, e * vol
end

function femstiffness(Λ, coord, cellnodes, icell)
    spacedim = size(coord, 1)
    celldim = size(cellnodes, 1)
    C = zeros(spacedim, spacedim)
    G = zeros(celldim, spacedim)
    S = zeros(celldim, celldim)
    coordmatrix!(C, coord, cellnodes, 1)
    femgrad!(G, C)
    femstiffness!(S, G, Λ)
    S
end

rotator(α) = [cos(α) -sin(α); sin(α) cos(α)]

function ΛMatrix(Λ11)
    Float64[Λ11;;] 
end

function ΛMatrix(Λ11, α)
    r = rotator(α)
    r * [Λ11 0; 0 1] * r'
end

function ΛMatrix(Λ11, Λ33, α)
    A2 = ΛMatrix(Λ11, α)
    A3 = zeros(3, 3)
    A3[1:2, 1:2] = A2
    A3[3, 3] = Λ33
    A3
end

"""
    finitebell_core(x)

Polynomial $(finitebell_core).
This polynomial has the following property:
- `p(0)=1`
- `p'(0)=1`
- `p''(0)=1`
- `p(1)=0`
- `p'(1)=0`
- `p''(1)=0`

Defined in $(joinpath("src",basename(@__FILE__))).
"""
const finitebell_core = Polynomial([1, 0, 0, -10, 15, -6])

"""
    finitebell(x)
    finitebell(x,y)
    finitebell(x,y,z)
    finitebell([x])
    finitebell([x,y])
    finitebell([x,y,z])

Twice differentiable function of one, two or three variables with finite support 
such that for `r=x` or `r=norm(X)`, `finitebell(r)=finitebell_core(r)` if `r<1` ,
otherwise, `finitebell(r)=0`. 

Defined in $(joinpath("src",basename(@__FILE__)))
"""
function finitebell end

function finitebell(X)
    rr = 2 * sum(ξ -> ξ^2, X)
    rr < 1 ? finitebell_core(rr) : zero(eltype(X))
end

finitebell(x::Number) = finitebell((x,))
finitebell(x, y) = finitebell((x, y))
finitebell(x, y, z) = finitebell((x, y, z))


paraprod(X) = abs(prod(x->(x+1)*(x-1), X))

"""
    d1finitebell(x)
Derivative of finitebell, calculated using automatic differentiation

Defined in $(joinpath("src",basename(@__FILE__)))
"""
d1finitebell(x) = Tensors.gradient(finitebell, x)

"""
    d2finitebell(x)
Second derivative of finitebell, calculated using automatic differentiation

Defined in $(joinpath("src",basename(@__FILE__)))
"""
d2finitebell(x) = Tensors.gradient(d1finitebell, x)

"""
     ∇Λ∇(u,x,Λ=I)

For a matrix Λ, with the help of automatic
differentiation apply differential operator ``\\nabla\\cdot \\Lambda  \\nabla`` to a function


Defined in $(joinpath("src",basename(@__FILE__)))
"""
function ∇Λ∇ end

∇Λ∇(u::Func, x, Λ = I) where {Func} = Tensors.divergence(x -> Tensors.Vec((Λ * Tensors.gradient(u, x))...), Tensors.Vec(x...))

∇Λ∇(u::Func, x::Number, Λ = 1) where {Func} = ∇Λ∇(x -> u(x[1]), Tensors.Vec(x), Λ * I)[1]

"""
   ∇ηΛ∇(u,x,η=u->u,Λ=I)

For a matrix Λ, and a function η with the help of automatic
differentiation apply differential operator ``\\nabla\\cdot \\eta\\Lambda  \\nabla`` to a function


Defined in $(joinpath("src",basename(@__FILE__)))
"""
function ∇ηΛ∇ end

∇ηΛ∇(u::Func, x, η::UFunc=u->u, Λ = I) where {Func, UFunc} = Tensors.divergence(x -> Tensors.Vec((η(u(x))*Λ * Tensors.gradient(u, x))...), Tensors.Vec(x...))

∇ηΛ∇(u::Func, x::Number, ηη::UFunc=u->u,  Λ = 1) where {Func, UFunc} = ∇ηΛ∇(x -> u(x[1]), Tensors.Vec(x),η, Λ * I)[1]

function hminmax(grid)
    cellnodes = grid[CellNodes]
    coord = grid[Coordinates]
    spacedim = size(coord, 1)
    en = local_edgenodes[spacedim]
    xcoord = reinterpret(reshape, SVector{spacedim, Float64}, coord)
    function run(xcoord::T) where {T}
        hmin = 1.0e30
        hmax = 0.0
        for icell = 1:size(cellnodes, 2)
            for ie = 1:size(en, 2)
                ig1 = cellnodes[en[1, ie], icell]
                ig2 = cellnodes[en[2, ie], icell]
                h = norm(xcoord[ig1] - xcoord[ig2])
                hmin = min(h, hmin)
                hmax = max(h, hmax)
            end
        end
        hmin, hmax
    end
    run(xcoord)
end


struct ScalarTestData{Tη, Tu, Tf}
    Λ::Matrix{Float64}
    η::Tη
    u::Tu
    f::Tf
end
function ScalarTestData(;
			Λ=[1.0 0; 0 1.0], 
			η=u->1, 
			u=X->0, 
			f=X->-∇ηΛ∇(u,X,η,Λ))
    ScalarTestData(Λ,η,u,f)
end


function udirichlet(y, u, bnodedata, userdata)
    dirichlet!(bnodedata, y, u, userdata.u(coord(bnodedata)))
end

function hdirichlet(y, u, bnodedata, userdata)
    for ispec=1:size(u,1)
        dirichlet!(bnodedata, y, u, 0.0; ispec)
    end
end

function hneumann(y, u, bnodedata, userdata)
end

function rectgrid(dim, nnodes; X=(-1,1))
    n0 = Int(ceil((nnodes)^(1 / dim)))
    X = range(X...; length = n0)
    simplexgrid(Tuple(X for i = 1:dim)...)
end

function randgrid(dim, nnodes; X=(-1,1))
    if dim == 1
        X = range(X...; length = nnodes) |> collect
        h = X[2] - X[1]
        for i = 2:(nnodes - 1)
            pert = 0.5 * h * rand() - 0.25 * h
            X[i] += pert
        end
        simplexgrid(X)
    elseif dim == 2
        builder = SimplexGridBuilder(; Generator = Triangulate)

        p1 = point!(builder, X[1], X[1])
        p2 = point!(builder, X[2], X[1])
        p3 = point!(builder, X[2], X[2])
        p4 = point!(builder, X[1], X[2])
        point!(builder, rand() - 0.5, rand() - 0.5)

        facetregion!(builder, 1)
        facet!(builder, p1, p2)
        facetregion!(builder, 2)
        facet!(builder, p2, p3)
        facetregion!(builder, 3)
        facet!(builder, p3, p4)
        facetregion!(builder, 4)
        facet!(builder, p4, p1)
        simplexgrid(builder; maxvolume = 0.75*(X[2]-X[1])^2 / nnodes)
    elseif dim == 3
        builder = SimplexGridBuilder(; Generator = TetGen)
        p1 = point!(builder, X[1], X[1], X[1])
        p2 = point!(builder, X[2], X[1], X[1])
        p3 = point!(builder, X[2], X[2], X[1])
        p4 = point!(builder, X[1], X[2], X[1])
        p5 = point!(builder, X[1], X[1], X[2])
        p6 = point!(builder, X[2], X[1], X[2])
        p7 = point!(builder, X[2], X[2], X[2])
        p8 = point!(builder, X[1], X[2], X[2])
        # perturb for randoness
        point!(builder, rand() - 0.5, rand() - 0.5, rand() - 0.5)
        facetregion!(builder, 1)
        facet!(builder, p1, p2, p3, p4)
        facetregion!(builder, 2)
        facet!(builder, p5, p6, p7, p8)
        facetregion!(builder, 3)
        facet!(builder, p1, p2, p6, p5)
        facetregion!(builder, 4)
        facet!(builder, p2, p3, p7, p6)
        facetregion!(builder, 5)
        facet!(builder, p3, p4, p8, p7)
        facetregion!(builder, 6)
        facet!(builder, p4, p1, p5, p8)
        simplexgrid(builder; maxvolume =  0.4*(X[2]-X[1])^3 / nnodes)
    end
end

function minplot(tsol; xlabel="t", ylabel="min(u(t))", kwargs...)
    mins=[minimum(tsol[i]) for i=1:length(tsol.t)]
    scalarplot(tsol.t,mins,size=(600,200); xlabel, ylabel, kwargs...)
end


function runconvergence(ref, dim, gengrid; celleval=nothing, bfaceeval=hdirichlet, data=ScalarTestData(), tol = 1.0e-8,
                        size=(300,300))
    h1norms = []
    l2norms = []
    h = []
    
    for r in ref
        grid = gengrid(dim, 10*2^(dim*r))
        hmin, hmax = hminmax(grid)
        push!(h, hmax)
        u = map(data.u, grid)
        sys=CVFEMSystem(grid,celleval,bfaceeval,data,1)
        sol = solve(sys; tol)
        l2, h1 = femnorms(grid, u - sol[1,:])
        @info "n=$(num_nodes(grid)), l2=$(l2), h1=$(h1)"
        push!(h1norms, h1)
        push!(l2norms, l2)
    end

    vis = GridVisualizer(; size)
    scalarplot!(vis[1,1], h, l2norms; color = :red, label = "l2", xscale = :log, yscale = :log, legend = :rb)
    scalarplot!(vis[1,1], h, (l2norms[1]/h[1]^2)*h .^ 2; color = :red, label = "O(h^2)", linestyle = :dot, clear = false)
    scalarplot!(vis[1,1], h, h1norms; color = :blue, label = "h1", linestyle = :solid, clear = false)
    scalarplot!(vis[1,1], h, (h1norms[1]/h[1])*h; color = :blue, label = "O(h)", linestyle = :dot, clear = false)
    reveal(vis)
end

function fourplots(grid,tsol;
                   ispec=1,
                   times=[tsol.t[1], 0.33*(tsol.t[end]-tsol.t[1]),0.66*(tsol.t[end]-tsol.t[1]), tsol.t[end]],
                   kwargs...)
	vis=GridVisualizer(layout=(2,2),resolution=(700,600),kwargs...)
	myplot(i,j,t)=
	scalarplot!(vis[i,j],grid,tsol(t)[ispec,:],colormap=:summer,levels=5, title="t=$t")	
	myplot(1,1,times[1])
	myplot(1,2,times[2])
	myplot(2,1,times[3])
	myplot(2,2,times[4])
		
#	mysave(fname,vis)
	reveal(vis)
end
