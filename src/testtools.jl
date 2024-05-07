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
