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
