function rectgrid(dim, nnodes)
    n0 = Int(ceil((nnodes)^(1 / dim)))
    X = range(-1, 1; length = n0)
    simplexgrid(Tuple(X for i = 1:dim)...)
end

function randgrid(dim, nnodes)
    if dim == 1
        X = zeros(nnodes)
        X = range(-1, 1; length = nnodes) |> collect
        h = X[2] - X[1]
        for i = 2:(nnodes - 1)
            pert = 0.5 * h * rand() - 0.25 * h
            X[i] += pert
        end
        simplexgrid(X)
    elseif dim == 2
        builder = SimplexGridBuilder(; Generator = Triangulate)

        p1 = point!(builder, -1, -1)
        p2 = point!(builder, 1, -1)
        p3 = point!(builder, 1, 1)
        p4 = point!(builder, -1, 1)
        point!(builder, rand() - 0.5, rand() - 0.5)

        facetregion!(builder, 1)
        facet!(builder, p1, p2)
        facetregion!(builder, 2)
        facet!(builder, p2, p3)
        facetregion!(builder, 3)
        facet!(builder, p3, p4)
        facetregion!(builder, 4)
        facet!(builder, p4, p1)
        simplexgrid(builder; maxvolume = 3 / nnodes)
    elseif dim == 3
        builder = SimplexGridBuilder(; Generator = TetGen)
        p1 = point!(builder, -1, -1, -1)
        p2 = point!(builder, 1, -1, -1)
        p3 = point!(builder, 1, 1, -1)
        p4 = point!(builder, -1, 1, -1)
        p5 = point!(builder, -1, -1, 1)
        p6 = point!(builder, 1, -1, 1)
        p7 = point!(builder, 1, 1, 1)
        p8 = point!(builder, -1, 1, 1)
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
        simplexgrid(builder; maxvolume = 2.5 / nnodes)
    end
end
