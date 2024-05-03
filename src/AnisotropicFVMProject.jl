module AnisotropicFVMProject
using TestItems: @testitem
using LinearAlgebra: det,I, dot
using StaticArrays: MMatrix
using ExtendableGrids: local_celledgenodes, Edge1D, Triangle2D,
    Tetrahedron3D,simplexgrid, ExtendableGrid,
    Coordinates, CellNodes
import Triangulate, TetGen
using SimplexGridFactory

include("fem.jl")
export femgrad!, coordmatrix!, femnorms


include("testgrids.jl")

include("femtest.jl")


end # module AnisotropicFVMProject
