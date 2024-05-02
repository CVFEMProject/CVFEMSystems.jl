module AnisotropicFVMProject
using TestItems: @testitem
using LinearAlgebra: det,I, dot
using StaticArrays: MMatrix
using ExtendableGrids: local_celledgenodes, Edge1D, Triangle2D, Tetrahedron3D

include("fem.jl")
export femgrad!, coordmatrix!

end # module AnisotropicFVMProject
