module AnisotropicFVMProject
using TestItems: @testitem
import Tensors
using LinearAlgebra: det,I, dot, norm, Diagonal
using Polynomials: Polynomial
using StaticArrays: MVector, MMatrix, SVector, @MMatrix, @MVector
using ExtendableGrids: local_celledgenodes, Edge1D, Triangle2D, num_nodes,dim_space,
    Tetrahedron3D,simplexgrid, ExtendableGrid,
    Coordinates, CellNodes, BFaceNodes
import Triangulate, TetGen
using SimplexGridFactory
using ExtendableSparse,SparseArrays
using LinearSolve: LinearProblem,solve
using AMGCLWrap
using Einsum: @einsum
using OMEinsum: @ein
using ForwardDiff, DiffResults

include("fem.jl")
export femgrad!, coordmatrix!, femnorms, femsolve

include("fvm.jl")
export fvmsolve

include("nlfvm.jl")


include("testgrids.jl")

include("testtools.jl")


end # module AnisotropicFVMProject
