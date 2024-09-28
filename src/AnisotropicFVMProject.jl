module AnisotropicFVMProject
using TestItems: @testitem
import Tensors
using LinearAlgebra: det, I, dot, norm, Diagonal
using Polynomials: Polynomial
using StaticArrays: MVector, MMatrix, SVector, @MMatrix, @MVector
using ExtendableGrids: local_celledgenodes, Edge1D, Triangle2D, num_nodes, dim_space,
                       Tetrahedron3D, simplexgrid, ExtendableGrid,
                       Coordinates, CellNodes, BFaceNodes
import Triangulate, TetGen
using SimplexGridFactory
using ExtendableSparse, SparseArrays
using LinearSolve: LinearProblem, solve
using AMGCLWrap
using Einsum: @einsum
using OMEinsum: @ein
using ForwardDiff, DiffResults
using Krylov: bicgstab, cg
using AlgebraicMultigrid: smoothed_aggregation, aspreconditioner
using SciMLBase: SciMLBase, solve
using RecursiveArrayTools: RecursiveArrayTools, AbstractDiffEqArray

include("fem.jl")
export femgrad!, coordmatrix!, femnorms, femsolve

include("fvm.jl")
export fvmsolve

include("nlfvm.jl")
export CFVEMSystem, solve
include("testgrids.jl")

include("testtools.jl")

end # module AnisotropicFVMProject
