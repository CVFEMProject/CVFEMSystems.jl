module CVFEMSystems
using TestItems: @testitem
import Tensors
using LinearAlgebra: det, I, norm
using Polynomials: Polynomial
using StaticArrays: SVector, @MMatrix, @MVector
using ExtendableGrids: local_celledgenodes, Edge1D, Triangle2D, num_nodes, dim_space,
                       Tetrahedron3D, simplexgrid, ExtendableGrid,
                       Coordinates, CellNodes, BFaceNodes, BFaceRegions
import Triangulate, TetGen
using ExtendableSparse, SparseArrays
using Einsum: @einsum
using OMEinsum: @ein
using DiffResults: DiffResults
using ForwardDiff: ForwardDiff
using SciMLBase: SciMLBase, solve
using RecursiveArrayTools: RecursiveArrayTools, AbstractDiffEqArray
using GridVisualize: scalarplot,scalarplot!, GridVisualizer, reveal
using SimplexGridFactory
using AlgebraicMultigrid, Krylov

include("elementcalculations.jl")
export femgrad!, coordmatrix!, femnorms

include("femsolve.jl")
export femsolve

include("fvmsolve.jl")
export fvmsolve

include("celldata.jl")

include("bnodedata.jl")

include("transientsolution.jl")

include("cvfemsystem.jl")

export CFVEMSystem, solve

include("testtools.jl")

end # module CVFEMSystems
