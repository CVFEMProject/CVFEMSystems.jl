using AnisotropicFVMProject
using AnisotropicFVMProject: randgrid,femgrad, femstiffness, extended_femgrad, femfactors
using ExtendableGrids: dim_space
using TestItemRunner
using LinearAlgebra: Diagonal
using Test

const coord1d=[0.0 1.0;]
const cellnodes1d=[1; 2]
const coord2d=[0.0 0.5 1.0; 0.0  0.5  0.0]
const cellnodes2d=[1 2 3;]'
const coord3d=[0.0 0.5 1.0 0.25; 0.0  0.5  0.0 0.25; 0.0 0.0 0.0 1.0  ]
const cellnodes3d=[1 2 3 4;]'


@testset "femgrad" begin
    G=femgrad(coord1d,cellnodes1d,1)
    @test G ≈ [-1.0; 1.0;;]
    Gext=extended_femgrad(coord1d,cellnodes1d,1)
    @test Gext≈G

    
    G=femgrad(coord2d,cellnodes2d,1)
    @test G≈[-1.0 -1.0; 0.0 2.0; 1.0 -1.0]
    Gext=extended_femgrad(coord2d,cellnodes2d,1)
    @test Gext≈G
    
    G=femgrad(coord3d,cellnodes3d,1)
    @test G≈[-1.0 -1.0 -0.5; -0.0 2.0 -0.5; 1.0 -1.0 0.0; 0.0 0.0 1.0]
    Gext=extended_femgrad(coord3d,cellnodes3d,1)
    @test Gext≈G
end

@testset "femstiffness_0" begin
    S=femstiffness(Diagonal(ones(1)),coord1d,cellnodes1d,1)
    @test S≈[1.0 -1.0; -1.0 1.0]
    S=femstiffness(Diagonal(ones(2)),coord2d,cellnodes2d,1)
    @test S≈[2.0 -2.0 0.0; -2.0 4.0 -2.0; 0.0 -2.0 2.0]
    S=femstiffness(Diagonal(ones(3)),coord3d,cellnodes3d,1)
    @test S ≈ [2.25 -1.75 0.0 -0.5; -1.75 4.25 -2.0 -0.5; 0.0 -2.0 2.0 0.0; -0.5 -0.5 0.0 1.0]
end


# @testset "femfactors_0" begin
#     ω,e=femfactors(Diagonal(ones(1)),coord1d,cellnodes1d,1)
#     @test ω≈[0.5, 0.5]
#     @test e≈[1.0]
#     ω,e=femfactors(Diagonal(ones(2)),coord2d,cellnodes2d,1)
#     @test ω≈[0.08333333333333333, 0.08333333333333333, 0.08333333333333333]
#     @test e≈[0.5, 0.5, -0.0]
#     ω,e=femfactors(Diagonal(ones(3)),coord3d,cellnodes3d,1)
#     @test ω≈[0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332]
#     @test e≈[0.14583333333333331, -0.0, 0.041666666666666664, 0.16666666666666666, 0.041666666666666664, -0.0]
#end
#




@testset "femnorms" begin
    function check_const(grid,c)
	dim=dim_space(grid)
        l2,h1= femnorms(grid,map((x...)->c, grid))
        l2≈c*sqrt(2^dim) && isapprox(h1,0, atol=1.0e-5)
    end
    
    function check_lin(grid,c)
	dim=dim_space(grid)
 	l2,h1= femnorms(grid,map((x...) -> (c * sum(x) / sqrt(dim)), grid))
        h1≈c*sqrt(2^dim) 
    end

    @test check_const(randgrid(1,100),1)
    @test check_const(randgrid(2,100),1)
    @test check_const(randgrid(3,100),1)

    @test check_lin(randgrid(1,100),1)
    @test check_lin(randgrid(2,100),1)
    @test check_lin(randgrid(3,100),1)
end

