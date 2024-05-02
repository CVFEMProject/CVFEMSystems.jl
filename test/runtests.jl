using AnisotropicFVMProject
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
    G,V=AnisotropicFVMProject.femgrad(coord1d,cellnodes1d,1)
    @test G ≈ [-1.0; 1.0;;]
    @test V ≈ 1.0
    Gext,Vext=AnisotropicFVMProject.extended_femgrad(coord1d,cellnodes1d,1)
    @test Gext≈G
    @test Vext≈V

    G,V=AnisotropicFVMProject.femgrad(coord2d,cellnodes2d,1)
    @test G≈[-1.0 -1.0; 0.0 2.0; 1.0 -1.0]
    @test V≈0.25
    Gext,Vext=AnisotropicFVMProject.extended_femgrad(coord2d,cellnodes2d,1)
    @test Gext≈G
    @test Vext≈V

    G,V=AnisotropicFVMProject.femgrad(coord3d,cellnodes3d,1)
    @test G≈[-1.0 -1.0 -0.5; -0.0 2.0 -0.5; 1.0 -1.0 0.0; 0.0 0.0 1.0]
    @test V≈0.08333333333333333
    Gext,Vext=AnisotropicFVMProject.extended_femgrad(coord3d,cellnodes3d,1)
    @test Gext≈G
    @test Vext≈V
end

rotator(α)=[cos(α) -sin(α); sin(α) cos(α)]
function ΛMatrix(Λ11,α)
    r=rotator(α)
    r*[Λ11 0 ; 0 1]*r'
end

@testset "femfactors_0" begin
    ω,e=AnisotropicFVMProject.femfactors(Diagonal(ones(1)),coord1d,cellnodes1d,1)
    @test ω≈[0.5, 0.5]
    @test e≈[1.0]
    ω,e=AnisotropicFVMProject.femfactors(Diagonal(ones(2)),coord2d,cellnodes2d,1)
    @test ω≈[0.08333333333333333, 0.08333333333333333, 0.08333333333333333]
    @test e≈[0.5, 0.5, -0.0]
    ω,e=AnisotropicFVMProject.femfactors(Diagonal(ones(3)),coord3d,cellnodes3d,1)
    @test ω≈[0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332]
    @test e≈[0.14583333333333331, -0.0, 0.041666666666666664, 0.16666666666666666, 0.041666666666666664, -0.0]
end


@testset "femstiffness_0" begin
    S=AnisotropicFVMProject.femstiffness(Diagonal(ones(1)),coord1d,cellnodes1d,1)
    @test S≈[1.0 -1.0; -1.0 1.0]
    S=AnisotropicFVMProject.femstiffness(Diagonal(ones(2)),coord2d,cellnodes2d,1)
    @test S≈[0.5 -0.5 0.0; -0.5 1.0 -0.5; 0.0 -0.5 0.5]
    S=AnisotropicFVMProject.femstiffness(Diagonal(ones(3)),coord3d,cellnodes3d,1)
    @test S ≈ [0.1875 -0.14583333333333331 0.0 -0.041666666666666664; -0.14583333333333331 0.35416666666666663 -0.16666666666666666 -0.041666666666666664; 0.0 -0.16666666666666666 0.16666666666666666 0.0; -0.041666666666666664 -0.041666666666666664 0.0 0.08333333333333333]
end
