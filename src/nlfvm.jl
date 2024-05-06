function zero!(m::ExtendableSparseMatrix)
    nzv = nonzeros(m)
    nzv .= zero(eltype(nzv))
end


mutable struct CellData{Tgc,Tgn,Ten, Tc,TG,TΛ}
    icell::Int64
    globalcoord::Tgc
    globalnodes::Tgn
    edgenodes::Ten
    factdim::Int64
    C::Tc
    G::TG
    vol::Float64
    Λ::TΛ
end

function CellData(coord,edgenodes)
    spacedim=size(coord,1)
    globalcoord=reinterpret(reshape, SVector{spacedim,Float64},coord)
    celldim=spacedim+1
    icell=0
    xedgenodes=@MMatrix zeros(Int64,2,size(edgenodes,2))
    xedgenodes.=edgenodes
    globalnodes=@MVector zeros(Int64,celldim)
    factdim=prod(1:spacedim)
    C=@MMatrix zeros(spacedim,spacedim)
    G=@MMatrix zeros(celldim,spacedim)
    Λ=@MVector zeros(size(edgenodes,2)) # Transmission coefficients
    CellData(icell,globalcoord,globalnodes,xedgenodes,factdim,C,G,0.0,Λ)
end

celldim(celldata::CellData)=size(celldata.G,1)
spacedim(celldata::CellData)=size(celldata.G,2)
nedges(celldata::CellData)=size(celldata.edgenodes,2)
coord(celldata,il)=celldata.globalcoord[celldata.globalnodes[il]]
globalnode(celldata,il)=celldata.globalnodes[il]

function update_celldata!(celldata,coord,cellnodes,icell)
    celldata.icell=icell
    for i=1:celldim(celldata)
        celldata.globalnodes[i]=cellnodes[i,icell]
    end
    coordmatrix!(celldata.C,coord,cellnodes,icell)
    celldata.vol=abs(det(celldata.C))/celldata.factdim
    @time femgrad!(celldata.G,celldata.C)
end

function transmission(celldata,Λ)
    femfactors!(celldata.Λ,celldata.G,Λ,celldata.edgenodes)
    for ie=1:nedges(celldata)
        celldata.Λ[ie]*=celldata.vol
    end
    celldata.Λ
end

function  fvmsolve(grid::ExtendableGrid,
                   celleval!::Tc,
                   userdata::Tu,
                   β::Tβ;
                   tol=1.0e-10,
                   maxiter=10
                   ) where {Tc,Tu,Tβ}

    coordinates=grid[Coordinates]
    cellnodes=grid[CellNodes]
    bfacenodes=grid[BFaceNodes]

    N=size(coordinates,2)
    Jac=ExtendableSparseMatrix(N,N)
    Res=zeros(N)
    U=zeros(N)
    
    spacedim=size(coordinates,1)
    celldim=spacedim+1
    celldata=CellData(coordinates,len[spacedim])
    ncells=size(cellnodes,2)

    # Define wrapper function with closure in order
    # to fit the format uses by ForwardDiff
    wrap_celleval!(y,u)=celleval!(y,u,celldata,userdata)
    
    ulocal=zeros(celldim)
    ylocal=zeros(celldim)
    result=DiffResults.JacobianResult(ylocal,ulocal)
    config = ForwardDiff.JacobianConfig(wrap_celleval!, ylocal, ulocal)
    it=1
    nalloc=0
    while it<maxiter
        # Set jaobian and residual to 0
        zero!(Jac)
        Res.=0.0
        # Evaluate residual and assemble Jacobian
        for icell=1:ncells
            update_celldata!(celldata, coordinates, cellnodes,icell)
            # Copy solution to ulocal
            for il=1:celldim
                ulocal[il]=U[cellnodes[il,icell]]
            end

            # AD residual and jacobian 
            ForwardDiff.vector_mode_jacobian!(result, wrap_celleval!, ylocal, ulocal,config)

            res=DiffResults.value(result)
            jac=DiffResults.jacobian(result)

            # Assemble into global data
            for il=1:celldim
                ig=cellnodes[il,icell]
                Res[ig]+=res[il]
                for jl=1:celldim
                    jg=cellnodes[jl,icell]
                    Jac[ig,jg]+=jac[il,jl]
                end
            end
        end # icell
        
        # Evaluate at boundary
        BC=@MMatrix zeros(spacedim,spacedim)     # local boundary coordinate matrix
        xcoord=reinterpret(reshape, SVector{spacedim,Float64},coordinates)
        nbfaces=size(bfacenodes,2)
        for ibface in 1:nbfaces
            for idim=1:spacedim
                i1=bfacenodes[idim,ibface];
                Jac[i1,i1]+=Dirichlet();
                Res[i1]+=Dirichlet()*β(xcoord[i1])
            end
        end

        # Solve residual system
        update=Jac\Res
        # update solution
        U.-=update

        nm=norm(update,Inf)
        @info it, nm
        if nm<tol
            if nalloc>0
                @warn "allocations in assembly loop"
            end
            return U 
        end
        it+=1
    end  # while it
    error("no convergence after maxiter=$(maxiter) iterations")
end



