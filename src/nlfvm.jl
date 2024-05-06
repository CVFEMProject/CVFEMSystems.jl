function zero!(m::ExtendableSparseMatrix)
    nzv = nonzeros(m)
    nzv .= zero(eltype(nzv))
end


mutable struct CellData{Tgc,Tgn,Ten, Tcm,Tsg,Ttm}
    icell::Int64
    region::Int64
    globalcoord::Tgc
    globalnodes::Tgn
    edgenodes::Ten
    factdim::Int64
    coordmatrix::Tcm
    shapegradients::Tsg
    volume::Float64
    transmission::Ttm
end




function CellData(coord,edgenodes)
    spacedim=size(coord,1)
    globalcoord=reinterpret(reshape, SVector{spacedim,Float64},coord)
    celldim=spacedim+1
    icell=0
    region=0
    xedgenodes=@MMatrix zeros(Int64,2,size(edgenodes,2))
    xedgenodes.=edgenodes
    globalnodes=@MVector zeros(Int64,celldim)
    factdim=prod(1:spacedim)
    coordmatrix=@MMatrix zeros(spacedim,spacedim)
    shapegradients=@MMatrix zeros(celldim,spacedim)
    transmission=@MVector zeros(size(edgenodes,2)) # Transmission coefficients
    CellData(icell,
             region,
             globalcoord,
             globalnodes,
             xedgenodes,
             factdim,
             coordmatrix,
             shapegradients,
             0.0,
             transmission
             )
end

nnodes(celldata::CellData)=size(celldata.shapegradients,1)
spacedim(celldata::CellData)=size(celldata.shapegradients,2)
nedges(celldata::CellData)=size(celldata.edgenodes,2)
coord(celldata::CellData,il)=celldata.globalcoord[celldata.globalnodes[il]]
globalnode(celldata::CellData,il)=celldata.globalnodes[il]
volume(celldata::CellData)=celldata.volume
edgenode(celldata::CellData,i,ie)=celldata.edgenodes[i,ie]


function transmission(celldata,Λ)
    femfactors!(celldata.transmission,celldata.shapegradients,Λ,celldata.edgenodes)
    for ie=1:nedges(celldata)
        celldata.transmission[ie]*=celldata.volume
    end
    celldata.transmission
end

function update_celldata!(celldata,coord,cellnodes,icell)
    celldata.icell=icell
    for i=1:nnodes(celldata)
        celldata.globalnodes[i]=cellnodes[i,icell]
    end
    coordmatrix!(celldata.coordmatrix,coord,cellnodes,icell)
    celldata.volume=abs(det(celldata.coordmatrix))/celldata.factdim
    femgrad!(celldata.shapegradients,celldata.coordmatrix)
end



mutable struct BFaceData{Tgc,Tgn}
    ibface::Int64
    region::Int64
    globalcoord::Tgc
    globalnodes::Tgn
    volume::Float64
end

function BFaceData(coord)
    spacedim=size(coord,1)
    globalcoord=reinterpret(reshape, SVector{spacedim,Float64},coord)
    ibface=0
    region=0
    globalnodes=@MVector zeros(Int64,spacedim)
    BFaceData(ibface,
              region,
              globalcoord,
              globalnodes,
              0.0
              )
end

nnodes(bfacedata::BFaceData)=length(bfacedata.globalnodes)
volume(bfacedata::BFaceData)=bfacedata.volume
coord(bfacedata::BFaceData,il)=bfacedata.globalcoord[bfacedata.globalnodes[il]]

function update_bfacedata!(bfacedata, coordinates,bfacenodes,ibface)
    bfacedata.ibface=ibface
    for i=1:nnodes(bfacedata)
        bfacedata.globalnodes[i]=bfacenodes[i,ibface]
    end
    bfacedata.volume=bfacevolume(coordinates,bfacenodes,ibface)
end

function dirichlet!(bfacedata, y,u,il,val)
    y[il]=Dirichlet()*(u[il]-val)
end

function  fvmsolve(grid::ExtendableGrid,
                   celleval!::Tc,
                   bfaceeval!::Tb;
                   userdata=nothing,
                   tol=1.0e-10,
                   maxiter=10
                   ) where {Tc,Tb}

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

    bfacedata=BFaceData(coordinates)

    # Define wrapper function with closure in order
    # to fit the format uses by ForwardDiff
    wrap_celleval!(y,u)=celleval!(y,u,celldata,userdata)
    wrap_bfaceeval!(y,u)=bfaceeval!(y,u,bfacedata,userdata)
    
    ulocal=zeros(celldim)
    ylocal=zeros(celldim)
    result=DiffResults.JacobianResult(ylocal,ulocal)
    config = ForwardDiff.JacobianConfig(wrap_celleval!, ylocal, ulocal)

    nbfaces=size(bfacenodes,2)
    
    # Evaluate at boundary
    xcoord=reinterpret(reshape, SVector{spacedim,Float64},coordinates)

    ublocal=zeros(spacedim)
    yblocal=zeros(spacedim)
    bresult=DiffResults.JacobianResult(yblocal,ublocal)
    bconfig = ForwardDiff.JacobianConfig(wrap_bfaceeval!, yblocal, ublocal)

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
        
        for ibface in 1:nbfaces
            update_bfacedata!(bfacedata, coordinates,bfacenodes,ibface)
            # Copy solution to ulocal
            for il=1:spacedim
                ublocal[il]=U[bfacenodes[il,ibface]]
            end

            # AD residual and jacobian 
            ForwardDiff.vector_mode_jacobian!(bresult, wrap_bfaceeval!, yblocal, ublocal,bconfig)
            bres=DiffResults.value(bresult)
            bjac=DiffResults.jacobian(bresult)

            for il=1:spacedim
                ig=bfacenodes[il,ibface]
                Res[ig]+=bres[il]
                for jl=1:spacedim
                    jg=bfacenodes[jl,ibface]
                    Jac[ig,jg]+=bjac[il,jl]
                end
            end
        end

        # Solve residual system
        solver=AMGSolver(Jac, param= (solver=(type="bicgstab",),))
        update=solver\Res
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



