function zero!(m::ExtendableSparseMatrix)
    nzv = nonzeros(m)
    nzv .= zero(eltype(nzv))
end


function  fvmsolve(coord,
                   cellnodes,
                   bfacenodes,
                   Λ,
                   f::Tf,
                   β::Tβ, # Boundary function
                   η::Tη,
                   tol,
                   ::Type{Val{spacedim}},
                   ::Type{Val{nedges}},
                   ) where {Tf,Tβ,Tη,spacedim,nedges}

    N=size(coord,2)
    Jac=ExtendableSparseMatrix(N,N)
    Res=zeros(N)
    U=zeros(N)
    maxit=10
    
    celldim=spacedim+1
    enodes=@MMatrix zeros(Int,2,nedges)
    enodes.=len[spacedim]
    E=@MVector zeros(nedges) # local stiffness matrix
    X=@MMatrix zeros(celldim, celldim) # local stiffness matrix
    C=@MMatrix zeros(spacedim,spacedim)	# local coordinate matrix
    G=@MMatrix zeros(celldim,spacedim)  # shape function grdients
    BC=@MMatrix zeros(spacedim,spacedim)     # local boundary coordinate matrix
    ncells=size(cellnodes,2)
    factdim=prod(1:spacedim)
    xcoord=reinterpret(reshape, SVector{spacedim,Float64},coord)
    vol::Float64=0.0
    icell::Int=0

    # Evaluate local residuum 
    function celleval!(y,u)
        y.=zero(eltype(y))
        ηavg=0.0
        for il=1:celldim
            ig=cellnodes[il,icell]
    	    y[il]-=f(xcoord[ig])*vol/celldim
            ηavg+=η(u[il])/celldim
        end

        for ie=1:nedges
            i1=enodes[1,ie]
	    i2=enodes[2,ie]
            efac=E[ie]*vol
            g=ηavg*efac*(u[i1]-u[i2])
            y[i1]+=g
            y[i2]-=g
        end
    end

    ulocal=zeros(celldim)
    ylocal=zeros(celldim)
    result=DiffResults.JacobianResult(ylocal,ulocal)
    config = ForwardDiff.JacobianConfig(celleval!, ylocal, ulocal)
    it=1
    while it<maxit
        # Set jaobian and residual to 0
        zero!(Jac)
        Res.=0.0
        # Evaluate residual and assemble Jacobian
        for iicell=1:ncells
            icell=iicell
            coordmatrix!(C,coord,cellnodes,icell)
            vol=abs(det(C))/factdim
            femgrad!(G,C)
            femfactors!(E,G,Λ,enodes)
            
            # Copy solution to ulocal
            for il=1:celldim
                ig=cellnodes[il,icell]
                ulocal[il]=U[ig]
            end

            # AD Jacobian 
            @time ForwardDiff.vector_mode_jacobian!(result, celleval!, ylocal, ulocal,config)

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
        nbfaces=size(bfacenodes,2)
        for ibface in 1:nbfaces
            for idim=1:spacedim
                i1=bfacenodes[idim,ibface];
                Jac[i1,i1]+=Dirichlet();
                Res[i1]+=Dirichlet()*β(xcoord[i1])
            end
        end
        if isnan(Res[1])
            error("NaN")
        end
        # Solve residual system
        update=Jac\Res
        # update solution
        U.-=update

        nm=norm(update,Inf)
        @info it, nm
        if nm<tol
            return U 
        end
        it+=1
    end  # while it
end



function fvmsolve(grid,Λ, f, β, η, tol=1.0e-8)
    coord=grid[Coordinates]
    cellnodes=grid[CellNodes]
    bfacenodes=grid[BFaceNodes]
    fvmsolve(coord, cellnodes, bfacenodes,Λ, f, β, η, tol,
             Val{dim_space(grid)}, Val{size(len[dim_space(grid)],2)})
end
