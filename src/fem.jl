const len=[
    local_celledgenodes(Edge1D),
    local_celledgenodes(Triangle2D),
    local_celledgenodes(Tetrahedron3D),
]

const local_massmatrix1d=[ 2.0 1.0; 1.0 2.0 ]/6
const local_massmatrix2d=[ 2.0 1.0 1.0; 1.0 2.0 1.0; 1.0  1.0  2.0 ]/12
const local_massmatrix3d=[ 2.0 1.0 1.0 1.0; 1.0 2.0 1.0 1.0; 1.0  1.0  2.0 1.0; 1.0 1.0 1.0 2.0]/20
const local_massmatrix=[local_massmatrix1d,
                        local_massmatrix2d,
                        local_massmatrix3d]



function coordmatrix!(C,coord, cellnodes,icell)
    spacedim=size(coord,1)
    celldim=size(cellnodes,1)
    for jj=1:spacedim
        for ii=1:spacedim
            C[ii,jj]=coord[ii,cellnodes[jj,icell]]-coord[ii,cellnodes[celldim,icell]]
        end
    end
end


function femgrad!(G::Tg,C::Tc) where {Tg,Tc}
    spacedim=size(C,1)
    celldim=spacedim+1
    G[1:spacedim,1:spacedim].=C\I
    for i=1:spacedim
        G[celldim,i]=0.0
        for j=1:spacedim
            G[celldim,i]-=G[j,i]
        end
    end
end

function femstiffness!(S,G,Λ)
    @einsum S[il,jl]=G[il,k]*Λ[k,m]*G[jl,m]
end


function femstiffness!(S,G)
    @einsum S[il,jl]=G[il,k]*G[jl,k]
end



# Boundary face volume calculation by Heron's formula.
# We can do better here.

function dist2(coord,ig1,ig2)
    x=coord[1,ig1]-coord[1,ig2]
    y=coord[2,ig2]-coord[2,ig2]
    sqrt(x^2+y^2)
end

function dist3(coord,ig1,ig2)
    x=coord[1,ig1]-coord[1,ig2]
    y=coord[2,ig1]-coord[2,ig2]
    z=coord[3,ig1]-coord[3,ig2]
    sqrt(x^2+y^2+z^2)
end

bfacevolume(coord,bfacenodes,ibface,::Type{Val{1}})=1

bfacevolume(coord,bfacenodes,ibface,::Type{Val{2}})=dist2(coord,bfacenodes[1,ibface],bfacenodes[2,ibface])

function bfacevolume(coord,bfacenodes,ibface,::Type{Val{3}})
    a=dist3(coord,bfacenodes[1,ibface],bfacenodes[2,ibface])
    b=dist3(coord,bfacenodes[1,ibface],bfacenodes[3,ibface])
    c=dist3(coord,bfacenodes[2,ibface],bfacenodes[3,ibface])
    s=0.5*(a+b+c)
    sqrt(s*(s-a)*(s-b)*(s-c))
end

bfacevolume(coord,bfacenodes,ibface)=bfacevolume(coord,bfacenodes,ibface, Val{size(coord,1)})


function femnorms(coord,cellnodes,u)
    l2norm=0.0
    h1norm=0.0
    spacedim=size(coord,1)
    celldim=size(cellnodes,1)
    S=zeros(celldim, celldim)  # local stiffness matrix
    C=zeros(spacedim,spacedim) # local coordinate matrix
    G=zeros(celldim, spacedim) # shape function gradients
    M=local_massmatrix[spacedim]
    ncells=size(cellnodes,2)
    factdim=prod(1:spacedim)
    for icell=1:ncells
	coordmatrix!(C,coord,cellnodes,icell)
        vol=abs(det(C))/factdim
        femgrad!(G,C)
        femstiffness!(S,G)
        for il in 1:celldim
            for jl in 1:celldim
                uij=u[cellnodes[jl,icell]]*u[cellnodes[il,icell]]*vol
                l2norm+=uij*M[jl,il]
                h1norm+=uij*S[jl,il]
            end
        end
    end
    return (sqrt(l2norm),sqrt(abs(h1norm)));
end


femnorms(g::ExtendableGrid,u)=femnorms(g[Coordinates], g[CellNodes],u)


Dirichlet()=1.0e30

function  femassemble!(A_h, # Global stiffness matrix
                       F_h, # Right hand side of FEM problem
                       coord,
                       cellnodes,
                       bfacenodes,
                       Λ,
                       f::Tf,
                       β::Tβ, # Boundary function
                       ::Type{Val{spacedim}}) where {Tf,Tβ,spacedim}
    celldim=spacedim+1
    S=@MMatrix zeros(celldim, celldim) # local stiffness matrix
    X=@MMatrix zeros(celldim, celldim) # local stiffness matrix
    C=@MMatrix zeros(spacedim,spacedim)	# local coordinate matrix
    G=@MMatrix zeros(celldim,spacedim)  # shape function grdients
    BC=@MMatrix zeros(spacedim,spacedim)     # local boundary coordinate matrix
    ncells=size(cellnodes,2)
    factdim=prod(1:spacedim)
    xcoord=reinterpret(reshape, SVector{spacedim,Float64},coord)
    for icell=1:ncells
        coordmatrix!(C,coord,cellnodes,icell)
        vol=abs(det(C))/factdim
        femgrad!(G,C)
        femstiffness!(S,G,Λ)
        for il=1:celldim
            ig=cellnodes[il,icell]
            for jl=1:celldim
                jg=cellnodes[jl,icell]
                A_h[ig,jg]+=S[il,jl]*vol
            end
	   F_h[ig]+=f(xcoord[ig])*vol/celldim
        end
    end    

    # Boundary part with penalty method
    nbfaces=size(bfacenodes,2)
    for ibface in 1:nbfaces
        for idim=1:spacedim
            i1=bfacenodes[idim,ibface];
            A_h[i1,i1]+=Dirichlet();
            F_h[i1]+=Dirichlet()*β(xcoord[i1])
        end
    end
end

function  femassemble!(A_h, # Global stiffness matrix
                       F_h, # Right hand side of FEM problem
                       grid::ExtendableGrid,
                       Λ,
                       f::Tf,
                       β::Tβ # Boundary function
                       ) where {Tf,Tβ}
    coord=grid[Coordinates]
    cellnodes=grid[CellNodes]
    bfacenodes=grid[BFaceNodes]
    femassemble!(A_h, F_h, coord, cellnodes, bfacenodes,Λ, f, β, Val{dim_space(grid)})
end


function femsolve(grid,Λ,f,β)
    # Initialize sparse matrix and right hand side
    n=num_nodes(grid)
    matrix=ExtendableSparseMatrix(n,n)
    rhs=zeros(n)
    # Call the assemble function.
    
    femassemble!(matrix,rhs,grid,Λ,f,β)
    solver=AMGSolver(matrix, param= (solver=(type="cg",),))
    solver\rhs
end 


