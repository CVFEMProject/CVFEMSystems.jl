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

function femgrad!(G,C)
    spacedim=size(C,1)
    celldim=spacedim+1
    vol=abs(det(C))/prod(1:spacedim)
    G[1:spacedim,1:spacedim].=C\I
    for i=1:spacedim
        G[spacedim+1,i]=0.0
        for j=1:spacedim
            G[celldim,i]-=G[j,i]
        end
    end
    G,vol
end



function femfactors!(ω,e,G,vol,Λ,len)
    ne=length(e)
    for ie=1:ne
        @views e[ie]=-dot(G[len[1,ie],:],Λ*G[len[2,ie],:])*vol
    end
    nn=length(ω)
    vol/=nn
    for i=1:nn
        ω[i]=vol
    end
    ω,e
end



function femstiffness!(S,G,vol,Λ)
    celldim=size(S,1)
    spacedim=celldim-1
    for il=1:celldim
        @views ΛGil=Λ*G[il,:]*vol
        @views S[il,il]=dot(ΛGil,G[il,:])
        for jl=il+1:celldim
            S[il,jl]=dot(ΛGil,G[jl,:])
            S[jl,il]=S[il,jl]
        end
    end
    return S
end

function femstiffness!(S,G)
    celldim=size(S,1)
    spacedim=celldim-1
    for il=1:celldim
        @views S[il,il]=dot(G[il,:],G[il,:])
        for jl=il+1:celldim
            @views S[il,jl]=dot(G[il,:],G[jl,:])
            S[jl,il]=S[il,jl]
        end
    end
    return S
end


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
    for icell=1:ncells
	coordmatrix!(C,coord,cellnodes,icell)
        G,vol=femgrad!(G,C)
        femstiffness!(S,G)
        for i in 1:celldim
            for j in 1:celldim
                uij=u[cellnodes[j,icell]]*u[cellnodes[i,icell]]*vol
                l2norm+=uij*M[j,i]
                h1norm+=uij*S[j,i]
            end
        end
    end
    return (sqrt(l2norm),sqrt(abs(h1norm)));
end


femnorms(g::ExtendableGrid,u)=femnorms(g[Coordinates], g[CellNodes],u)


Dirichlet()=1.0e30

function  femassemble!(A_h, # Global stiffness matrix
                       F_h, # Right hand side of FEM problem
                       grid, # Discretization grid  
                       Λ,
                       f::Tf,
                       β::Tβ # Boundary function
                       ) where {Tf,Tβ}
    coord=grid[Coordinates]
    cellnodes=grid[CellNodes]
    bfacenodes=grid[BFaceNodes]
    spacedim=size(coord,1)
    celldim=spacedim+1
    S=zeros(nnodes, nnodes) # local stiffness matrix
    C=zeros(spacedim,spacedim)  # local coordinate matrix
    G=zeros(celldim,spacedim)  # shape function grdients
    BC=zeros(spacedim,spacedim)     # local boundary coordinate matrix
    ncells=size(cellnodes,2)
    for icell=1:ncells
        coordmatrix!(C,coord,cellnodes,icell)
        G,vol=femgrad!(G,C)
        femstiffness!(S,G,vol,Λ)
        for il=1:nnodes
            i=cellnodes[il,icell]
            for jl=1:nnodes
                j=cellnodes[jl,icell]
                A_h[i,j]+=S[il,jl]
            end
	    @views F_h[i]+=f(coord[:,cellnodes[il,icell]])*vol/celldim
        end
    end    
    
    # Boundary part with penalty method
    nbfaces=size(bfacenodes,2)
    bfaceregions=grid[BFaceRegions]
    for ibface in 1:nbfaces
        for idim=1:dim
            i1=bfacenodes[idim,ibface];
            A_h[i1,i1]+=Dirichlet();
            F_h[i1]+=Dirichlet()*β(coord[:,i1])
        end
    end
end

