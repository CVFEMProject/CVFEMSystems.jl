
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

function femgrad(coord, cellnodes,icell)
    spacedim=size(coord,1)
    celldim=size(cellnodes,1)
    C=zeros(spacedim,spacedim)
    G=zeros(celldim,spacedim)
    coordmatrix!(C, coord, cellnodes,1)
    femgrad!(G,C)
end


function extended_femgrad(C)
    celldim=size(C,1)
    spacedim=celldim-1
    vol=abs(det(C))/prod(1:spacedim)
    G=view(C\I,:,2:celldim)
    G,vol
end

function extended_coordmatrix!(C,coord, cellnodes,icell)
    spacedim=size(coord,1)
    celldim=size(cellnodes,1)
    for jj=1:celldim
        C[1,jj]=1
        for ii=1:spacedim
            C[ii+1,jj]=coord[ii,cellnodes[jj,icell]]
        end
    end
end

function extended_femgrad(coord, cellnodes,icell)
    spacedim=size(coord,1)
    celldim=size(cellnodes,1)
    C=zeros(celldim,celldim)
    extended_coordmatrix!(C, coord, cellnodes,1)
    extended_femgrad(C)
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

const len=[
    local_celledgenodes(Edge1D),
    local_celledgenodes(Triangle2D),
    local_celledgenodes(Tetrahedron3D),
]

function femfactors(Λ,coord, cellnodes,icell)
    spacedim=size(coord,1)
    celldim=size(cellnodes,1)
    C=zeros(spacedim,spacedim)
    G=zeros(celldim,spacedim)
    ω=zeros(celldim)
    e=zeros(size(len[spacedim],2))
    coordmatrix!(C, coord, cellnodes,1)
    G,vol=femgrad!(G,C)
    femfactors!(ω,e,G,vol,Λ,len[spacedim])
    ω,e
end


function femstiffness!(S,G,vol,Λ,len)
    celldim=size(S,1)
    spacedim=celldim-1
    for il=1:celldim
        ΛGil=Λ*G[il,:]*vol
        S[il,il]=dot(ΛGil,G[il,:])
        for jl=il+1:celldim
            S[il,jl]=dot(ΛGil,G[jl,:])
            S[jl,il]=S[il,jl]
        end
    end
    return S
end

function femstiffness(Λ,coord, cellnodes,icell)
    spacedim=size(coord,1)
    celldim=size(cellnodes,1)
    C=zeros(spacedim,spacedim)
    G=zeros(celldim,spacedim)
    S=zeros(celldim,celldim)
    coordmatrix!(C, coord, cellnodes,1)
    G,vol=femgrad!(G,C)
    femstiffness!(S,G,vol,Λ,len[spacedim])
    S
end
