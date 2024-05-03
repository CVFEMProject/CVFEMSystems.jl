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

function femstiffness(Λ,coord, cellnodes,icell)
    spacedim=size(coord,1)
    celldim=size(cellnodes,1)
    C=zeros(spacedim,spacedim)
    G=zeros(celldim,spacedim)
    S=zeros(celldim,celldim)
    coordmatrix!(C, coord, cellnodes,1)
    G,vol=femgrad!(G,C)
    femstiffness!(S,G,vol,Λ)
    S
end
