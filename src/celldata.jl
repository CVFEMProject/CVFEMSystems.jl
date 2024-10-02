"""
    struct CellData

Data structure holding information for user cell assembly.
"""
mutable struct CellData{Tgc, Tgn, Ten, Tcm, Tsg, Ttm, Tuold}
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
    uold::Tuold
    time::Float64
    tstep::Float64
end

"""
    CellData(coord,edgenodes)

Constructor for cell data from coordinate array and nodes per edge matrix.
"""
function CellData(coord, nspec,time, tstep)
    spacedim = size(coord, 1)
    globalcoord = reinterpret(reshape, SVector{spacedim, Float64}, coord)
    celldim = spacedim + 1
    icell = 0
    region = 0
    edgenodes=local_edgenodes[spacedim]
    xedgenodes = @MMatrix zeros(Int64, 2, size(edgenodes, 2))
    xedgenodes .= edgenodes
    globalnodes = @MVector zeros(Int64, celldim)
    factdim = prod(1:spacedim)
    coordmatrix = @MMatrix zeros(spacedim, spacedim)
    shapegradients = @MMatrix zeros(celldim, spacedim)
    transmission = @MVector zeros(size(edgenodes, 2)) # Transmission coefficients
    uold= @MMatrix zeros(nspec,celldim)
    CellData(icell,
             region,
             globalcoord,
             globalnodes,
             xedgenodes,
             factdim,
             coordmatrix,
             shapegradients,
             0.0,
             transmission,
             uold,
             Float64(time),
             Float64(tstep))
end

"""
    nnodes(celldata)

Number of nodes of cell. To be used in user callback.
"""
nnodes(celldata::CellData) = size(celldata.shapegradients, 1)

"""
    edges(celldata)

Number of nodes of cell. To be used in user callback.
"""
nedges(celldata::CellData) = size(celldata.edgenodes, 2)

"""
    spacedim(celldata)

Space dimension. To be used in user callback.
"""
spacedim(celldata::CellData) = size(celldata.shapegradients, 2)

"""
    coord(celldata,i)

Coordinates of i-th node. To be used in user callback.
"""
coord(celldata::CellData, il) = celldata.globalcoord[celldata.globalnodes[il]]

"""
   edgenode(celldata,inode,iedge)

Local number of node of edge.
"""
edgenode(celldata::CellData, i, ie) = celldata.edgenodes[i, ie]

"""
    globalnode(celldata, i)

Global (grid) node number of cell node.
"""
globalnode(celldata::CellData, il) = celldata.globalnodes[il]

"""
    volume(celldata)

Return volume (volume,area, length) of cell.
"""
volume(celldata::CellData) = celldata.volume

"""
    nodevolume(celldata)

Return node contribution  (volume,area, length) of cell.
"""
nodevolume(celldata::CellData) = volume(celldata)/nnodes(celldata)

"""
        transmission(celldata, Λ)

Calculate and edge transmission coefficients in an array of length equal
to number of cell edges.
"""
function transmission(celldata, Λ)
    femfactors!(celldata.transmission, celldata.shapegradients, Λ, celldata.edgenodes)
    for ie = 1:nedges(celldata)
        celldata.transmission[ie] *= celldata.volume
    end
    celldata.transmission
end

"""
    update!(celldata,coord, cellnodes, icell)

Update cell data for current cell.
"""
function update!(celldata::CellData, coord, cellnodes, icell)
    celldata.icell = icell
    for i = 1:nnodes(celldata)
        celldata.globalnodes[i] = cellnodes[i, icell]
    end
    coordmatrix!(celldata.coordmatrix, coord, cellnodes, icell)
    celldata.volume = abs(det(celldata.coordmatrix)) / celldata.factdim
    femgrad!(celldata.shapegradients, celldata.coordmatrix)
end
