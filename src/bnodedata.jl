
"""
    struct BNodeData

Data structure holding information for user boundary condition assembly.
"""
mutable struct BNodeData{Tgc}
    ibface::Int64
    region::Int64
    globalcoord::Tgc
    globalnode::Int64
    volume::Float64
end

"""
    BNodeData(coord)

Constructor for BNodeData
"""
function BNodeData(coord)
    spacedim = size(coord, 1)
    globalcoord = reinterpret(reshape, SVector{spacedim, Float64}, coord)
    ibface = 0
    region = 0
    globalnode = 0
    BNodeData(ibface,
              region,
              globalcoord,
              globalnode,
              0.0)
end

"""
    volume(bnodedata)


Volume (area, length, 1)  corresponding to bondary node.
"""
volume(bnodedata::BNodeData) = bnodedata.nodevolume

"""
    coord(bonodedata)

Coordinates of boundary node.
"""
coord(bnodedata::BNodeData) = bnodedata.globalcoord[bnodedata.globalnode]

"""
    globalnode(bnodedata)

Global node number of boundary node.
"""
globalnode(bnodedata::BNodeData) = bnodedata.globalnode

function update!(bnodedata::BNodeData, coordinates, bfacenodes, ibface, inode, volume)
    bnodedata.ibface = ibface
    bnodedata.globalnode = bfacenodes[inode, ibface]
    bnodedata.volume = volume
end

"""
    dirichlet!(bnodedata,y,u,value; ispec)

Set Dirichlet value for bnodedata.
"""
function dirichlet!(bnodedata, y, u, val; ispec=1)
    y[ispec,1] = Dirichlet() * (u[ispec,1] - val)
end
