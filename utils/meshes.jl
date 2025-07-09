using Gmsh: gmsh

"""
    p, t = get_p_t(fname)

Return the node coordinates `p` and the connectivities `t` of a mesh.

`p[i, j]` is the `j`-th coordinate of node `i`, while `t[i, :]` contains the
connectivities of triangle `i` (i.e., the indices of the nodes that form the
triangle). The mesh is loaded from the file `fname`, which should be in GMSH 
format.
"""
function get_p_t(fname)
    # load model
    gmsh.initialize()
    gmsh.open(fname)

    # get nodes
    _, coords, _ = gmsh.model.mesh.getNodes()
    p = reshape(coords, 3, length(coords) ÷ 3)
    p = Matrix(p')

    # get connectivities
    _, _, t = gmsh.model.mesh.getElements(2)
    t = Int.(t[1])
    t = reshape(t, 3, length(t) ÷ 3)
    t = Matrix(t')

    gmsh.finalize()

    return p, t
end