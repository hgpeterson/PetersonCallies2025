using LinearAlgebra
using SparseArrays
using JLD2

include("derivatives.jl")

"""
    x = chebyshev_nodes(n)

Return `n` Chebyshev nodes in the interval `[-1, 0]`.
"""
function chebyshev_nodes(n)
    return ([-cos((i - 1)*π/(n - 1)) for i ∈ 1:n] .- 1)/2
end

"""
    LHS, RHS, rhs = build_b(z, κ, μϱ, α, θ, ε, Δt; horiz_diff=true)

Build the matrix representation of the diffusion equation for buoyancy.

The equation is given by

``bⁿ⁺¹ - a ∂_z [κ (1 + Γ ∂_z bⁿ⁺¹)] = bⁿ + a ∂_z[κ (1 + Γ ∂_z bⁿ)]``

where `a = ε²/μϱ * Δt/2` and `Γ = 1 + α^2*tan(θ)^2`. We discretize using finite
differences and write as `LHS*bⁿ⁺¹ = RHS*bⁿ + rhs`. `LHS` and `RHS` are 
returned as sparse matrices while `rhs` is a dense vector.

# Arguments
- `z`: vector of vertical grid points
- `κ`: vector of diffusivity at each grid point
- `μϱ`: Prandtl times Burger number
- `α`: aspect ratio
- `θ`: slope angle
- `ε`: Ekman number
- `Δt`: time step size
- `horiz_diff=false`: boolean to determine if horizontal diffusion is used.
If `true`, `Γ = 1 + α^2*tan(θ)^2`. Otherwise, `Γ = 1`.
"""
function build_b(z, κ, μϱ, α, θ, ε, Δt; horiz_diff=false)
    # coeffs
    a = ε^2/μϱ * Δt/2
    if horiz_diff 
        Γ = 1 + α^2*tan(θ)^2
    else
        Γ = 1
    end

    # initialize
    N = length(z)
    LHS = Tuple{Int64,Int64,Float64}[] 
    RHS = Tuple{Int64,Int64,Float64}[]
    rhs = zeros(N)

    # interior nodes 
    for j=2:N-1
        # dz stencil
        fd_z = mkfdstencil(z[j-1:j+1], z[j], 1)

        # dz(κ)
        κ_z = fd_z[1]*κ[j-1] + fd_z[2]*κ[j] + fd_z[3]*κ[j+1]

        # dzz stencil
        fd_zz = mkfdstencil(z[j-1:j+1], z[j], 2)

        # LHS: b - a*dz(κ*(1 + Γ*dz(b)))
        #    = b - a*dz(κ + Γ*κ*dz(b))
        #    = b - a*dz(κ) - a*Γ*dz(κ)*dz(b) - a*Γ*κ*dzz(b)
        push!(LHS, (j, j, 1))
        push!(LHS, (j, j-1, (-a*Γ*κ_z*fd_z[1] - a*Γ*κ[j]*fd_zz[1])))
        push!(LHS, (j, j,   (-a*Γ*κ_z*fd_z[2] - a*Γ*κ[j]*fd_zz[2])))
        push!(LHS, (j, j+1, (-a*Γ*κ_z*fd_z[3] - a*Γ*κ[j]*fd_zz[3])))
        rhs[j] += a*κ_z # -α*dz(κ) move to rhs

        # RHS: b + a*dz(κ*(1 + Γ*dz(b)))
        #    = b + a*dz(κ + Γ*κ*dz(b))
        #    = b + a*dz(κ) + a*Γ*dz(κ)*dz(b) + a*Γ*κ*dzz(b)
        push!(RHS, (j, j, 1))
        push!(RHS, (j, j-1, (a*Γ*κ_z*fd_z[1] + a*Γ*κ[j]*fd_zz[1])))
        push!(RHS, (j, j,   (a*Γ*κ_z*fd_z[2] + a*Γ*κ[j]*fd_zz[2])))
        push!(RHS, (j, j+1, (a*Γ*κ_z*fd_z[3] + a*Γ*κ[j]*fd_zz[3])))
        rhs[j] += a*κ_z
    end

    # z = -H: 1 + Γ*dz(b) = 0 -> dz(b) = -1/Γ
    fd_z = mkfdstencil(z[1:3], z[1], 1)
    push!(LHS, (1, 1, fd_z[1]))
    push!(LHS, (1, 2, fd_z[2]))
    push!(LHS, (1, 3, fd_z[3]))
    rhs[1] = -1/Γ

    # z = 0: b = 0
    push!(LHS, (N, N, 1))

    # Create CSC sparse matrices
    LHS = sparse((x->x[1]).(LHS), (x->x[2]).(LHS), (x->x[3]).(LHS), N, N)
    RHS = sparse((x->x[1]).(RHS), (x->x[2]).(RHS), (x->x[3]).(RHS), N, N)

    return LHS, RHS, rhs
end

"""
    LHS = build_LHS_inversion(z, ν, ε, θ, α, f; no_Px=false)

Build the matrix representation of the inversion equations for the 1D flow.

The inversion is written as

``-ε²Γ² ∂_z(ν ∂_z u) - f v + P_x = b tan(θ)``

``-ε²Γ  ∂_z(ν ∂_z v) + f u       = 0``

where `Γ = 1 + α^2*tan(θ)^2`. Boundary conditions:

``∂_z u = ∂_z v = 0`` at ``z = 0``

``u = v = 0`` at ``z = -H``

``∫ u dz = 0`` or ``P_x = 0`` depending on `no_Px`

We discretize using finite differences and write as `LHS * [u; v; Px] = b`.

# Arguments
- `z`: vector of vertical grid points
- `ν`: vector of viscosity at each grid point
- `ε`: Ekman number
- `θ`: slope angle
- `α`: aspect ratio
- `f`: Coriolis parameter
- `no_Px=true`: boolean to determine boundary condition. If `true` (default), ``P_x = 0``. Otherwise, ``∫ u dz = 0``.
"""
function build_LHS_inversion(z, ν, ε, θ, α, f; no_Px=false)
    # setup
    nz = length(z)
    umap = 1:nz
    vmap = nz+1:2nz
    iPx = 2nz + 1
    Γ = 1 + α^2*tan(θ)^2
    LHS = Tuple{Int64,Int64,Float64}[]  

    # interior nodes
    for j ∈ 2:nz-1
        # dz stencil
        fd_z = mkfdstencil(z[j-1:j+1], z[j], 1)
        ν_z = sum(fd_z.*ν[j-1:j+1])

        # dzz stencil
        fd_zz = mkfdstencil(z[j-1:j+1], z[j], 2)
        
        # eq 1: -ε²Γ²*dz(ν*dz(u)) - f*v + Px = b*tan(θ)
        # term 1 = -ε²Γ²*[dz(ν)*dz(u) + ν*dzz(u)] 
        c = ε^2*Γ^2
        push!(LHS, (umap[j], umap[j-1], -c*(ν_z*fd_z[1] + ν[j]*fd_zz[1])))
        push!(LHS, (umap[j], umap[j],   -c*(ν_z*fd_z[2] + ν[j]*fd_zz[2])))
        push!(LHS, (umap[j], umap[j+1], -c*(ν_z*fd_z[3] + ν[j]*fd_zz[3])))
        # term 2 = -f*v
        push!(LHS, (umap[j], vmap[j], -f))
        # term 3 = Px
        push!(LHS, (umap[j], iPx, 1))

        # eq 2: -ε²Γ *dz(ν*dz(v)) + f*u = 0
        # term 1 = -ε²Γ*[dz(ν)*dz(v) + ν*dzz(v)]
        c = ε^2*Γ
        push!(LHS, (vmap[j], vmap[j-1], -c*(ν_z*fd_z[1] + ν[j]*fd_zz[1])))
        push!(LHS, (vmap[j], vmap[j],   -c*(ν_z*fd_z[2] + ν[j]*fd_zz[2])))
        push!(LHS, (vmap[j], vmap[j+1], -c*(ν_z*fd_z[3] + ν[j]*fd_zz[3])))
        # term 2 = f*u
        push!(LHS, (vmap[j], umap[j], f))
    end

    # bottom boundary conditions: u = v = 0
    push!(LHS, (umap[1], umap[1], 1))
    push!(LHS, (vmap[1], vmap[1], 1))

    # surface boundary conditions: dz(u) = dz(v) = 0
    fd_z = mkfdstencil(z[end-2:end], z[end], 1)
    push!(LHS, (umap[end], umap[end-2], fd_z[1]))
    push!(LHS, (umap[end], umap[end-1], fd_z[2]))
    push!(LHS, (umap[end], umap[end],   fd_z[3]))
    push!(LHS, (vmap[end], vmap[end-2], fd_z[1]))
    push!(LHS, (vmap[end], vmap[end-1], fd_z[2]))
    push!(LHS, (vmap[end], vmap[end],   fd_z[3]))

    # last degree of freedom: ∫ u dz = 0 or Px = 0
    if no_Px
        push!(LHS, (iPx, iPx, 1))
    else
        for j in 1:nz-1
            # trapezoidal rule
            dz = z[j+1] - z[j]
            push!(LHS, (iPx, umap[j],   dz/2))
            push!(LHS, (iPx, umap[j+1], dz/2))
        end
    end

    # Create CSC sparse matrix from matrix elements
    LHS = sparse((x->x[1]).(LHS), (x->x[2]).(LHS), (x->x[3]).(LHS), 2nz+1, 2nz+1)

    return LHS
end

"""
    u, v, w = invert1D(LHS, b, θ)

Invert for flow from 1D model.

# Arguments
- `LHS`: sparse matrix representation of left-hand side of the inversion
- `b`: vector of buoyancy at each grid point
- `θ`: slope angle

See also [`build_LHS_inversion`](@ref).
"""
function invert1D(LHS, b, θ)
    nz = length(b)
    rhs = zeros(2nz + 1)
    rhs[2:nz-1] .= b[2:nz-1]*tan(θ)
    sol = LHS\rhs
    return sol[1:nz], sol[nz+1:2nz], sol[1:nz]*tan(θ)
end