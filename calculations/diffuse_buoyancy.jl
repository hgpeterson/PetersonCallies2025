using JLD2
using Printf

include("../utils/derivatives.jl")
include("../utils/1D.jl")

"""
    x, z, b = diffuse_columns()

Solve the diffusion problem for buoyancy in a bowl geometry with ``H = 1 - x²``.

The state is saved periodically, including 1D profile of the flow at x = 0.5.
"""
function diffuse_columns()
    # grid in x
    nx = 2^10 + 1 # odd to make sure 0.5 is included
    x = range(0, 1, length=nx)

    # index of x = 0.5
    i_profile = findfirst(x .== 0.5)

    # depth function
    H = @. 1 - x^2
    Hx = -2*x
    θs = -atan.(Hx)

    # vertical grids
    nz = 2^10
    σ = chebyshev_nodes(nz)
    z = H*σ'

    # turbulent mixing coefficients
    ν = ones(nx, nz)
    κ = [1e-2 + exp(-(z[i, j] + H[i])/0.1) for i in 1:nx, j in 1:nz]

    # parameters
    ε = 2e-2
    f = 1
    μϱ = 1e-4
    α = 1/2
    Δt = 1e-4*μϱ/ε^2

    # build matrices
    LHS_b = []
    RHS_b = []
    rhs_b = []
    for i in 1:nx-1 # last column has H = 0, so no diffusion
        L, R, r = build_b(z[i, :], κ[i, :], μϱ, α, θs[i], ε, Δt)
        push!(LHS_b, lu(L))
        push!(RHS_b, R)
        push!(rhs_b, r)
    end
    LHS_inversion = build_LHS_inversion(z[i_profile, :], ν[i_profile, :], ε, θs[i_profile], α, f)
    LHS_inversion = lu(LHS_inversion)

    # initial condition
    b = zeros(nx, nz)

    # simulation length and save interval
    T = 4e-2*μϱ/ε^2
    t_save = 4e-3*μϱ/ε^2

    # run
    n_steps = Int(round(T/Δt))
    n_save = Int(round(t_save/Δt))
    for k in 1:n_steps
        # update b for each column
        for i in 1:nx-1 # last column has H = 0, so no diffusion
            # ldiv!(b[i, :], LHS_b[i], RHS_b[i]*b[i, :] + rhs_b[i])
            b[i, :] = LHS_b[i]\(RHS_b[i]*b[i, :] + rhs_b[i])
        end

        # invert and save periodically
        if mod(k, n_save) == 0
            # invert 1D model at x = 0.5
            u1D, v1D, w1D = invert1D(LHS_inversion, b[i_profile, :], θs[i_profile])

            # compute buoyancy gradient
            bx = compute_bx(b, x, σ, H, Hx)

            # save
            ofile = @sprintf("data/buoyancy_%1.1e.jld2", k*Δt)
            jldsave(ofile; x, z, b, bx, u1D, v1D, w1D, i_profile, t=k*Δt)
            @info "Saved '$ofile' for step $k of $n_steps"
        end
    end

    return x, z, b
end

"""
    bx = compute_bx(b, x, σ, H, Hx)

Compute horizontal gradient of buoynacy using the terrain-following formula ``bx = ∂_ξ b - σ H_x/H ∂_σ b ``.
"""
function compute_bx(b, x, σ, H, Hx)
    nx, nz = size(b)
    bx = zeros(nx, nz)

    for j in 1:nz
        # ∂ξ(b)
        bx[:, j] .+= differentiate(b[:, j], x)
    end
    for i in 1:nx
        # -σ*Hx/H ∂σ(b)
        bx[i, :] .-= σ*Hx[i]/H[i].*differentiate(b[i, :], σ)
    end
    # for H = 0
    bx[nx, :] .= 0

    return bx
end

# run
x, z, b = diffuse_columns()