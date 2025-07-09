using SparseArrays

include("derivatives.jl")

function bl_spiral(z, q, c1, c2)
    pm = z[end] > 0 ? -1 : +1
    return @. exp(pm*q*z)*(c1*cos(q*z) + c2*sin(q*z))
end

"""
   u, v = solve_baroclinic_problem(; ε, z, ν, f, bx, by, U, V=nothing, τx, τy, vz_bot=nothing)
   
Solve the baroclinic problem for the zonal and meridional flow.

The baroclinic problem is given by the equations

``-f τʸ - ε² ∂_{zz}(ν τˣ) = -b_x``
             
``+f τˣ - ε² ∂_{zz}(ν τʸ) = -b_y``

with boundary conditions

    ``τˣ(z=0) = `τx```

    ``τʸ(z=0) = `τy```

    ``∫ z τˣ dz = -U``

    ``∫ z τʸ dz = -V`` or ``τʸ(z=-H) = `vz_bot```

We use finite differences on a discrete grid `z` to turn this into a matrix
problem. After the solution is computed, we integrate in the vertical to 
obtain the zonal and meridional flow profiles `u` and `v`.

# Arguments
- `ε`: Ekman number
- `z`: vector of vertical grid points
- `ν`: vector of vertical viscosity profile
- `f`: Coriolis parameter
- `bx`: vector of x-derivative of buoyancy
- `by`: vector of y-derivative of buoyancy
- `U`: vertical integral of u
- `V=nothing`: vertical integral of v (optional)
- `τx`: value of τx at the surface
- `τy`: value of τy at the surface
- `vz_bot=nothing`: value of ``∂_z v`` at the bottom
"""
function solve_baroclinic_problem(; ε, z, ν, f, bx, by, U, V=nothing, τx, τy, vz_bot=nothing)
    if V === nothing && vz_bot === nothing
        throw(ArgumentError("Either V or vz_bot must be provided."))
    end

    nz = length(z)

    ν_z  = differentiate(ν, z)
    ν_zz = differentiate(ν_z, z)

    A = Tuple{Int64, Int64, Float64}[]
    r = zeros(2nz)
    imap = reshape(1:2nz, 2, :)
    for i ∈ 2:nz-1
        fd_z  = mkfdstencil(z[i-1:i+1], z[i], 1)
        fd_zz = mkfdstencil(z[i-1:i+1], z[i], 2)

        ## -f τy - ε² (ν τx)_zz = -b_x

        # -f τy
        push!(A, (imap[1, i], imap[2, i], -f))
        # -2 ε² ν_z τx_z
        push!(A, (imap[1, i], imap[1, i-1], -2ε^2*ν_z[i]*fd_z[1]))
        push!(A, (imap[1, i], imap[1, i  ], -2ε^2*ν_z[i]*fd_z[2]))
        push!(A, (imap[1, i], imap[1, i+1], -2ε^2*ν_z[i]*fd_z[3]))
        # -ε² ν τx_zz
        push!(A, (imap[1, i], imap[1, i-1], -ε^2*ν[i]*fd_zz[1]))
        push!(A, (imap[1, i], imap[1, i  ], -ε^2*ν[i]*fd_zz[2]))
        push!(A, (imap[1, i], imap[1, i+1], -ε^2*ν[i]*fd_zz[3]))
        # -ε² ν_zz τx
        push!(A, (imap[1, i], imap[1, i], -ε^2*ν_zz[i]))
        # -b_x
        r[imap[1, i]] = -bx[i]

        ## f τx - ε² (ν τy)_zz = -b_y

        # f τx
        push!(A, (imap[2, i], imap[1, i], f))
        # -2 ε² ν_z τy_z
        push!(A, (imap[2, i], imap[2, i-1], -2ε^2*ν_z[i]*fd_z[1]))
        push!(A, (imap[2, i], imap[2, i  ], -2ε^2*ν_z[i]*fd_z[2]))
        push!(A, (imap[2, i], imap[2, i+1], -2ε^2*ν_z[i]*fd_z[3]))
        # -ε² ν τy_zz
        push!(A, (imap[2, i], imap[2, i-1], -ε^2*ν[i]*fd_zz[1]))
        push!(A, (imap[2, i], imap[2, i  ], -ε^2*ν[i]*fd_zz[2]))
        push!(A, (imap[2, i], imap[2, i+1], -ε^2*ν[i]*fd_zz[3]))
        # -ε² ν_zz τy
        push!(A, (imap[2, i], imap[2, i], -ε^2*ν_zz[i]))
        # -b_x
        r[imap[2, i]] = -by[i]
    end

    # boundary conditions
    push!(A, (imap[1, nz], imap[1, nz], ε^2*ν[nz]))
    r[imap[1, nz]] = τx
    push!(A, (imap[2, nz], imap[2, nz], ε^2*ν[nz]))
    r[imap[2, nz]] = τy
    for i ∈ 1:nz-1
        push!(A, (imap[1, 1], imap[1, i],   z[i]*(z[i+1] - z[i])/2))
        push!(A, (imap[1, 1], imap[1, i+1], z[i]*(z[i+1] - z[i])/2))
        if V !== nothing
            push!(A, (imap[2, 1], imap[2, i],   z[i]*(z[i+1] - z[i])/2))
            push!(A, (imap[2, 1], imap[2, i+1], z[i]*(z[i+1] - z[i])/2))
        end
    end
    r[imap[1, 1]] = -U
    if V !== nothing
        r[imap[2, 1]] = -V
    else
        push!(A, (imap[2, 1], imap[2, 1], 1))
        r[imap[2, 1]] = vz_bot
    end

    # solve
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), 2nz, 2nz)
    sol = A\r
    τx = sol[imap[1, :]]
    τy = sol[imap[2, :]]

    # integrate to get u and v
    return cumtrapz(τx, z), cumtrapz(τy, z)
end

"""
    uBL, vBL, wBL = solve_baroclinic_problem_BL(; ε, z, ν, f, β, bx, by, U, V, τx, τy, Hx, Hy, α=1/2)

Compute boundary layer (BL) solution to the baroclinic problem.

Here `β` is the meridional gradient of the Coriolis parameter, `Hx` and `Hy` are 
the bottom slopes in the x- and y-direction, respectively, and `α` is the aspect ratio.

See also [`solve_baroclinic_problem`](@ref) and [`solve_baroclinic_problem_BL_U0`](@ref).
"""
function solve_baroclinic_problem_BL(; ε, z, ν, f, β, bx, by, U, V, τx, τy, Hx, Hy, α=1/2)
    H = -z[1]

    # surface BL coords
    z_s = z/ε
    q_s = sqrt(f/2/ν[end])

    # bottom BL coords
    z_b = (z .+ H)/ε
    q_b = sqrt(f/2/ν[1])*(1 + α^2*Hx^2)^(-3/4)

    # interior O(1)
    wI0_sfc = 0 # for now
    uI0_bot = U/H - 1/(H*f)*trapz(z.*by, z) - τy/(H*f)
    vI0_bot = V/H + 1/(H*f)*trapz(z.*bx, z) + τx/(H*f)
    uI0 = uI0_bot .- 1/f*cumtrapz(by, z)
    vI0 = vI0_bot .+ 1/f*cumtrapz(bx, z)
    wI0 = wI0_sfc .- β/f*(trapz(vI0, z) .- cumtrapz(vI0, z))

    # bottom BL O(1)
    c1 = -uI0_bot
    c2 = -vI0_bot
    uB0_b = bl_spiral(z_b, q_b, c1,  c2)
    vB0_b = bl_spiral(z_b, q_b, c2, -c1)
    wB0_b = -Hx*uB0_b - Hy*vB0_b

    # surface BL O(1/ε)
    c1 = (τx + τy)/(2ν[end]*q_s)
    c2 = (τx - τy)/(2ν[end]*q_s)
    uBm1_s = bl_spiral(z_s, q_s,  c1, c2)
    vBm1_s = bl_spiral(z_s, q_s, -c2, c1)

    # interior O(ε)
    uI1 =  (uI0_bot + vI0_bot)/(2H*q_b)
    vI1 = -(uI0_bot - vI0_bot)/(2H*q_b)
    wI1 = -β/f*(H*vI1 .- cumtrapz(vI1*ones(length(z)), z))

    # bottom BL O(ε)
    c1 = -uI1
    c2 = -vI1
    uB1_b = bl_spiral(z_b, q_b, c1,  c2)
    vB1_b = bl_spiral(z_b, q_b, c2, -c1)
    wB1_b = -Hx*uB1_b - Hy*vB1_b

    # surface BL O(ε)
    c1 = -(bx[end] - by[end])/(2*f*q_s)
    c2 = +(bx[end] + by[end])/(2*f*q_s)
    uB1_s = bl_spiral(z_s, q_s,  c1, c2)
    vB1_s = bl_spiral(z_s, q_s, -c2, c1)

    # interior O(ε²)
    uI2_bot = ν[1]/H/f^2*bx[1] + 1/f^2*differentiate_pointwise(ν[1:3].*bx[1:3], z[1:3], z[1], 1) + (uI1 + vI1)/(2q_b*H)
    vI2_bot = ν[1]/H/f^2*by[1] + 1/f^2*differentiate_pointwise(ν[1:3].*by[1:3], z[1:3], z[1], 1) - (uI1 - vI1)/(2q_b*H)
    uI2 = uI2_bot .+ 1/f^2*differentiate(ν.*bx, z) .- 1/f^2*differentiate_pointwise(ν[1:3].*bx[1:3], z[1:3], z[1], 1)
    vI2 = vI2_bot .+ 1/f^2*differentiate(ν.*by, z) .- 1/f^2*differentiate_pointwise(ν[1:3].*by[1:3], z[1:3], z[1], 1)
    # wI2 = -β/f*(trapz(vI2, z) .- cumtrapz(vI2, z)) # this is missing the ∂z(ν ∂z(ζ)) term

    # bottom BL O(ε²)
    c1 = -uI2_bot
    c2 = -vI2_bot
    uB2_b = bl_spiral(z_b, q_b, c1,  c2)
    vB2_b = bl_spiral(z_b, q_b, c2, -c1)
    # wB2_b = -Hx*uB2_b - Hy*vB2_b

    # total
    uBL = 1/ε*uBm1_s .+ uI0 .+ uB0_b .+ ε*(uI1 .+ uB1_b .+ uB1_s) .+ ε^2*(uI2 .+ uB2_b)
    vBL = 1/ε*vBm1_s .+ vI0 .+ vB0_b .+ ε*(vI1 .+ vB1_b .+ vB1_s) .+ ε^2*(vI2 .+ vB2_b)
    wBL =               wI0 .+ wB0_b .+ ε*(wI1 .+ wB1_b)          #.+ ε^2*(wI2 .+ wB2_b)

    return uBL, vBL, wBL
end

"""
    uBL, vBL, wBL = solve_baroclinic_problem_BL_U0(; ε, z, ν, f, bx, Hx, α=1/2)

Compute boundary layer (BL) solution to the baroclinic problem for the special case of `U = 0`.

See also [`solve_baroclinic_problem`](@ref) and [`solve_baroclinic_problem_BL_U0`](@ref).
"""
function solve_baroclinic_problem_BL_U0(; ε, z, ν, f, bx, Hx, α=1/2)
    H = -z[1]
    q_b = sqrt(f/2/ν[1])
    Γ = 1 + α^2*Hx^2
    q_b *= Γ^(-3/4)
    z_b = (z .+ H)/ε

    # interior O(1)
    # uI0 = 0
    vI0 = 1/f*cumtrapz(bx, z)
    # wI0 = 0

    # interior O(ε)
    # uI1 = 0
    vI1 = -1/(f*q_b)*bx[1]
    # wI1 = 0

    # bottom BL O(ε)
    c1 = 0
    c2 = -vI1/√Γ
    uB1 =    bl_spiral(z_b, q_b, c1,  c2)
    vB1 = √Γ*bl_spiral(z_b, q_b, c2, -c1)
    wB1 = -Hx*uB1

    # interior O(ε²)
    uI2 =  Γ/f^2*differentiate(ν.*bx, z)
    vI2 = -uI2[1]*√Γ
    wI2 = -Hx*uI2

    # bottom BL O(ε²)
    c1 = -uI2[1]
    c2 = -vI2/√Γ
    uB2 =    bl_spiral(z_b, q_b, c1,  c2)
    vB2 = √Γ*bl_spiral(z_b, q_b, c2, -c1)
    wB2 = -Hx*uB2

    # total
    uBL = ε*uB1 + ε^2*(uI2 .+ uB2)
    vBL = vI0 .+ ε*(vI1 .+ vB1) .+ ε^2*(vI2 .+ vB2)
    wBL = ε*wB1 + ε^2*(wI2 .+ wB2)

    return uBL, vBL, wBL
end