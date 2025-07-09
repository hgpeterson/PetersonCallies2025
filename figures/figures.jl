using PyPlot
using PyCall
using JLD2
using Printf

# full width (39 picas)
# just under full width (33 picas)
# two-thirds page width (27 picas)
# single column width (19 picas)
pc = 1/6 # pica

pl = pyimport("matplotlib.pylab")
cm = pyimport("matplotlib.cm")
colors = pyimport("matplotlib.colors")

pygui(false)
plt.style.use("plots.mplstyle")
plt.close("all")

include("../utils/derivatives.jl")
include("../utils/integrals.jl")
include("../utils/baroclinic.jl")
include("../utils/handle_nans.jl")
include("../utils/meshes.jl")

"""
    buoyancy()

Generate Figure 1 of the manuscript.
"""
function buoyancy()
    width = 39
    s = 4 # width of spacing
    c = 1 # width of colorbar
    p = (width - s - c)/(3 + 1.62) # width of profile plot (assuming b = height = 1.62p)
    height = p*1.62
    b = height # width of bowl plot
    fig, ax = plt.subplots(1, 6, figsize=(width*pc, height*pc), gridspec_kw=Dict("width_ratios"=>[p, p, p, s, b, c]))

    # (a) buoyancy profiles
    ax[1].annotate("(a)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[1].set_xlabel(L"Stratification $\partial_z b$")
    ax[1].set_ylabel(L"Vertical coordinate $z$")
    ax[1].set_xlim(0, 1.3)
    ax[1].set_yticks(-0.75:0.25:0)
    ts = 1e-3:1e-3:1e-2
    colors = pl.cm.BuPu_r(range(0, 0.7, length=length(ts)))
    for i in eachindex(ts)
        file = jldopen(@sprintf("data/buoyancy_%1.1e.jld2", ts[i]), "r")
        b = file["b"]
        z = file["z"]
        i_profile = file["i_profile"]
        close(file)
        b = b[i_profile, :]
        z = z[i_profile, :]
        bz = differentiate(b, z)
        ax[1].plot(1 .+ bz, z, c=colors[i, :])
    end
    ax[1].annotate("", xy=(0.6, -0.57), xytext=(0.42, -0.42), arrowprops=Dict("color"=>"k", "arrowstyle"=>"-|>"))
    ax[1].annotate(L"$t = 10^{-2}$", xy=(0.15, -0.4))

    # (b) u profile
    file = jldopen("data/buoyancy_1.0e-02.jld2", "r")
    z = file["z"]
    i_profile = file["i_profile"]
    u = file["u1D"]
    v = file["v1D"]
    close(file)
    z = z[i_profile, :]
    ax[2].annotate("(b)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[2].set_xlabel("Cross-slope flow\n"*L"$u$ ($\times 10^{-2}$)")
    ax[2].set_xlim(-0.4, 1.1)
    ax[2].set_yticks([])
    ax[2].spines["left"].set_visible(false)
    ax[2].axvline(0, color="k", lw=0.5)
    ax[2].plot(1e2*u, z, c=colors[end, :], ls="-")
    ax[2].annotate(L"$\int_{-H}^0 \, u \; dz = 0$", xy=(0.2, -0.2))

    # (c) v profile
    ax[3].annotate("(c)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[3].set_xlabel("Along-slope flow\n"*L"$v$ ($\times 10^{-2}$)")
    ax[3].set_xlim(-3, 13)
    ax[3].set_yticks([])
    ax[3].spines["left"].set_visible(false)
    ax[3].axvline(0, color="k", lw=0.5)
    ax[3].axvline(1e2*v[end], color="C1", lw=0.5, ls="--")
    ax[3].plot(1e2*v, z, c=colors[end, :], ls="-")
    ax[3].annotate(L"$\partial_x P$", xy=(9, -0.7), color="C1")
    ax[3].annotate(L"$\partial_z v = 0$", xy=(0, -0.75), xytext=(-8, -0.65), arrowprops=Dict("color"=>"k", "arrowstyle"=>"-|>"))

    # spacing
    ax[4].axis("off")

    # (d) buoyancy gradient in bowl
    ax[5].annotate("(d)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[5].set_xlabel(L"Zonal coordinate $x$")
    ax[5].set_ylabel(L"Vertical coordinate $z$")
    ax[5].axis("equal")
    ax[5].set_xticks([0, 0.5, 1])
    ax[5].set_yticks([-1, -0.5, 0])
    ax[5].spines["left"].set_visible(false)
    ax[5].spines["bottom"].set_visible(false)
    d = jldopen("data/buoyancy_1.0e-02.jld2")
    x = d["x"]
    z = d["z"]
    b = d["b"]
    bx = d["bx"]
    close(d)
    xx = repeat(x, 1, size(z, 2))
    vmax = 2
    img = ax[5].pcolormesh(xx, z, bx, shading="gouraud", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=true)
    ax[5].contour(xx, z, z .+ b, colors="k", linewidths=0.5, alpha=0.3, linestyles="-", levels=-0.9:0.1:-0.1)
    ax[5].plot([0.5, 0.5], [-0.75, 0.0], "r-", alpha=0.7)

    # colorbar
    ax[6].axis("off")
    cb = fig.colorbar(img, ax=ax[6], label=L"Buoyancy gradient $\partial_x b$", extend="max", fraction=0.5)
    cb.set_ticks([-2, 0, 2])

    savefig("figures/buoyancy.png")
    @info "Saved 'figures/buoyancy.png'"
    savefig("figures/buoyancy.pdf")
    @info "Saved 'figures/buoyancy.pdf'"
    plt.close()
end

function f_over_H_map()
    # load filtered data
    d = jldopen("data/topo_25.1_coarsened1024_filtered5e5.jld2", "r")
    lon = d["lon"]
    lat = d["lat"]
    z = d["z"]
    close(d)

    # data
    f = [2Ω*sin(deg2rad(lat[j])) for i in eachindex(lon), j in eachindex(lat)] # Coriolis parameter
    H = -z
    H[H .<= 0] .= NaN
    f_over_H = abs.(f ./ H)

    # colorbar
    cmap = "GnBu"
    vmin = -9
    vmax = -7
    cb_ticks = vmin:vmax

    # figure
    fig = plt.figure(1)
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
    img = ax.pcolormesh(lon, lat, log10.(f_over_H)', cmap=cmap, rasterized=true,
                        vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    cb = plt.colorbar(img, label=L"$f/H$ (m$^{-1}$ s$^{-1}$)", ax=ax, orientation="horizontal", extend="both", shrink=0.6)
    cb.set_ticks(cb_ticks)
    cb.set_ticklabels([latexstring(@sprintf("\$10^{%d}\$", tick)) for tick in cb_ticks])
    ax.contour(lon, lat, log10.(f_over_H)', levels=range(-9, -5, 20), 
               colors="k", linestyles="-", linewidths=0.2, transform=ccrs.PlateCarree())
    # ax.coastlines(lw=0.25)
    savefig("figures/f_over_H_map.png")
    @info "Saved 'figures/f_over_H_map.png'"
    savefig("figures/f_over_H_map.pdf")
    @info "Saved 'figures/f_over_H_map.pdf'"
    plt.close()
end

"""
    f_over_H()

Generate Figure 3 of the manuscript.
"""
function f_over_H()
    # params/funcs
    f₀ = 1
    H(x, y) = 1 - x^2 - y^2
    f_over_H(x, y; β = 0) = (f₀ + β*y) / (H(x, y) + eps())
    vmax = 6

    # circular grid
    p, t = get_p_t("data/circle.msh")
    x = p[:, 1]
    y = p[:, 2]
    t = t .- 1 # convert to 0-based indexing for python plotting

    # setup
    fig, ax = plt.subplots(1, 4, figsize=(33pc, 11pc), gridspec_kw=Dict("width_ratios"=>[1, 1, 1, 0.05]))
    ax[1].set_title(L"\beta = 0")
    ax[2].set_title(L"\beta = 0.5")
    ax[3].set_title(L"\beta = 1")
    ax[1].annotate("(a)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[3].annotate("(c)", xy=(-0.04, 1.05), xycoords="axes fraction")
    for a ∈ ax
        a.set_xlabel(L"Zonal coordinate $x$")
        a.axis("equal")
        a.set_xticks(-1:1:1)
        a.set_yticks(-1:1:1)
        a.set_xlim(-1.05, 1.05)
        a.set_ylim(-1.05, 1.05)
        a.spines["left"].set_visible(false)
        a.spines["bottom"].set_visible(false)
    end
    ax[1].set_ylabel(L"Meridional coordinate $y$")
    ax[2].set_yticks([])
    ax[3].set_yticks([])
    ax[2].set_yticklabels([])
    ax[3].set_yticklabels([])
    ax[4].set_visible(false)

    # plot
    levels = (1:4)/4 * vmax
    cmap = "GnBu"
    z = f_over_H.(x, y, β=0.0)
    ax[1].tripcolor(x, y, t, z, cmap=cmap, shading="gouraud", rasterized=true, vmin=0, vmax=vmax)
    ax[1].tricontour(x, y, t, z, levels=levels, colors="k", linewidths=0.5)
    ax[1].plot([-1.0, 1.0], [0.0, 0.0], "r-", alpha=0.7)
    ax[1].plot([0.5], [0.0], "ro", alpha=0.7, ms=2)
    z = f_over_H.(x, y, β=0.5)
    ax[2].tripcolor(x, y, t, z, cmap=cmap, shading="gouraud", rasterized=true, vmin=0, vmax=vmax)
    ax[2].tricontour(x, y, t, z, levels=levels, colors="k", linewidths=0.5)
    z = f_over_H.(x, y, β=1.0)
    img = ax[3].tripcolor(x, y, t, z, cmap=cmap, shading="gouraud", rasterized=true, vmin=0, vmax=vmax)
    ax[3].tricontour(x, y, t, z, levels=levels, colors="k", linewidths=0.5)
    cb = fig.colorbar(img, ax=ax[4], label=L"Planetary vorticity $f/H$", extend="max", fraction=1.0)

    # save
    savefig("figures/f_over_H.png")
    @info "Saved 'figures/f_over_H.png'"
    savefig("figures/f_over_H.pdf")
    @info "Saved 'figures/f_over_H.pdf'"
    plt.close()
end

"""
    zonal_sections()

Generate Figure 4 of the manuscript.
"""
function zonal_sections()
    # setup
    fig, ax = plt.subplots(3, 4, figsize=(39pc, 24pc), gridspec_kw=Dict("width_ratios"=>[1, 1, 1, 0.05]))
    ax[1, 1].set_title(L"\beta = 0")
    ax[1, 2].set_title(L"\beta = 0.5")
    ax[1, 3].set_title(L"\beta = 1")
    ax[1, 1].annotate("(a)", xy=(-0.00, 0.95), xycoords="axes fraction")
    ax[1, 2].annotate("(b)", xy=(-0.00, 0.95), xycoords="axes fraction")
    ax[1, 3].annotate("(c)", xy=(-0.00, 0.95), xycoords="axes fraction")
    ax[2, 1].annotate("(d)", xy=(-0.00, 0.95), xycoords="axes fraction")
    ax[2, 2].annotate("(e)", xy=(-0.00, 0.95), xycoords="axes fraction")
    ax[2, 3].annotate("(f)", xy=(-0.00, 0.95), xycoords="axes fraction")
    ax[3, 1].annotate("(g)", xy=(-0.00, 0.95), xycoords="axes fraction")
    ax[3, 2].annotate("(h)", xy=(-0.00, 0.95), xycoords="axes fraction")
    ax[3, 3].annotate("(i)", xy=(-0.00, 0.95), xycoords="axes fraction")
    for a ∈ ax
        a.axis("equal")
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlim(-1.05, 1.05)
        a.set_ylim(-1.05, 0.05)
        a.spines["left"].set_visible(false)
        a.spines["bottom"].set_visible(false)
    end
    for i ∈ 1:3
        ax[3, i].set_xlabel(L"Zonal coordinate $x$")
        ax[i, 1].set_ylabel(L"Vertical coordinate $z$")
        ax[3, i].set_xticks([-1, 0, 1])
        ax[i, 1].set_yticks([-1, 0])
    end
    ax[1, 4].set_visible(false)
    ax[2, 4].set_visible(false)
    ax[3, 4].set_visible(false)

    # plot
    βs = [0.0, 0.5, 1.0]
    umax = 0.043
    vmax = 0.206
    wmax = 0.060
    for i ∈ eachindex(βs)
        # load gridded sigma data
        d = jldopen(@sprintf("data/gridded_sigma_beta%1.1f_eps2e-02_n0257_i040.jld2", βs[i]))
        x = d["x"]
        y = d["y"]
        σ = d["σ"]
        H = d["H"]
        u = d["u"]
        v = d["v"]
        w = d["w"]
        b = d["b"]
        close(d)
        xx = repeat(x, 1, length(σ))
        j = argmin(abs.(y)) # index where y = 0
        z = H[:, j]*σ'
        u = u[:, j, :]
        v = v[:, j, :]
        w = w[:, j, :]
        fill_nans!(u)
        fill_nans!(v)
        fill_nans!(w)
        u[:, 1] .= 0
        v[:, 1] .= 0
        w[:, 1] .= 0
        w[:, end] .= 0
        b = z .+ b[:, j, :]
        fill_nans!(b)
        b[:, end] .= 0
        @info "vmax values" maximum(abs.(u)) maximum(abs.(v)) maximum(abs.(w))
        ax[1, i].pcolormesh(xx, z, 1e2*u, shading="gouraud", cmap="RdBu_r", vmin=-1e2*umax, vmax=1e2*umax, rasterized=true)
        ax[1, i].contour(xx, z, b, colors="k", linewidths=0.5, alpha=0.3, linestyles="-", levels=-0.9:0.1:-0.1)
        ax[2, i].pcolormesh(xx, z, 1e2*v, shading="gouraud", cmap="RdBu_r", vmin=-1e2*vmax, vmax=1e2*vmax, rasterized=true)
        ax[2, i].contour(xx, z, b, colors="k", linewidths=0.5, alpha=0.3, linestyles="-", levels=-0.9:0.1:-0.1)
        ax[3, i].pcolormesh(xx, z, 1e2*w, shading="gouraud", cmap="RdBu_r", vmin=-1e2*wmax, vmax=1e2*wmax, rasterized=true)
        ax[3, i].contour(xx, z, b, colors="k", linewidths=0.5, alpha=0.3, linestyles="-", levels=-0.9:0.1:-0.1)
    end
    ax[1, 1].plot([0.5, 0.5], [-0.75, 0.0], "r-", alpha=0.7)
    sm = cm.ScalarMappable(norm=colors.Normalize(vmin=-1e2*umax, vmax=1e2*umax), cmap="RdBu_r")
    cb = fig.colorbar(sm, ax=ax[1, 4], label=L"Zonal flow $u$"*"\n"*L"($\times 10^{-2}$)", fraction=1.0)
    sm = cm.ScalarMappable(norm=colors.Normalize(vmin=-1e2*vmax, vmax=1e2*vmax), cmap="RdBu_r")
    cb = fig.colorbar(sm, ax=ax[2, 4], label=L"Meridional flow $v$"*"\n"*L"($\times 10^{-2}$)", fraction=1.0)
    sm = cm.ScalarMappable(norm=colors.Normalize(vmin=-1e2*wmax, vmax=1e2*wmax), cmap="RdBu_r")
    cb = fig.colorbar(sm, ax=ax[3, 4], label=L"Vertical flow $w$"*"\n"*L"($\times 10^{-2}$)", fraction=1.0)

    # save
    savefig("figures/zonal_sections.png")
    @info "Saved 'figures/zonal_sections.png'"
    savefig("figures/zonal_sections.pdf")
    @info "Saved 'figures/zonal_sections.pdf'"
    plt.close()
end

"""
    flow_profiles()

Generate Figure 5 of the manuscript.
"""
function flow_profiles()
    width = 27pc
    fig, ax = plt.subplots(1, 3, figsize=(width, width/3*1.62))
    ax[1].annotate("(a)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[3].annotate("(c)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[1].set_ylabel(L"Vertical coordinate $z$")
    ax[1].set_xlabel(L"Zonal flow $u$"*"\n"*L"($\times 10^{-2}$)")
    ax[2].set_xlabel(L"Meridional flow $v$"*"\n"*L"($\times 10^{-2}$)")
    ax[3].set_xlabel(L"Vertical flow $w$"*"\n"*L"($\times 10^{-2}$)")
    ax[1].set_xlim(-4.5, 4.5)
    ax[2].set_xlim(-13, 13)
    ax[3].set_xlim(-4.5, 4.5)
    ax[1].set_yticks([-0.75, -0.5, -0.25, 0])
    ax[2].set_yticks([])
    ax[3].set_yticks([])
    for a ∈ ax 
        a.spines["left"].set_visible(false)
        a.axvline(0, color="k", lw=0.5)
    end
    βs = [0.0, 0.5, 1.0]
    for i ∈ eachindex(βs)
        # load gridded sigma data
        d = jldopen(@sprintf("data/gridded_sigma_beta%1.1f_eps2e-02_n0257_i040.jld2", βs[i]))
        x = d["x"]
        y = d["y"]
        σ = d["σ"]
        H = d["H"]
        u = d["u"]
        v = d["v"]
        w = d["w"]
        close(d)
        j = argmin(abs.(x .- 0.5)) # index where x = 0.5
        k = argmin(abs.(y)) # index where y = 0
        u = u[j, k, :]; u[1] = 0
        v = v[j, k, :]; v[1] = 0
        w = w[j, k, :]; w[1] = 0; w[end] = 0
        z = H[j, k]*σ
        umask = isnan.(u) .== 0
        vmask = isnan.(v) .== 0
        wmask = isnan.(w) .== 0
        ax[1].plot(1e2*u[umask], z[umask], label=latexstring(@sprintf("\$\\beta = %0.1f\$", βs[i])))
        ax[2].plot(1e2*v[vmask], z[vmask], label=latexstring(@sprintf("\$\\beta = %0.1f\$", βs[i])))
        ax[3].plot(1e2*w[wmask], z[wmask], label=latexstring(@sprintf("\$\\beta = %0.1f\$", βs[i])))
        @info "Actual transport:" U=trapz(u[umask], z[umask]) V=trapz(v[vmask], z[vmask])
    end


    # load bx
    d = jldopen("data/buoyancy_1.0e-02.jld2", "r")
    x = d["x"]
    z = d["z"]
    i_profile = d["i_profile"]
    bx = d["bx"]
    close(d)
    nz = size(z, 2)
    z = z[i_profile, :] 
    bx = bx[i_profile, :]

    # get BL solutions
    u, v, w = solve_baroclinic_problem_BL_U0(ε=2e-2, z=z, ν=ones(nz), f=1, bx=bx, Hx=-1)
    ax[1].plot(1e2*u, z, "k--", lw=0.5, label=L"$U = 0$ theory")
    ax[2].plot(1e2*v, z, "k--", lw=0.5, label=L"$U = 0$ theory")
    ax[3].plot(1e2*w, z, "k--", lw=0.5, label=L"$U = 0$ theory")
    u, v, w = solve_baroclinic_problem_BL(ε=2e-2, z=z, ν=ones(nz), f=1, β=1, bx=bx, by=zeros(nz), U=0, V=0, τx=0, τy=0, Hx=-1, Hy=0)
    # u, v, w = solve_baroclinic_problem_BL(ε=2e-2, z=z, ν=ones(nz), f=1, β=1, bx=bx, by=zeros(nz), U=-0.00196, V=0.00784, τx=0, τy=0, Hx=-1, Hy=0)
    ax[1].plot(1e2*u, z, "k-.", lw=0.5, label=L"$U = V = 0$ theory")
    ax[2].plot(1e2*v, z, "k-.", lw=0.5, label=L"$U = V = 0$ theory")
    ax[3].plot(1e2*w, z, "k-.", lw=0.5, label=L"$U = V = 0$ theory")

    ax[2].legend(loc=(-0.65, 0.5))
    savefig("figures/flow_profiles.png")
    @info "Saved 'figures/flow_profiles.png'"
    savefig("figures/flow_profiles.pdf")
    @info "Saved 'figures/flow_profiles.pdf'"
    plt.close()
end

"""
    psi()

Generate Figure 6 of the manuscript.
"""
function psi()
    # params/funcs
    f₀ = 1
    H(x, y) = 1 - x^2 - y^2
    f_over_H(x, y; β = 0) = (f₀ + β*y) / (H(x, y) + eps())
    f_over_H_levels = (1:4)/4 * 6
    get_levels(vmax) = [-vmax, -3vmax/4, -vmax/2, -vmax/4, vmax/4, vmax/2, 3vmax/4, vmax]

    # setup
    fig, ax = plt.subplots(1, 4, figsize=(33pc, 11pc), gridspec_kw=Dict("width_ratios"=>[1, 1, 1, 0.05]))
    ax[1].set_title(L"\beta = 0")
    ax[2].set_title(L"\beta = 0.5")
    ax[3].set_title(L"\beta = 1")
    ax[1].annotate("(a)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[3].annotate("(c)", xy=(-0.04, 1.05), xycoords="axes fraction")
    for a ∈ ax
        a.set_xlabel(L"Zonal coordinate $x$")
        a.axis("equal")
        a.set_xticks(-1:1:1)
        a.set_yticks(-1:1:1)
        a.set_xlim(-1.05, 1.05)
        a.set_ylim(-1.05, 1.05)
        a.spines["left"].set_visible(false)
        a.spines["bottom"].set_visible(false)
    end
    ax[1].set_ylabel(L"Meridional coordinate $y$")
    ax[2].set_yticklabels([])
    ax[3].set_yticklabels([])
    ax[2].set_yticks([])
    ax[3].set_yticks([])
    ax[4].set_visible(false)

    # plot
    βs = [0.0, 0.5, 1.0]
    img = nothing
    vmax = 0.037
    for i ∈ eachindex(βs)
        d = jldopen(@sprintf("data/psi_beta%1.1f_eps2e-02_n0257_i040.jld2", βs[i]))
        x = d["x"]
        y = d["y"]
        Ψ = d["Ψ"]
        close(d)
        img = ax[i].pcolormesh(x, y, 1e2*Ψ', shading="nearest", cmap="RdBu_r", vmin=-1e2*vmax, vmax=1e2*vmax, rasterized=true)
        ax[i].contour(x, y, 1e2*Ψ', colors="k", linewidths=0.5, linestyles="-", levels=get_levels(1e2*vmax))
        foH = [f_over_H(x[j], y[k], β=βs[i]) for j ∈ eachindex(x), k ∈ eachindex(y)]
        foH[isnan.(Ψ)] .= NaN
        ax[i].contour(x, y, foH', colors=(0.2, 0.5, 0.2), linewidths=0.5, alpha=0.5, linestyles="-", levels=f_over_H_levels)
    end
    fig.colorbar(img, ax=ax[4], label="Barotropic streamfunction\n"*L"$\Psi$ ($\times 10^{-2}$)", fraction=1.0)

    # save
    savefig("figures/psi.png")
    @info "Saved 'figures/psi.png'"
    savefig("figures/psi.pdf")
    @info "Saved 'figures/psi.pdf'"
    plt.close()
end

"""
    baroclinic()

Generate Figure 7 of the manuscript.
"""
function baroclinic()
    # params
    ε = 2e-2
    f = 1
    β = 0
    H = 0.75
    Hx = -1
    Hy = 0
    nz = 2^10
    z = -H*(cos.(π*(0:nz-1)/(nz-1)) .+ 1)/2
    ν = ones(nz)

    # U transport
    τx = 0
    τy = 0
    U = 1
    V = 0
    bx = zeros(nz)
    by = zeros(nz)
    uU, vU = solve_baroclinic_problem(; ε, z, ν, f, bx, by, U, V, τx, τy)
    uUBL, vUBL, wUBL = solve_baroclinic_problem_BL(; ε, z, ν, f, β, bx, by, U, V, τx, τy, Hx, Hy, α=0)

    # V transport
    τx = 0
    τy = 0
    U = 0
    V = 1
    bx = zeros(nz)
    by = zeros(nz)
    uV, vV = solve_baroclinic_problem(; ε, z, ν, f, bx, by, U, V, τx, τy)
    uVBL, vVBL, wVBL = solve_baroclinic_problem_BL(; ε, z, ν, f, β, bx, by, U, V, τx, τy, Hx, Hy, α=0)

    # buoyancy
    τx = 0
    τy = 0
    U = 0
    V = 0
    d = jldopen("data/buoyancy_1.0e-02.jld2", "r")
    i_profile = d["i_profile"]
    z_sim = d["z"][i_profile, :]
    bx = d["bx"][i_profile, :]
    close(d)
    by = zeros(length(z_sim))
    ν_sim = ones(length(z_sim))
    ub, vb = solve_baroclinic_problem(; ε, z=z_sim, ν=ν_sim, f, bx, by, U, V, τx, τy)
    ubBL, vbBL, wbBL = solve_baroclinic_problem_BL(; ε, z=z_sim, ν=ν_sim, f, β, bx, by, U, V, τx, τy, Hx, Hy, α=0)

    # bottom stress stats
    qb = sqrt(f/2/ν_sim[1])
    @printf("∂z(uU)(-H) = % .2e (% .2e)\n", differentiate_pointwise(uU[1:3], z[1:3], z[1], 1),  qb/ε/H)
    @printf("∂z(vU)(-H) = % .2e (% .2e)\n", differentiate_pointwise(vU[1:3], z[1:3], z[1], 1),  qb/ε/H)
    @printf("∂z(uV)(-H) = % .2e (% .2e)\n", differentiate_pointwise(uV[1:3], z[1:3], z[1], 1), -qb/ε/H)
    @printf("∂z(vV)(-H) = % .2e (% .2e)\n", differentiate_pointwise(vV[1:3], z[1:3], z[1], 1),  qb/ε/H)
    @printf("∂z(ub)(-H) = % .2e (% .2e)\n", differentiate_pointwise(ub[1:3], z_sim[1:3], z_sim[1], 1),  -qb/ε*trapz(z_sim.*bx, z_sim)/(H*f))
    @printf("∂z(vb)(-H) = % .2e (% .2e)\n", differentiate_pointwise(vb[1:3], z_sim[1:3], z_sim[1], 1),   qb/ε*trapz(z_sim.*bx, z_sim)/(H*f))

    width = 27pc
    fig, ax = plt.subplots(1, 3, figsize=(width, width/3*1.62))
    for a ∈ ax
        a.spines["left"].set_visible(false)
        a.axvline(0, lw=0.5, c="k")
        a.set_xlabel(L"Velocity $u$, $v$")
    end
    ax[1].text(-0.04, 1.05, s="(a)", transform=ax[1].transAxes, ha="center")
    ax[2].text(-0.04, 1.05, s="(b)", transform=ax[2].transAxes, ha="center")
    ax[3].text(-0.04, 1.05, s="(c)", transform=ax[3].transAxes, ha="center")
    ax[1].set_xlim(-0.5, 1.5)
    ax[1].set_xticks([0, 1/H])
    ax[1].set_xticklabels(["0", L"$1/H$"])
    ax[2].set_xlim(-0.5, 1.5)
    ax[2].set_xticks([0, 1/H])
    ax[2].set_xticklabels(["0", L"$1/H$"])
    ax[3].set_xlim(-0.08, 0.08)
    ax[1].set_ylabel(L"Vertical coordinate $z$")
    ax[1].set_yticks([-0.75, -0.5, -0.25, 0])
    ax[2].set_yticks([])
    ax[3].set_yticks([])
    ax[1].plot(uU, z, label=L"u")
    ax[1].plot(vU, z, label=L"v")
    ax[1].plot(uUBL, z, "k--", lw=0.5, label="BL theory")
    ax[1].plot(vUBL, z, "k--", lw=0.5)
    ax[1].set_title(L"$U = 1, \; V = 0$")
    ax[2].plot(uV, z, label=L"u")
    ax[2].plot(vV, z, label=L"v")
    ax[2].plot(uVBL, z, "k--", lw=0.5, label="BL theory")
    ax[2].plot(vVBL, z, "k--", lw=0.5)
    ax[2].set_title(L"$U = 0, \; V = 1$")
    ax[3].plot(ub, z_sim, label=L"u")
    ax[3].plot(vb, z_sim, label=L"v")
    ax[3].plot(ubBL, z_sim, "k--", lw=0.5, label="BL theory")
    ax[3].plot(vbBL, z_sim, "k--", lw=0.5)
    ax[3].set_title(L"$\partial_x b \neq 0$")
    ax[3].legend(loc=(0.55, 0.7))
    savefig("figures/baroclinic.png")
    @info "Saved 'figures/baroclinic.png'"
    savefig("figures/baroclinic.pdf")
    @info "Saved 'figures/baroclinic.pdf'"
    plt.close()
end

"""
    psi_bl()

Generate Figure 8 of the manuscript.
"""
function psi_bl()
    # load Ψ from 3D model
    d = jldopen("data/psi_beta0.0_eps2e-02_n0257_i040.jld2")
    x3D = d["x"]
    Ψ3D = d["Ψ"]
    close(d)

    # slice at y = 0 from x = 0 to 1
    i0 = size(Ψ3D, 1)÷2 + 1
    j0 = size(Ψ3D, 2)÷2 + 1
    x3D = x3D[i0:end]
    Ψ3D = Ψ3D[i0:end, j0]
    d = jldopen("data/buoyancy_1.0e-02.jld2", "r")
    x = d["x"]
    z = d["z"]
    bx = d["bx"]
    close(d)

    # compute Ψ from BL theory
    ε = 2e-2
    α = 1/2
    V_BL_1 = compute_V_BL(bx, x, z, ε, α; order=1)
    Ψ_BL_1 = cumtrapz(V_BL_1, x) .- trapz(V_BL_1, x)
    V_BL_2 = compute_V_BL(bx, x, z, ε, α; order=2)
    Ψ_BL_2 = cumtrapz(V_BL_2, x) .- trapz(V_BL_2, x)
    # V_BL_3 = compute_V_BL(bx, x, z, ε, α; order=3)
    # Ψ_BL_3 = cumtrapz(V_BL_3, x) .- trapz(V_BL_3, x)

    # plot
    fig, ax = plt.subplots(1)
    ax.set_xlim(0, 1)
    ax.set_xticks(0:0.5:1)
    ax.set_ylim(-6, 0)
    ax.set_yticks(-6:2:0)
    ax.spines["bottom"].set_position("zero")
    ax.xaxis.set_label_coords(0.5, 1.25)
    ax.tick_params(axis="x", top=true, labeltop=true, bottom=false, labelbottom=false)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.plot(x3D, 1e2*Ψ3D,    "C0",          label="3D model")
    ax.plot(x,   1e2*Ψ_BL_1, "k-",  lw=0.5, label=L"BL theory to $O(1)$")
    ax.plot(x,   1e2*Ψ_BL_2, "k--", lw=0.5, label=L"BL theory to $O(\varepsilon)$")
    # ax.plot(x,   1e2*Ψ_BL_3, "k-.", lw=0.5, label=L"BL theory to $O(\varepsilon^2)$")
    ax.legend()
    ax.set_xlabel(L"Zonal coordinate $x$")
    ax.set_ylabel(L"Barotropic streamfunction $\Psi$ ($\times 10^{-2}$)")
    savefig("figures/psi_bl.png")
    @info "Saved 'figures/psi_bl.png'"
    savefig("figures/psi_bl.pdf")
    @info "Saved 'figures/psi_bl.pdf'"
    plt.close()
end
function compute_V_BL(bx, x, z, ε, α; order)
    if !(order in [1, 2, 3])
        throw(ArgumentError("Invalid `order`: $order; must be 1, 2, or 3."))
    end
    
    # parameters
    H = -z[:, 1]
    Hx = differentiate(H, x)
    Γ = @. 1 + α^2*Hx^2
    f = 1
    ν = 1
    q = @. Γ^(-3/4)*√(f/2/ν)

    # order 1
    V = [-1/f*trapz(bx[i, :].*z[i, :], z[i, :]) for i in axes(bx, 1)]
    order -= 1

    if order != 0
        # order 2
        V -= @. ε*H/f/q*bx[:, 1]
        order -= 1
    end

    if order != 0
        # order 3
        bxz_bot = zeros(size(bx, 1))
        for i in 1:size(bx, 1)-1
            bxz_bot[i] = differentiate_pointwise(bx[i, 1:3], z[i, 1:3], z[i, 1], 1) 
        end
        V += @. -ε^2*Γ*H/f^2*bxz_bot + ε^2*bx[:, 1]/(2f*q^2)
        order -= 1
    end

    # if all went well, order should be 0
    @assert order == 0

    return V
end

"""
    alpha()

Generate Figure A1 of the manuscript.
"""
function alpha()
    width = 19pc
    fig, ax = plt.subplots(1, 2, figsize=(width, width/2*1.62))
    ax[1].annotate("(a)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[1].set_ylabel(L"Vertical coordinate $z$")
    ax[1].set_xlabel(L"Cross-slope flow $u$"*"\n"*L"($\times 10^{-2}$)")
    ax[2].set_xlabel(L"Along-slope flow $v$"*"\n"*L"($\times 10^{-2}$)")
    ax[1].spines["left"].set_visible(false)
    ax[2].spines["left"].set_visible(false)
    ax[1].axvline(0, color="k", lw=0.5)
    ax[2].axvline(0, color="k", lw=0.5)
    ax[1].set_yticks(-0.75:0.25:0)
    ax[2].set_yticks([])
    ax[1].set_xlim(-0.6, 1.1)
    ax[2].set_xlim(-3, 13)
    # α = 0
    file = jldopen("data/1D_0.00.jld2")
    u0 = file["u"]
    v0 = file["v"]
    z0 = file["z"]
    ax[1].plot(1e2*u0, z0, label=L"\alpha = 0")
    ax[2].plot(1e2*v0, z0, label=L"\alpha = 0")
    close(file)
    # α = 1/2
    file = jldopen("data/1D_0.50.jld2")
    u = file["u"]
    v = file["v"]
    z = file["z"]
    ax[1].plot(1e2*u, z, label=L"\alpha = 1/2")
    ax[2].plot(1e2*v, z, label=L"\alpha = 1/2")
    close(file)
    ax[1].legend(loc=(0.45, 0.65))
    savefig("figures/alpha.png")
    println("figures/alpha.png")
    savefig("figures/alpha.pdf")
    println("figures/alpha.pdf")
    plt.close()

    # error
    @printf("u error: %e\n", maximum(abs.(u - u0)))
    @printf("v error: %e\n", maximum(abs.(v - v0)))
end

# buoyancy()
f_over_H()
# zonal_sections()
# flow_profiles()
# psi()
# baroclinic()
# psi_bl()
# alpha()