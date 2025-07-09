using NCDatasets
using JLD2
using ProgressMeter
using Printf
using PyPlot
using PyCall

ccrs = pyimport("cartopy.crs")

pygui(false)
plt.style.use("plots.mplstyle")
plt.close("all")

const R_E = 6.371e6      # Earth's radius in meters
const Ω = 2π/(24*60*60)  # Earth's angular velocity in rad/s

"""
    z_filtered = filter_topo(lon, lat, z; L=1e5)

Filter the topography `z` on a (`lat`, `lon`) grid by averaging over circles of diameter `L`.

Land points (``z > 0``) are ignored.

See also [`circle_average`](@ref).
"""
function filter_topo(lon, lat, z; L=1e5)
    z_filtered = copy(z)
    @showprogress for i in axes(z, 1), j in axes(z, 2)
        if z[i, j] <= 0 # only filter ocean points
            z_filtered[i, j] = circle_average(lon, lat, z, i, j; D=L)
        end
    end
    return z_filtered
end

"""
    z_avg = circle_average(lon, lat, z, i0, j0; D)

Compute the average of `z` within a circle of diameter `D` centered at `(i0, j0)`.

The average is computed only over ocean points (``z ≤ 0``).

See also [`index_square_ring`](@ref) and [`sphere_distance`](@ref).
"""
function circle_average(lon, lat, z, i0, j0; D)
    lon0 = lon[i0]    # center longitude
    lat0 = lat[j0]    # center latitude
    z_tot = z[i0, j0] # initialize with the value at the center
    count = 1         # number of points within the circle
    n = 1             # starting index to add to (i0, j0)
    ocean_count = 1   # number of ocean points in the ring
    while ocean_count > 0
        ocean_count = 0
        for I in index_square_ring(i0, j0, n, length(lon), length(lat))
            if z[I] <= 0
                if sphere_distance(lon0, lat0, lon[I[1]], lat[I[2]]) <= D/2
                    z_tot += z[I]
                    count += 1
                    ocean_count += 1
                end
            end
        end
        n += 1
    end
    return z_tot/count
end

"""
    indices = index_square_ring(i0, j0, n, nlon, nlat)

Return a vector of indices in a square ring of size `n` centered at `(i0, j0)`.

The indices wrap around `nlon` and `nlat`.
"""
function index_square_ring(i0, j0, n, nlon, nlat)
    # list of all indices in a square ring of size n centered at (i0, j0)
    indices = Vector{CartesianIndex}(undef, 8n)
    idx = 1
    for i in (i0-n):(i0+n)
        indices[idx]   = CartesianIndex(mod1(i, nlon), mod1(j0-n, nlat))  # left edge
        indices[idx+1] = CartesianIndex(mod1(i, nlon), mod1(j0+n, nlat))  # right edge
        idx += 2
    end
    for j in (j0-n+1):(j0+n-1) # avoid double counting corners
        indices[idx]   = CartesianIndex(mod1(i0-n, nlon), mod1(j, nlat))  # left edge
        indices[idx+1] = CartesianIndex(mod1(i0+n, nlon), mod1(j, nlat))  # right edge
        idx += 2
    end 
    @assert idx == 8n + 1
    return indices
end

"""
    d = sphere_distance(lon1, lat1, lon2, lat2)

Compute the distance between two points on the sphere given their longitudes and latitudes.
"""
function sphere_distance(lon1, lat1, lon2, lat2)
    dlon = abs(lon2 - lon1)
    if dlon > 180
        dlon = 360 - dlon  # handle wrap-around at 180 degrees
    end
    dlat = abs(lat2 - lat1)
    dx = R_E*deg2rad(dlon)*cos(deg2rad(lat1))
    dy = R_E*deg2rad(dlat)
    return √(dx^2 + dy^2)
end

"""
    plot_H(lon, lat, z)

Plot the topography `z` as depth `H = -z` on a Robinson projection.
"""
function plot_H(lon, lat, z)
    H = -z
    H[H .<= 0] .= NaN

    fig, ax = plt.subplots(1)
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
    img = ax.pcolormesh(lon, lat, H'/1e3, cmap="magma_r", rasterized=true, 
                        vmin=0, vmax=6, transform=ccrs.PlateCarree())
    plt.colorbar(img, label=L"Depth $H$ (km)", shrink=0.6, extend="max")
    ax.coastlines(lw=0.25)
    ax.spines["geo"].set_linewidth(0.25)
    savefig("H.png")
    @info "Saved 'H.png'"
    plt.close()
end

# load dataset
ds = NCDataset("data/topo_25.1_coarsened1024.nc")

# load arrays
lon = Float64.(ds["lon"][:])
lat = Float64.(ds["lat"][:])
z = Float64.(ds["z"][:, :])
@info "Dimensons" length(lon) length(lat) size(z)

# filter
z = filter_topo(lon, lat, z; L=5e5)
ofile = "data/topo_25.1_coarsened1024_filtered5e5.jld2"
jldsave(ofile; lon, lat, z)
@info "Saved '$ofile'"

# plot
plot_H(lon, lat, z)