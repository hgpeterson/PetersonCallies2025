"""
    fill_nan(f, i, j)

Compute the value of `f[i, j]` by averaging the values of its neighbors in a 3x3 grid.
"""
function fill_nan(f, i, j)
    sum = 0
    count = 0
    for m ∈ i-1:i+1, n ∈ j-1:j+1
        if m ≥ 1 && n ≥ 1 && m ≤ size(f, 1) && n ≤ size(f, 2)
            if !isnan(f[m, n])
                sum += f[m, n]
                count += 1
            end
        end
    end
    return sum / count
end

"""
    fill_nans!(f)

Fill NaN values in the array `f` by averaging the values of their neighbors in a 3x3 grid.

This function modifies `f` in place.

See also [`fill_nan`](@ref)
"""
function fill_nans!(f)
    ff = copy(f)
    for i ∈ axes(f, 1), j ∈ axes(f, 2)
        if isnan(f[i, j])
            ff[i, j] = fill_nan(f, i, j)
        end
    end
    f .= ff
    return f
end
