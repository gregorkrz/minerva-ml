'''from numba import njit


@njit(cache=True)
def find_narrowest_interval(centers, cumulative_weights, percentage, epsilon, wmin, wmax):
    """
    Find the narrowest interval containing the specified percentage of the distribution.
    Returns: (width, low, high) or (100.0, wmin, wmax) if no valid interval found.
    
    Matches original behavior: wy = points[j][1] - points[i][1]
    where cumulative_weights[i] is the cumulative weight up to centers[i].
    """
    n = len(centers)
    best_width = 100.0
    best_low = wmin
    best_high = wmax
    
    for i in range(n):
        for j in range(i, n):
            # Match original: wy = points[j][1] - points[i][1]
            wy = cumulative_weights[j] - cumulative_weights[i]
            if abs(wy - percentage) < epsilon:
                wx = centers[j] - centers[i]
                if wx < best_width:
                    best_width = wx
                    best_low = centers[i]
                    best_high = centers[j]
    
    return best_width, best_low, best_high

'''