def calc_dice(x,y):
    return ((np.sum(x[y == 1]) * 2) / (np.sum(x) + np.sum(y))) * 100
    
def calc_jaccard(x,y):
    eps = 1e-9
    intersection = np.sum(x[y == 1])
    union = np.sum(x) + np.sum(y) - intersection + eps
    return (intersection / union) * 100
