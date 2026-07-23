import numpy as np

# ========================================
def get_rand_proj_mat(data_dim, red_dim, rand_seed=None):
    '''
    Function to get random matrix for Gaussian random projection
    :param data_dim: original data dimensionality
    :param red_dim: reduced dimensionality
    :param rand_seed: random seed
    :return: random matrix
    '''
    # ===================
    if rand_seed != None: np.random.seed(rand_seed)
    rand_mat = np.random.normal(0, 1.0/red_dim, (data_dim, red_dim))

    return rand_mat