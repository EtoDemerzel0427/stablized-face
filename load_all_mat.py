import scipy.io
import os
import numpy as np


def load_all_mat(mat_path='./MAT', from_npy=False) -> object:
    """
    Load all needed MATLAB mat via scipy.io.loadmat.

    :param mat_path: The directory path of all mats.
    :param from_npy: whether to load data from npz file or matlab MAT file.
    :return: 8 different matrices, now in numpy array format.
    """
    if not from_npy:
        blendshapes = scipy.io.loadmat(os.path.join(mat_path, 'blendshapes.mat'))['Blendshapes'][0][0][
            0]  # 34530 x 150 x 47

        # pt87.keys() includes 'ParaZK_87', 'index87in186', 'index_new87',
        # only the latter two are used.
        pt87 = scipy.io.loadmat(os.path.join(mat_path, 'pt87.mat'))
        index_new87 = pt87['index_new87']  # 87 x 1
        mean_face = pt87['meanFace']  # 11510 x 3

        # input50_orig47.keys() includes 'Cr', 'U2', 'single_value', 'w_exp_initial', 'w_id_initial'
        # 'U2' is not used.
        input50_ori47 = scipy.io.loadmat(os.path.join(mat_path, 'input50_ori47.mat'))
        cr = input50_ori47['Cr'][0][0][0]  # 34530 x 50 x 47
        single_value = input50_ori47['single_value']  # 50 x 1
        w_exp_initial = input50_ori47['w_exp_initial']  # 47 x 1, first dim = 1, others zeros.
        w_id_initial = input50_ori47['w_id_initial']  # 50 x 1

        triangles = scipy.io.loadmat(os.path.join(mat_path, 'triangles'))['triangles']  # 11400 x 4
        triangles = triangles.astype(int)  # cast to int, since every entry is an index.
    else:
        data = np.load('matrices.npz')
        blendshapes, index_new87, mean_face, cr, single_value, w_exp_initial, w_id_initial, triangles = \
            data['blendshapes'], data['index_new87'], data['mean_face'], data['cr'], data['single_value'], \
            data['w_exp_initial'], data['w_id_initial'], data['triangles']

    return blendshapes, index_new87, mean_face, cr, single_value, w_exp_initial, w_id_initial, triangles
