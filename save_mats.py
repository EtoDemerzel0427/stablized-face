import numpy as np
from load_all_mat import load_all_mat

matrices = load_all_mat()
matrices = list(matrices)

names = ['blendshapes', 'index_new87', 'mean_face', 'cr', 'single_value', 'w_exp_initial', 'w_id_initial', 'triangles']

d = dict(zip(names, matrices))
np.savez('matrices.npz', **d)

if __name__ == '__main__':
    data = np.load('matrices.npz')
    blendshapes, index_new87, mean_face, cr, single_value, w_exp_initial, w_id_initial, triangles = \
        data['blendshapes'], data['index_new87'], data['mean_face'], data['cr'], data['single_value'], \
        data['w_exp_initial'], data['w_id_initial'], data['triangles']

    print(blendshapes.shape, index_new87.shape, mean_face.shape, cr.shape, single_value.shape, w_exp_initial.shape,
          w_id_initial.shape, triangles.shape)
