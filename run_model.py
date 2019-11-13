import numpy as np
import cv2
from load_all_mat import load_all_mat

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# 1. load segmentation result
# current unsolved problems:
# 1.1 数据。之前缺少连续帧mesh的数据，TODO: 之后可以用祥哥那个生成的视频的结果来重新做一次。
# currently use the former results.
face_index = np.load('/Users/momo/Desktop/region_seg/face_index.npy')
clusters = np.load('/Users/momo/Desktop/region_seg/clusters.npy')
n_cluster = 12

# 2. region-based multi-linear face model

# 2.1 gather data
print(bcolors.OKGREEN + "[Loading data] load_all_mat..." + bcolors.ENDC)
blendshapes, index_new87, mean_face, cr, single_value, w_exp_initial, \
w_id_initial, triangles = load_all_mat(from_npy=True)
print(bcolors.OKBLUE + "[Loaded] Done!" + bcolors.ENDC)

shape_weight = np.load('/Users/momo/Desktop/face_model/para_id.npy')  # result from another model

core = np.tensordot(cr, shape_weight, axes=(1,0)).squeeze()
keys = np.vstack((3 * face_index, 3 * face_index + 1, 3 * face_index + 2)).T
keys = keys.flatten()
print(keys.shape)
core[:, 1:] = core[:, 1:] - np.expand_dims(core[:, 0], 1)  # convert to delta
core = core[keys, :]  # (3 x 5913) x 47

# 2.2 model parameters
beta = np.random.random((46, n_cluster))  # expression parameters
rot = np.random.random((3,3))  # pose parameters: rotation
translate = np.random.random((1,3))  # pose parameters: translation

# 2.3 build model
face = np.expand_dims(core[:, 0], 1) + np.dot(core[:, 1:], beta)
print(face.shape)




