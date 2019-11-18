import numpy as np
import scipy.optimize
import glob
import os
import cv2
from load_all_mat import load_all_mat
from landmark import landmark_loss
from dis_flow import dis_flow_loss

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def apply_exp():
    pass

def apply_pose():
    pass

# def build_model(face, rt):
#     """
#     From 3d to 2d.
#     :param core: (3xn) x exp_num [here (3x7794) x 47]
#     :param beta: (exp_num - 1) x ncluster [here 46 x 1]
#     :param embedding: n x nclusters. [here 7794 x 12] embedding matrix.
#     :param rt: extrinsic matrix, 3 x 4
#     :return:
#     model: n x 3, the first 2 dims are (x,y), z indicates depth, for rendering use.
#     """
#     # scale * rotation * face + translation
#     rt[:, :3] *= scale
#     model = np.dot(rt, face)
#
#     return model


def pose_loss(rt, face, landmarks, points, w, inst, prev_projected, prev_rt, i):
    scale = 1
    rt[:, :3] *= scale
    projected = np.dot(rt, face)[:2, :]
    # model = build_model(face, rt)
    # projected = model[:2, :]
    # the landmark loss value
    key_loss = landmark_loss(projected, landmarks, points, w, pose=True)
    # print("Landmark energy：", key_loss)

    # the dense optical flow loss
    # if i == 0:
    #     dis_loss = 0
    #     temp_loss = 0
    # else:
    #     prev = cv2.imread(pic_names[i - 1])
    #     cur = cv2.imread(pic_names[i])
    #     prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    #     cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    #
    #     dis_loss = dis_flow_loss(prev_gray, cur_gray, inst, prev_projected, projected, w, pose=True)
    #     print("Dense flow energy: ", dis_loss)
    #
    #     # todo add temp_loss
    #     temp_loss = 0

    return key_loss # + 0.01 * dis_loss + temp_loss


def pose_opt(flat_rt, face, landmarks, points, w, inst, prev_projected, prev_rt, i):
    def f(flat_rt):
        rt = flat_rt.reshape(-1, 4)
        return pose_loss(rt, face, landmarks, points, w, inst, prev_projected, prev_rt, i)

    print(bcolors.HEADER + "[Start Opt] Processing..." + bcolors.ENDC)
    res = scipy.optimize.minimize(f, flat_rt, options={'disp':True})
    if not res.success:
        print(bcolors.FAIL + f'[Opt END] Opt Success: {res.success}' + bcolors.ENDC)
        print(bcolors.FAIL + res.message + bcolors.ENDC)
    else:
        print(bcolors.OKGREEN + f'[Opt END] Opt Success: {res.success}' + bcolors.ENDC)
    new_rt = res.x.reshape(-1, 4)
    return new_rt



# 1. load segmentation result
face_index = np.load('final_face_index.npy')  # 7794
embedding = np.load('embedding.npy')  # 7794 x 12, normalized embedding matrix
assert face_index.shape == (7794,) and embedding.shape == (7794, 12)
n_cluster = 12
weight = np.ones((n_cluster, 1))  # todo: change to dynamic weight
w = np.dot(embedding, weight)


# 2. region-based multi-linear face model
# 2.1 gather base data
print(bcolors.OKGREEN + "[Loading data] load_all_mat..." + bcolors.ENDC)
blendshapes, index_new87, mean_face, cr, single_value, w_exp_initial, \
w_id_initial, triangles = load_all_mat(from_npy=True)  # core: 34530 x 50 x 47
print(bcolors.OKBLUE + "[Loaded] Done!" + bcolors.ENDC)


landmarks = index_new87.squeeze() - 1
mapped = dict(zip(face_index, np.arange(len(face_index))))
landmarks = [mapped[i] for i in landmarks]  # map to the row of projected matrix

# TODO: learn shape weight from online adaptation method
shape_weight = np.load('/Users/momo/Desktop/face_model/para_id.npy')  # 50 x 1, result from another model

core = np.tensordot(cr, shape_weight, axes=(1,0)).squeeze()  # 34530 x 47

keys = np.vstack((3 * face_index, 3 * face_index + 1, 3 * face_index + 2)).T
keys = keys.flatten()

core[:, 1:] = core[:, 1:] - np.expand_dims(core[:, 0], 1)  # convert to delta
core = core[keys, :]  # (3 x 7794) x 47

# 2.2 model parameters
# 2.2 gather pic data
root_path = '/Users/momo/Desktop/test_frames/test_video_frames'
pic_names = sorted(glob.glob(os.path.join(root_path, '*.jpeg')))
pt_names = sorted(glob.glob(os.path.join(root_path, '*lds87.txt')))
print(bcolors.OKBLUE + f'[Counted] Total pic number is {len(pic_names)}' + bcolors.ENDC)

beta = np.random.random((46, n_cluster))  # 46 x 12 expression parameters
rt = np.zeros((3, 4))  # pose parameters: [R|t] extrinsic matrix
rt[:, :3] = np.eye(3)



# 2.3 build model
inst = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)  # online version: should be fast.
inst.setUseSpatialPropagation(True)

prev_projected, prev_rt = None, None
for i in range(len(pic_names)):
    print(bcolors.OKGREEN + f'[Processing] pic number {i + 1}...' + bcolors.ENDC)
    points = np.loadtxt(pt_names[i])  # 87 x 2

    beta = np.zeros((46, n_cluster))  # 46 x 12 expression parameters
    rt = np.zeros((3, 4))  # pose parameters: [R|t] extrinsic matrix
    rt[:,:3] = np.eye(3)

    # 2.4 perform pose optimization
    # 2.4.1 apply expression parameters
    face = np.expand_dims(core[:, 0], 1) + np.dot(core[:, 1:], beta)  # (3 x n) x 12
    face = np.sum(face * np.repeat(embedding, 3, axis=0), axis=1).reshape(-1, 3).T  # 3 x n
    face = np.vstack((face, np.ones((1, face.shape[1]))))  # 4 x n

    # 2.4.2 optimize pose parameters
    rt = rt.flatten()
    rt = pose_opt(rt, face, landmarks, points, w, inst, prev_projected, prev_rt, i)

    # print('optimized rt: ', rt)
    scale = 1 #160.53447  # TODO： 这个是从之前matlab代码求出来的前20张图片的f的平均值
    rt[:, :3] *= scale
    prev_projected = np.dot(rt, face)[:2, :]
    prev_rt = rt

    if i == 2:
        break



