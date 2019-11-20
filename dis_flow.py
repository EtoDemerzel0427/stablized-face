import cv2
from scipy import interpolate
import numpy as np


def dis_flow_loss(vec_fx, vec_fy, prev_projected, projected, w, pose=True):
    """

    :param vec_fx: flow function of x-axis from bilinear interpolation
    :param vec_fy: flow function of y-axis from bilinear interpolation
    :param prev_projected: the projected 2d face form previous frame.  2 x n
    :param projected: the current projected 2d face.
    :param w: 300 x 1
    :return:
    """
    # flow = inst.calc(prev_gray, cur_gray, None)
    # height, width = prev_gray.shape
    # x = np.arange(height)
    # y = np.arange(width)
    # fx = interpolate.interp2d(x, y, flow[:, :, 0].T)
    # fy = interpolate.interp2d(x, y, flow[:, :, 1].T)
    #
    # vec_fx = np.vectorize(fx)
    # vec_fy = np.vectorize(fy)
    #print('prev_projected:', prev_projected[0, :].shape)
    # print(vec_fy(1,2))

    #pred_x = vec_fx(prev_projected[0,:], prev_projected[1, :])  # 7794
    # assert pred_x[0] == fx(prev_projected[0,0], prev_projected[1, 0])
    # pred_x = list(map(lambda x,y: vec_fx(x,y), prev_projected[0, :], prev_projected[1, :]))
    # pred_y = list(map(lambda x,y: vec_fy(x,y), prev_projected[0, :], prev_projected[1, :]))
    # print(pred_x)
    # print(pred_x.shape)

    #pred_y = vec_fy(prev_projected[0, :], prev_projected[1, :])
    # assert pred_y[1] == fy(prev_projected[0, 1], prev_projected[1, 1])

    pred_x = np.diag(vec_fx(prev_projected[0, :], prev_projected[1, :]))
    pred_y = np.diag(vec_fy(prev_projected[0, :], prev_projected[1, :]))


    motion = np.vstack((pred_x, pred_y))
    assert motion.shape[0] == 2
    diff = (projected - prev_projected - motion).T  # n x 2

    if pose:
        loss = np.sqrt(np.mean(w * np.sum(diff * diff, axis=1, keepdims=True)))
        # loss = np.mean(w * np.linalg.norm(diff.T, axis=1, keepdims=True))
    else:
        loss = np.sqrt(np.mean(diff * diff))
        #loss = np.mean(np.linalg.norm(diff, axis=0))


    return loss, motion

if __name__ == '__main__':
    import os, glob


    root_path = '/Users/momo/Desktop/test_frames/test_video_frames'
    pic_names = sorted(glob.glob(os.path.join(root_path, '*.jpeg')))

    inst = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)  # online version: should be fast.
    inst.setUseSpatialPropagation(True)


    prev = cv2.imread(pic_names[0])
    cur = cv2.imread(pic_names[1])
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

    flow = inst.calc(prev_gray, cur_gray, None)
    height, width = prev_gray.shape

    test_x = np.random.randn(300) * height
    test_y = np.random.randn(300) * width



    x = np.arange(height)
    y = np.arange(width)
    fx = interpolate.interp2d(x, y, flow[:, :, 0].T)
    fy = interpolate.interp2d(x, y, flow[:, :, 1].T)

    import time
    start = time.time()
    a = np.diag(fx(test_x, test_y))
    b = np.diag(fy(test_x, test_y))
    motion = np.vstack((a,b))
    from temporal import calc_eta
    print(calc_eta(motion, np.ones((12, 1))))
    end = time.time()
    print('interp2d: ', end - start)

    start = time.time()
    out1 = [fx(XX, YY) for XX, YY in zip(test_x, test_y)]
    out2 = [fy(XX, YY) for XX, YY in zip(test_x, test_y)]
    end = time.time()
    print("Spline: ", end-start)

    #new_fx = interpolate.SmoothBivariateSpline(x, y, flow[:, :, 0].T, kx=1, ky=1, s=0)

