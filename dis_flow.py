import cv2
from scipy import interpolate
import numpy as np


def dis_flow_loss(prev_gray, cur_gray, inst, prev_projected, projected, w, pose=True):
    """

    :param prev_gray: previous frame (gray scale), here 720 x 1280
    :param cur_gray: current frame (gray scale), here 720 x 1280
    :param inst: the dis flow algorithm object
    :param prev_projected: the projected 2d face form previous frame.  2 x n
    :param projected: the current projected 2d face.
    :param w:
    :return:
    """
    flow = inst.calc(prev_gray, cur_gray, None)
    height, width = prev_gray.shape
    x = np.arange(height)
    y = np.arange(width)
    fx = interpolate.interp2d(x, y, flow[:, :, 0].T)
    fy = interpolate.interp2d(x, y, flow[:, :, 1].T)

    vec_fx = np.vectorize(fx)
    vec_fy = np.vectorize(fy)

    pred_x = vec_fx(prev_projected[0,:], prev_projected[1, :])  # 7794
    assert pred_x[0] == fx(prev_projected[0,0], prev_projected[1, 0])

    pred_y = vec_fy(prev_projected[0, :], prev_projected[1, :])
    assert pred_y[1] == fy(prev_projected[0, 1], prev_projected[1, 1])

    motion = np.vstack((pred_x, pred_y))
    assert motion.shape[0] == 2
    diff = projected - prev_projected - motion  # 2 x n

    if pose:
        loss = np.mean(w * np.linalg.norm(diff.T, axis=1, keepdims=True))
    else:
        loss = np.mean(np.linalg.norm(diff, axis=0))

    return loss

if __name__ == '__main__':
    import scipy.optimize
    import numpy as np
    x = np.random.random((3, 3))
    a = np.random.random((3, 4))
    y = np.random.random((3, 4))

    def f(flat_x):
        x = flat_x.reshape(3,3)
        return np.linalg.norm(y - np.dot(x, a))

    res = scipy.optimize.minimize(f, x, method='BFGS', options={'disp':True})
    print(res.x)
    print(res.x.reshape(3,3).dot(a))
    print(y)