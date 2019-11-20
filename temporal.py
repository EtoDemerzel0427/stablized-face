import numpy as np

def temporal_loss(rt, prev_rt, eta):
    diff = (rt - prev_rt) / 160  # todo: the scale problem
    return eta * (np.sqrt(np.sum(diff[:, :3] * diff[:, :3])) + 0.01 * np.sqrt(np.sum((diff[:,3])**2)))

def calc_eta(motion, w):
    """

    :param motion: 300 x 2
    :param w: 300 x 1
    :return:
    eta, the parameter of temporal loss.
    """
    assert motion.shape == (2, 300)
    assert w.shape == (12, 1)

    sigma2 = 0.01
    gamma = np.load('nums.npy').reshape(-1, 1)
    assert gamma.shape == w.shape
    p = np.zeros((12, 1))
    p[:-1, :] = w[:-1, :] / gamma[:-1, :]


    partial_emb = np.load('part_emb.npy')
    assert partial_emb.shape == (300, 12)


    motion_norm = np.linalg.norm(motion, axis=0, keepdims=True).T

    eta = np.exp(-sigma2 * np.sum(motion_norm * np.dot(partial_emb, p)))

    return eta

if __name__ == '__main__':
    rt = np.zeros((3, 4))
    rt[:, :3] = np.eye(3)

    noise = np.random.random((3, 4))
    prev_rt = rt - noise

    print(temporal_loss(rt, prev_rt, 1))
    print(np.linalg.norm(noise[:,:3], ord='fro') + 0.01 * np.linalg.norm(noise[:, 3]))