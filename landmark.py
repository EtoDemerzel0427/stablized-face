import numpy as np


def landmark_loss(projected, landmarks, points, w, pose=True):
    """
    landmark energy.
    :param projected: 2 x n. The projected 2d point of all vertices.
    :param landmarks: (87,). 87 landmarks indices in 11510.
    :param points: 87 x 2. the coordinates of landmarks.
    :param w: weight for each region.
    :return:
    """
    # scale = 1 #160.53447
    # rt[:, :3] *= scale
    # projected = np.dot(rt, face)[:2, :]
    cor = projected[:, landmarks].T
    diff = cor - points # n x 2
    if pose:
        w = w[landmarks]  # n x 1
        #loss = np.mean(w * np.square(diff))
        #loss = np.sqrt(np.mean(diff * diff)).squeeze()  

        loss = np.sqrt(np.mean(w * np.sum(diff * diff, axis=1, keepdims=True)))
        #loss = np.mean(w * np.linalg.norm(diff, axis=1, keepdims=True))  # there is an abs op here, making derivative inconstant
    else:
        #loss = np.mean(np.linalg.norm(diff, axis=1))
        # loss = np.mean(np.square(diff))
        loss = np.sqrt(np.mean(diff * diff)).squeeze()

    return loss






if __name__ == '__main__':
    pass
