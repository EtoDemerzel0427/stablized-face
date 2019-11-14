import numpy as np


def landmark_loss(projected, face_index, landmark_index, points, w, pose=True):
    """
    landmark energy.
    :param embedding: n x 12. Normalized embedding matrix.
    :param projected: 2 x n. The projected 2d point of all vertices.
    :param indices: (n,). Indices of all front face vertices.
    :param landmarks: (87,). 87 landmarks indices in 11510.
    :param points: 87 x 2. the coordinates of landmarks.
    :param w:
    :return:
    """
    mapped = dict(zip(face_index, np.arange(len(face_index))))
    landmarks = [mapped[i] for i in landmark_index]  # map to the row of projected matrix
    cor = projected[:, landmarks].T
    diff = cor - points # n x 2
    if pose:
        w = w[landmarks]  # n x 1
        loss = np.mean(w * np.linalg.norm(diff, axis=1, keepdims=True))
    else:
        loss = np.mean(np.linalg.norm(diff, axis=1))

    return loss






if __name__ == '__main__':
    pass
