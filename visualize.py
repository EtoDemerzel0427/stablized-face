import numpy as np
from utils.lighting import RenderPipeline
import imageio
import os.path


cfg = {
    'intensity_ambient': 0.3,
    'color_ambient': (1, 1, 1),
    'intensity_directional': 0.6,
    'color_directional': (1, 1, 1),
    'intensity_specular': 0.1,
    'specular_exp': 5,
    'light_pos': (0, 0, 5),
    'view_pos': (0, 0, 5)
}

def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr

def visualize(projected, faces, face_index, pic_name, path='render_res'):
    # avoid isolated vertices and non-face triangles.
    new_faces = []
    new_index = set()
    for i in faces:
        if i[0] in face_index and i[1] in face_index and i[2] in face_index and i[3] in face_index:
            new_index.update([i[0], i[1], i[2], i[3]])
            new_faces.append(i)

    new_index = list(new_index)
    mapping = dict(zip(new_index, list(range(len(new_index)))))

    for face in new_faces:
        for i, j in enumerate(face):
            face[i] = mapping[j]

    new_faces = np.stack(new_faces, axis=0)
    vertices = projected[np.where(np.in1d(face_index, new_index))[0], :]


    triangles = np.zeros((new_faces.shape[0] * 2, new_faces.shape[1] - 1))
    triangles[:new_faces.shape[0], :] = new_faces[:, 0:3]
    triangles[new_faces.shape[0]:, :] = new_faces[:, (0, 2, 3)]

    triangles = _to_ctype(triangles).astype(np.int32)  # 3 x (2850 x 2)
    vertices = _to_ctype(vertices).astype(np.float32)

    img = imageio.imread(pic_name).astype(np.float32)/255.

    app = RenderPipeline(**cfg)
    img_render = app(vertices, triangles, img)

    pic_path = os.path.join(path, os.path.basename(pic_name))
    imageio.imwrite(pic_path, img_render)
    print(f'writing rendered picture to: {pic_path}')

if __name__ == '__main__':
    projected = np.load('projected.npy').T
    faces_load = np.load('face_load.npy')
    face_index = np.load('final_face_index.npy')
    print(projected.shape, faces_load.shape, face_index.shape)
    visualize(projected, faces_load-1, face_index, None)
