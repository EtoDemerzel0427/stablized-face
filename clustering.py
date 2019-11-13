import numpy as np
import open3d as o3d

prefix = '../new_mannual/'

# 1. find vertices indices of each region in all 11510 nodes.
def find_rows(a, b):
    dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))

    a_view = np.ascontiguousarray(a).view(dt).ravel()
    b_view = np.ascontiguousarray(b).view(dt).ravel()

    sort_b = np.argsort(b_view)
    where_in_b = np.searchsorted(b_view, a_view,
                                 sorter=sort_b)
    where_in_b = np.take(sort_b, where_in_b)
    which_in_a = np.take(b_view, where_in_b) == a_view
    where_in_b = where_in_b[which_in_a]
    which_in_a = np.nonzero(which_in_a)[0]
    return np.column_stack((which_in_a, where_in_b))

def get_index(filename, is_region=True):
    if is_region:
        region = o3d.io.read_triangle_mesh(prefix + filename)
    else:
        region = o3d.io.read_triangle_mesh(filename)

    total = o3d.io.read_triangle_mesh('../new_orig.obj')

    a = np.asarray(region.vertices)
    b = np.asarray(total.vertices)

    res = find_rows(a, b)
    return res

# get indices in 11510 for each region.
head = get_index('head.obj')[:, 1]
left_eb = get_index('left_eb.obj')[:, 1]
right_eb = get_index('right_eb.obj')[:, 1]
nose = get_index('nose.obj')[:, 1]
left_cheek = get_index('left_cheek.obj')[:, 1]
right_cheek = get_index('right_cheek.obj')[:, 1]
left_side = get_index('left_side.obj')[:, 1]
right_side = get_index('right_side.obj')[:, 1]
left_bu = get_index('left_bu.obj')[:, 1]
right_bu = get_index('right_bu.obj')[:, 1]
down = get_index('down.obj')[:, 1]
neck = get_index('neck.obj')[:, 1]

# these regions does not share any neighbors currently. total number is 6633 now.
from functools import reduce
all_index = reduce(np.union1d, (head, left_eb, right_eb, left_side, right_side, left_cheek, right_cheek, left_bu,
                                right_bu, nose, down, neck))


# store in dict
def to_dict(arrs):
    clusters = dict()
    for i, arr in enumerate(arrs):
        for point in arr:
            if point not in clusters:
                clusters[point] = []
            clusters[point].append(i)
    return clusters


clusters = to_dict((head, left_eb, right_eb, left_side, right_side, left_cheek, right_cheek, left_bu,
                    right_bu, nose, down, neck))
print('clusters key number: ', len(clusters.keys()))

# 2. add these lost vertices
face_index = np.load('final_face_index.npy')
selection = o3d.io.read_triangle_mesh('../new_selection.obj')
tris = np.asarray(selection.triangles)

# change the indices to 11510 scale.
for point in tris:
    point[0] = face_index[point[0]]
    point[1] = face_index[point[1]]
    point[2] = face_index[point[2]]

all_index = list(all_index)
for i in range(5):
    early_stop = True
    for face in tris:
        in_region = np.in1d(face, all_index)
        isolated = face[~in_region]
        others = face[in_region]
        if others.size == 3:
            continue
        elif others.size == 2:
            region = clusters[others[0]] or clusters[others[1]]  # pick the first
            # print('others size 2, region:', region)
            clusters[isolated[0]] = region
            all_index.append(isolated[0])
            early_stop = False
        elif others.size == 1:
            clusters[isolated[0]] = clusters[others[0]]
            clusters[isolated[1]] = clusters[others[0]]
            all_index.append(isolated[0])
            all_index.append(isolated[1])
            early_stop = False
        elif others.size == 0:
            # print('other size 0, face:', face)
            early_stop = False

    if early_stop:
        break

print('len of all index:', len(all_index))
print('len of final index:', len(face_index))

# Now all nodes have their own regions, but each node only belong to one region.
for item in clusters.items():
    if len(item[1]) > 1:
        print(item)

# 3. choose shared nodes.
for face in tris:
    a = clusters[face[0]]
    b = clusters[face[1]]
    c = clusters[face[2]]

    if len(a)== 1 and len(b) == 1 and len(c) == 1:
        new_region = list(set(a) | set(b) | set(c))
        clusters[face[0]] = new_region
        clusters[face[1]] = new_region
        clusters[face[2]] = new_region
    elif len(a)== 1 and len(b) == 1:
        new_region = list(set(a) | set(b))
        clusters[face[0]] = new_region
        clusters[face[1]] = new_region
    elif len(a)== 1 and len(c) == 1:
        new_region = list(set(a) | set(c))
        clusters[face[0]] = new_region
        clusters[face[2]] = new_region
    elif len(b)== 1 and len(c) == 1:
        new_region = list(set(b) | set(c))
        clusters[face[1]] = new_region
        clusters[face[2]] = new_region

num = 0
for item in clusters.items():
    if len(item[1]) > 1:
        num += 1
        print(item)

print(num)

# 5. create a one-hot encoded embedding matrix
embedding = np.zeros((len(all_index), 12))
res = get_index('../new_selection.obj')
mapped = dict(zip(res[:,1], res[:, 0]))
print(mapped)

for key, value in clusters.items():
    for col in value:
        embedding[mapped[key], col] = 1/len(value)

np.save('embedding', embedding)
print(embedding[:10, :])
print(embedding[655:680, :])
print(embedding.shape)