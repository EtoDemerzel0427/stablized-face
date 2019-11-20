import numpy as np

flow_select = np.load('flow_select.npy')
print(flow_select.shape)

embedding = np.load('embedding.npy')
print(embedding.shape)

selected = embedding[flow_select, :]
print(selected.shape)

print(selected)


for i in range(selected.shape[0]):
    for j in range(selected.shape[1]):
        if 0 < selected[i, j] < 1:
            selected[i, j] = 1

# np.save('part_emb', selected)

x = np.sum(selected, axis=1)
print(len(np.where(x > 1)[0]))

print(selected.min())
nums = np.sum(selected, axis=0)
print(nums)
print(nums.sum())

print(nums)
print(selected.dot(nums.reshape(-1,1)))
#np.save('nums', nums)