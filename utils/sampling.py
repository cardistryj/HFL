import numpy as np
from torchvision import datasets, transforms
from typing import List

def partition(labels: np.ndarray, num: int or List[int], noniid_level: int, origin_indices: np.ndarray, offset: int = 0):
    if isinstance(num, int) or all([n == num[0] for n in num]):
        num = num if isinstance(num, int) else len(num)
        assert len(labels) % num == 0
        shard_sizes = [len(labels) // num] * num
    else: # num is given as a list
        assert not noniid_level
        sum_num = sum(num)
        shard_sizes = [len(labels) * n // sum_num for n in num]
    indices = np.arange(len(labels))

    shard_split = []
    count = 0
    for i in shard_sizes:
        shard_split.append(count)
        count += i
    shard_split.append(count)

    if not noniid_level:
        np.random.shuffle(indices)
    else:
        idxs_labels = np.split(np.vstack((indices, labels)), noniid_level, axis = 1)
        indice_list = [sub_idx_label[0,sub_idx_label[1,:].argsort()] for sub_idx_label in idxs_labels]
        indices = np.concatenate(indice_list, 0)

        shards = np.split(indices, noniid_level * num)
        shard_idxs = np.arange(noniid_level * num)
        np.random.shuffle(shard_idxs)
        indices = np.concatenate([shards[i] for i in shard_idxs], 0)
    
    return {i + offset: origin_indices[indices[shard_split[i]: shard_split[i+1]]] for i in range(len(shard_split[:-1]))}
