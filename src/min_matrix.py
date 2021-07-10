import numpy as np
from itertools import permutations

def score_align(batch, aligned_matrix, p):
    score = 0
    for g_id in range(len(batch)):
        one_hot = batch[g_id]
        if 1 in one_hot:  # ignore if unclassified
            unshift_c = np.argmax(one_hot)
            shift_c = p[unshift_c]
            for layer in aligned_matrix:
                score += layer[g_id, shift_c]
    return score


def make_slice(lin_batch, batch_ids, BAC):
    B, A, C = BAC
    batch = np.zeros((A, C))
    for i in range(len(lin_batch)):
        i_class = lin_batch[i]
        g_id = batch_ids[i]
        if i_class == 0:
            batch[g_id, 0] = 1

        elif i_class == 1:
            batch[g_id, 1] = 1
    
    return batch


def make_line(al_matrix):
    B, A, C = al_matrix.shape
    lin_matrix = []
    for g_id in range(A):
        scores = [sum(al_matrix[: ,g_id, c]) for c in range(C)]
        lin_matrix.append(np.argmax(scores))
    
    return lin_matrix


def align_batches(matrix):
    """Sparse matrix from multiple batches with size:
    no. batches * no. particles * no. classes 
    
    As each batch assigns label to class differently we need a way of
    locating the consensus labelling.  Here is attempted an iterative
    method where each permutation is tried and best consensus is taken
    forward.
    
    2nd batch compared to first. 3rd compared to 1st and 2nd. 4th
    compared to 1st, 2nd and 3rd etc.
    Could potentially mix back in to get more robust consensus.

    """
    B, A, C = matrix.shape
    aligned_matrix = np.array([matrix[0]], dtype=int)
    perm = permutations(range(C))
    perm = [p for p in perm]
    for batch in matrix[1:]:
        scores = [score_align(batch, aligned_matrix, p) for p in perm]

        opt_perm = perm[np.argmax(scores)]
        opt_batch = np.zeros((A, C), dtype=int)
        for a in range(A):
            one_hot = batch[a]
            if 1 in one_hot:
                unshift_c = np.argmax(one_hot)
                shift_c = opt_perm[unshift_c]
                one_hot_shift = np.zeros(C, dtype=int)
                one_hot_shift[shift_c] = int(1)
                opt_batch[a] = one_hot_shift
        aligned_matrix = np.concatenate((aligned_matrix, [opt_batch]))

    return aligned_matrix


if __name__ == "__main__":
    matrix = [
        [[1, 0, 0, 0],  # ---> C
        [0, 1, 0, 0],  #|
        [1, 0, 0, 0],  #|    #|
        [0, 1, 0, 0],  #V A  #|
        ],                   #|
        [[0, 1, 0, 0],        #V B
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        ],
        [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        ],
        [[0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        ],
    ]
    matrix = np.load("/home/lexi/Documents/CLIC/test_data/output/matrix.npy")
    
    gt_ids_bin = np.load("/home/lexi/Documents/CLIC/test_data/output/gt_ids_bin.npy")
    count = 0
    # for i, j in zip(matrix[0], matrix[3]):
    #     if 1 in i and 1 in j:
    #         count += 1
    # print(count)
    # exit()
    matrix = np.array(matrix, dtype=int)
    aligned_matrix = align_batches(matrix)
    all_classes = make_line(aligned_matrix)
    import clustering
    score = clustering.score_bins(gt_ids_bin, all_classes)
    print(score)


