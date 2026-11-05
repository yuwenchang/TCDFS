import numpy as np

def get_equal_matrix_from_equal_class(target_label):
    """Return a class-by-sample membership matrix from decision labels."""
    item_num = len(target_label)
    label_idx = {}
    for index, label in enumerate(target_label):
        if label in label_idx:
            label_idx[label].append(index)
        else:
            label_idx[label] = [index]
    target_partition = [idxs for idxs in label_idx.values()]
    D = []
    for Di in target_partition:
        D.append([0] * item_num)
        for j in Di:
            D[-1][j] = 1
    return D
def test1():
    equal_class = [1,1,1,2,2]
    print(get_equal_matrix_from_equal_class(equal_class))

def get_set_from_similary_seriers(similary_seriers):
    return {index for index in similary_seriers.index if similary_seriers[index] >0}
def get_set_from_similary_vector(similary_vector):
    return {index for index,item in enumerate(similary_vector) if item >0}
def and_operator(v1,v2):
    return [min(i,j) for i,j in zip(v1,v2)]
def and_operator2(xi,dj):
    return [min(xi[idx],dj[idx]) for idx in xi.index]
def test2():
    similary_vect = [1,0.8,0.2,0]
    s1 = get_set_from_similary_vector(similary_vect)
    print(s1)
    s2 = get_set_from_similary_vector([1,1,0,0])
    print(s2)
    s3 = s1&s2
    print(s3)
    print(len(s3))
    print(and_operator(similary_vect,[1,1,0,0]))



if __name__ == "__main__":
    pass
    # test1()
    # test2()
