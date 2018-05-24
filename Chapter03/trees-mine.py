import operator
from math import log

import treePlotter


def create_dataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def calc_shannonent(dataset):
    nam_entries = len(dataset)
    label_counts = {}
    for vec in dataset:
        current_label = vec[-1]
        label_counts[current_label] = label_counts.get(current_label, 0) + 1
    shannonent = 0
    for k, v in label_counts.items():
        prob = v / nam_entries
        shannonent -= prob * log(prob, 2)
    return shannonent


def split_dataset(dataset, index, value):
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[index] == value:
            reduced_feat_vec = feat_vec[:index] + feat_vec[index + 1:]
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset


def choose_best_feature_to_split(dataset):
    num_features = len(dataset[0]) - 1
    base_entropy = calc_shannonent(dataset)
    best_infogain, best_feature = 0, -1
    for i in range(num_features):
        feat_list = [vec[i] for vec in dataset]
        feat_set = set(feat_list)
        temp_entropy = 0
        for value in feat_set:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / len(dataset)
            temp_entropy += prob * calc_shannonent(sub_dataset)
        temp_infogain = base_entropy - temp_entropy
        print('infoGain=', temp_infogain, 'bestFeature=', i, base_entropy, temp_entropy)
        if temp_infogain > best_infogain:
            best_infogain = temp_infogain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    classCount = {}
    for vote in class_list:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def create_tree(dataset, labels):
    class_list = [vec[-1] for vec in dataset]
    if class_list.count(class_list[0]) == class_list.__len__():
        return class_list[0]

    if len(dataset[0]) == 1:
        return majority_cnt(class_list)

    best_feat = choose_best_feature_to_split(dataset)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    feat_values = [vec[best_feat] for vec in dataset]
    unique_values = set(feat_values)
    sublabels = labels[:]
    del (sublabels[best_feat])
    for value in unique_values:
        my_tree[best_feat_label][value] = create_tree(split_dataset(dataset, best_feat, value), sublabels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    key = test_vec[feat_index]
    valueOfFeat = second_dict[key]
    print('+++', first_str, 'xxx', second_dict, '---', key, '>>>', valueOfFeat)
    if isinstance(valueOfFeat, dict):
        classlabel = (classify(valueOfFeat, feat_labels, test_vec))
    else:
        classlabel = valueOfFeat
    return classlabel


def store_tree(inputtree, filename):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(inputtree, f)


def grab_tree(filename):
    import pickle
    with open(filename, 'rb') as f:
        return pickle.loads(f)


with open('.\lenses.txt', 'r') as fr:
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = create_tree(lenses, lensesLabels)
treePlotter.createPlot(lensesTree)
