import numpy as np


def train_error(dataset):
    '''
        The training error of the dataset
    '''
    p = calc_p(dataset)
    return min(p, 1 - p)


def entropy(dataset):
    '''
        The entropy formula https://en.wikipedia.org/wiki/ID3_algorithm#Entropy
        C(p) = -p*log(p) - (1-p)log(1-p)
    '''
    p = calc_p(dataset)
    # Can't take log of 0
    if p == 0 or p == 1:
        return 0
    return (-1 * p)*(np.log2(p)) - (1 - p)*(np.log2(1 - p))


def gini_index(dataset):
    '''
        Gini index formula
        C(p) = 2*p*(1-p)
    '''
    p = calc_p(dataset)
    return 2*p*(1-p)


def calc_p(dataset):
    num_y_1 = 0
    for data in dataset:
        if data[0] == 1:
            num_y_1 += 1
    if len(dataset) == 0:
        return 0

    p = num_y_1/len(dataset)

    return p


class Node:
    def __init__(self, left=None, right=None, depth=0, index_split_on=0, isleaf=False, label=1, info=None):
        self.left = left
        self.right = right
        self.depth = depth
        self.index_split_on = index_split_on
        self.isleaf = isleaf
        self.label = label
        self.info = {} if info is None else info


class DecisionTree:

    def __init__(self, data, validation_data=None, gain_function=entropy, max_depth=40):
        self.max_depth = max_depth
        self.root = Node()
        self.root.info['cost'] = 0
        self.gain_function = gain_function
        self.majority_label = self.m_label(data)  # holds the most common label in the whole dataset

        indices = list(range(1, len(data[0])))

        self._split_recurs(self.root, data, indices)

        # Pruning
        if not (validation_data is None):
            self._prune_recurs(self.root, validation_data)

    def m_label(self, data):
        self.majority_label = data[0][0]
        return self.calc_label(data)

    def predict(self, features):
        return self._predict_recurs(self.root, features)

    def accuracy(self, data):
        return 1 - self.loss(data)

    def loss(self, data):
        cnt = 0.0
        test_Y = [row[0] for row in data]
        for i in range(len(data)):
            prediction = self.predict(data[i])
            if (prediction != test_Y[i]):
                cnt += 1.0
        return cnt/len(data)

    def _predict_recurs(self, node, row):
        if node.isleaf or node.index_split_on == 0:
            return node.label
        split_index = node.index_split_on
        if row[split_index]:
            return self._predict_recurs(node.left, row)
        else:
            return self._predict_recurs(node.right, row)

    def _prune_recurs(self, node, validation_data):
        if node.isleaf:
            return

        self._prune_recurs(node.left, validation_data)
        self._prune_recurs(node.right, validation_data)

        current_loss = self.loss(validation_data)
        if node.left.isleaf and node.right.isleaf:
            node.isleaf = True
            node.label = node.info['predicted']
            if self.loss(validation_data) >= current_loss:
                node.isleaf = False
                node.label = -1
            else:
                node.left = None
                node.right = None

    def _is_terminal(self, node, data, indices):
        if (
            len(data) == 0
            or len(indices) == 0
            or self.same_class(data)
            or node.depth > self.max_depth
           ):
            return (True, self.calc_label(data))

        return (False, -1)

    def calc_label(self, data):
        labels = [item[0] for item in data]
        # If we have no label, return the most likely label for the set
        if len(labels) == 0:
            return self.majority_label
        # Otherwise return the most likely label for this branch
        return max(set(labels), key=labels.count)

    def same_class(self, data):
        labels = [item[0] for item in data]
        if len(labels) == 0:
            return True
        first = labels[0]
        for data_item in labels:
            if data_item == first:
                continue
            else:
                return False
        return True

    def _split_recurs(self, node, rows, indices):
        terminal = self._is_terminal(node, rows, indices)
        node.isleaf = terminal[0]
        node.label = terminal[1]
        node.info['data_size'] = len(rows)

        if terminal[0]:
            return

        node.info['predicted'] = self.calc_label(rows)
        split_index = indices[0]
        max_gain = 0
        split_pos = 0

        for i in range(0, len(indices)):
            gain = self._calc_gain(rows, indices[i], self.gain_function)
            # print("gain choice ", gain)
            if gain > max_gain:
                split_index = indices[i]
                max_gain = gain
                split_pos = i
        # print("gain chosen ", max_gain)

        node.index_split_on = split_index
        del indices[split_pos]

        split_on_true = [data for data in rows if data[split_index]]
        split_on_false = [data for data in rows if not data[split_index]]

        node.right = Node()
        node.right.info['cost'] = max_gain
        node.right.depth = node.depth + 1

        node.left = Node()
        node.left.info['cost'] = max_gain
        node.left.depth = node.depth + 1

        # print(len(split_on_true), " ", len(split_on_false))

        self._split_recurs(node.left, split_on_true, indices[:])
        self._split_recurs(node.right, split_on_false, indices[:])

    def _calc_gain(self, data, split_index, gain_function):
        split_on_true = [data_point for data_point in data if data_point[split_index]]
        split_on_false = [data_point for data_point in data if not data_point[split_index]]

        return gain_function(data) - (
                len(split_on_true)/len(data) * gain_function(split_on_true) +
                len(split_on_false)/len(data) * gain_function(split_on_false)
                )

    def loss_plot_vec(self, data):
        self._loss_plot_recurs(self.root, data, 0)
        loss_vec = []
        q = [self.root]
        num_correct = 0
        while len(q) > 0:
            node = q.pop(0)
            num_correct = num_correct + node.info['curr_num_correct']
            loss_vec.append(num_correct)
            if node.left is not None:
                q.append(node.left)
            if node.right is not None:
                q.append(node.right)

        return 1 - np.array(loss_vec)/len(data)

    def _loss_plot_recurs(self, node, rows, prev_num_correct):
        labels = [row[0] for row in rows]
        curr_num_correct = labels.count(node.label) - prev_num_correct
        node.info['curr_num_correct'] = curr_num_correct

        if not node.isleaf:
            left_data, right_data = [], []
            left_num_correct, right_num_correct = 0, 0
            for row in rows:
                if row[node.index_split_on]:
                    left_data.append(row)
                else:
                    right_data.append(row)

            left_labels = [row[0] for row in left_data]
            left_num_correct = left_labels.count(node.label)
            right_labels = [row[0] for row in right_data]
            right_num_correct = right_labels.count(node.label)

            if node.left is not None:
                self._loss_plot_recurs(node.left, left_data, left_num_correct)
            if node.right is not None:
                self._loss_plot_recurs(node.right, right_data, right_num_correct)
