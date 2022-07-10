import numpy as np


def pp_float_list(ps):#pretty print functionality
    return ["%2.3f" % p for p in ps]


def _gini(p):
    """
    p: class frequencies as numpy array with np.sum(p)=1
    returns: impurity according to gini criterion
    """
    return 1. - np.sum(p ** 2)


def _entropy(p):
    """
    p: class frequencies as numpy array with np.sum(p)=1
    returns: impurity according to entropy criterion
    """
    idx = np.where(p == 0.)  # consider 0*log(0) as 0
    p[idx] = 1.
    r = p * np.log2(p)
    return -np.sum(r)


def _misclass(p):
    """
    p: class frequencies as numpy array with np.sum(p)=1
    returns: impurity according to misclassification rate
    """
    return 1 - np.max(p)


def impurity_reduction(X, a_i, y, impurity, verbose=0):
    """
    X: data matrix n rows, d columns
    a_i: column index of the attribute to evaluate the impurity reduction for
    y: concept vector with n rows and 1 column
    impurity: impurity function of the form impurity(p_1....p_k) with k=|X[a].unique|
    returns: impurity reduction
    Note: for more readable code we do not check any assertion
    """

    N, d = float(X.shape[0]), float(X.shape[1])

    y_v = np.unique(y)

    # Compute relative frequency of each class in X
    p = (1. / N) * np.array([np.sum(y == c) for c in y_v])
    # ..and corresponding impurity l(D)
    H_p = impurity(p)

    if verbose: print("\t Impurity %0.3f: %s" % (H_p, pp_float_list(p)))

    a_v = np.unique(X[:, a_i])

    # Create and evaluate splitting of X induced by attribute a_i
    # We assume nominal features and perform m-ary splitting
    H_pa = []
    for a_vv in a_v:
        mask_a = X[:, a_i] == a_vv
        N_a = float(mask_a.sum())

        # Compute relative frequency of each class in X[mask_a]
        pa = (1. / N_a) * np.array([np.sum(y[mask_a] == c) for c in y_v])
        H_pa.append((N_a / N) * impurity(pa))
        if verbose: print("\t\t Impurity %0.3f for attribute %d with value %s: " % (H_pa[-1], a_i, a_vv),
                          pp_float_list(pa))

    IR = H_p - np.sum(H_pa)
    if verbose:  print("\t Estimated reduction %0.3f" % IR)
    return IR


def get_split_attribute(X, y, attributes, impurity, verbose=0):
    """
    X: data matrix n rows, d columns
    y: vector with n rows, 1 column containing the target concept
    attributes: A dictionary mapping an attribute's index to the attribute's domain
    impurity: impurity function of the form impurity(p_1....p_k) with k=|y.unique|
    returns: (1) idx of attribute with maximum impurity reduction and (2) impurity reduction
    """

    N, d = X.shape

    IR = [0.] * d
    for a_i in attributes.keys():
        IR[a_i] = impurity_reduction(X, a_i, y, impurity, verbose)
    if verbose: print("Impurity reduction for class attribute (ordered by attributes)", (pp_float_list(IR)))
    b_a_i = np.argmax(IR)
    return b_a_i, IR[b_a_i]


def most_common_class(y):
    """
    :param y: the vector of class labels, i.e. the target
    returns: (1) the most frequent class label in 'y' and (2) a boolean flag indicating whether y is pure
    """
    y_v, y_c = np.unique(y, return_counts=True)
    label = y_v[np.argmax(y_c)]
    fIsPure = len(y_v) == 1
    return label, fIsPure


class DecisionNode(object):
    NODEID = 0

    def __init__(self, attr=-1, children=None, label=None):
        self.attr = attr
        self.children = children
        self.label = label
        self.id = DecisionNode.NODEID
        DecisionNode.NODEID += 1


class DecisionTreeID3(object):
    def __init__(self, criterion=_entropy):
        """
        :param criterion: The function to assess the quality of a split
        """
        self.criterion = criterion
        self.root = None

    def fit(self, X, y, verbose=0):

        def _fit(X, y, attributes=None):

            # Set up temporary variables
            N, d = X.shape
            if attributes == None:
                attributes = {a_i: np.unique(X[:, a_i]) for a_i in range(d)}
            depth = d - len(attributes) + 1

            # if len(X) == 0: return DecisionNode()

            label, fIsPure = most_common_class(y)
            # Stop criterion 1: Node is pure -> create leaf node
            if fIsPure:
                if verbose: print("\t\t Leaf Node with label %s due to purity." % label)
                return DecisionNode(label=label)

            # Stop criterion 2: Exhausted attributes -> create leaf node
            if len(attributes) == 0:
                if verbose: print("\t\t Leaf Node with label %s due to exhausted attributes." % label)
                return DecisionNode(label=label)

            # Get attribute with maximum impurity reduction
            a_i, a_ig = get_split_attribute(X, y, attributes, self.criterion, verbose=verbose)
            if verbose: print(
                "Level %d: Choosing attribute %d out of %s with gain %f" % (depth, a_i, attributes.keys(), a_ig))

            values = attributes.pop(a_i)
            splits = [X[:, a_i] == v for v in values]
            branches = {}

            for v, split in zip(values, splits):
                if not np.any(split):
                    if verbose: print("Level %d: Empty split for value %s of attribute %d" % (depth, v, a_i))
                    branches[v] = DecisionNode(label=label)
                else:
                    if verbose: print("Level %d: Recursion for value %s of attribute %d" % (depth, v, a_i))
                    branches[v] = _fit(X[split, :], y[split], attributes=attributes)

            attributes[a_i] = values
            return DecisionNode(attr=a_i, children=branches, label=label)

        self.root = _fit(X, y)
        return self

    def predict(self, X):
        def _predict(x, node):
            if not node.children:
                return node.label
            else:
                v = x[node.attr]
                child_node = node.children[v]
                return _predict(x, child_node)

        return [_predict(x, self.root) for x in X]

    def print_tree(self, ai2an_map, ai2aiv2aivn_map):
        """
        ai2an_map: list of attribute names
        ai2aiv2aivn_map: list of lists of attribute values,
                         i.e. a value, encoded as integer 2, of attribute with index 3 has name ai2aiv2aivn_map[3][2]
        """

        def _print(node, test="", level=0):
            """
            node: node of the (sub)tree
            test: string specifying the test that yielded the node 'node'
            level: current level of the tree
            """

            prefix = ""
            prefix += " " * level
            prefix += " |--(%s):" % test
            if not node.children:
                print("%s assign label %s" % (prefix, ai2aiv2aivn_map[6][node.label]))
            else:
                print("%s test attribute %s" % (prefix, ai2an_map[node.attr]))
                for v, child_node in node.children.items():
                    an = ai2an_map[node.attr]
                    vn = ai2aiv2aivn_map[node.attr][v]
                    _print(child_node, "%s=%s" % (an, vn), level + 1)

        return _print(self.root)

    def depth(self):
        def _depth(node):
            if not node.children:
                return 0
            else:
                return 1 + max([_depth(child_node) for child_node in node.children.values()])

        return _depth(self.root)