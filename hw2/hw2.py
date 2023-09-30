import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0

    labels = data[:,-1]
    _, counts = np.unique(labels, return_counts=True)
    countsSquared = np.dot(counts, counts)
    instances = data.shape[0]
    sum = np.sum(countsSquared) / (instances * instances)
     
    gini = 1 - sum
    
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0

    labels = data[:,-1]
    _, counts = np.unique(labels, return_counts=True)
    instances = data.shape[0]
    countsLog = np.log2(counts / instances)
    sum = np.dot(counts, countsLog) / instances
        
    entropy = 0 - sum

    return entropy

def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    goodness = 0
    groups = {} # groups[feature_value] = data_subset

    currentImpurity = impurity_func(data)
    instances = data.shape[0]
    values, counts = np.unique(data[:,feature], return_counts=True)
    
    sum = 0
    
    for i in range(len(counts)):
        dataTrimmed = data[data[:, feature] == values[i], :]
        groups[values[i]] = dataTrimmed
        impurity = impurity_func(dataTrimmed)
        sum += ((counts[i] / instances) * impurity)
        
    goodness = currentImpurity - sum  # gain
    
    if gain_ratio:
        featureCol = data[:,feature]
        _, counts = np.unique(featureCol, return_counts=True)
        instances = data.shape[0]
        countsLog = np.log2(counts / instances)
        sum = np.dot(counts, countsLog) / instances
        
        splitInformation = 0 - sum
        
        if splitInformation != 0:
            goodness /= splitInformation
        else:
            goodness = 0
    
    return goodness, groups

class DecisionNode:

    def __init__(self, data, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio 
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """

        pred = None
        
        values, counts = np.unique(self.data[:,-1], return_counts=True)
        pred = values[np.argmax(counts)]  # most common label
        
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
     
    def split(self, impurity_func):

        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """

        if self.depth == self.max_depth:
            self.terminal = True
            return
        
        bestGoodness = -1
        numberOfFeatures = self.data.shape[1] - 1
        
        # Find best feature to split according to
        for feature in range(numberOfFeatures):
            goodness, groups = goodness_of_split(self.data, feature, impurity_func, self.gain_ratio)
            
            if goodness > bestGoodness:
                bestGoodness = goodness
                self.feature = feature
                
        values, counts = np.unique(self.data[:,self.feature], return_counts=True) # list of values of best feature
        
        if len(values) <= 1 or bestGoodness <= 0:  # if there is only one value for the selected feature in the node
            self.terminal = True
            
        elif len(values) > 1 and bestGoodness > 0:
            
            values1, y0 = np.unique(self.data[:,-1], return_counts=True)
            
            # Calculate chi squared for the node
            py0 = y0[0] / self.data.shape[0]            
            chiSquared = 0
            
            for i in range(len(values)):
                dataChild = self.data[self.data[:, self.feature] == values[i], :]
                df = dataChild.shape[0]
                values2, pf = (np.unique(dataChild[:,-1], return_counts=True))
                
                if len(values2) > 1:
                    nf = df - pf[0]
                elif values2[0] == values1[0]:
                    nf = 0
                else:
                    nf = pf[0]
                    pf[0] = 0
                    
                e0 = df * py0
                e1 = df * (1 - py0)
                chiSquared += ((((pf[0]-e0)**2)/e0) + (((nf-e1)**2)/e1))
            
            # Prune according to chi table
            if self.chi < 1 and len(values) > 1:
                if chiSquared <= chi_table[len(values) - 1][self.chi]:
                    self.terminal = True
                    return
            
            # Split node according to selected feature
            for i in range(len(values)):
                dataChild = self.data[self.data[:, self.feature] == values[i], :]
                child = DecisionNode(dataChild, depth=self.depth+1, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
                self.add_child(child, values[i])       
            
def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = None
    
    root = DecisionNode(data, chi=chi, max_depth=max_depth, gain_ratio=gain_ratio)
    queue = []
    queue.append(root)
    
    while queue:
        node = queue.pop(0)
        if not node.terminal:
            node.split(impurity)
            for child in node.children:
                queue.append(child)
    
    return root

def predict(root, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    pred = None
    
    if root.terminal:
        pred = root.pred
        
    else:
        featureValue = instance[root.feature]
        
        for i, child in enumerate(root.children):
            if root.children_values[i] == featureValue:
                pred = predict(child, instance)
                  
        if not pred:
            pred = root.pred

    return pred

def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    
    count_match = 0
    
    for instance in dataset:
        if predict(node, instance) == instance[-1]:  # there is a match between the prediction and the label
            count_match += 1
            
    accuracy = count_match / dataset.shape[0]
    
    return accuracy

def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output: the training and testing accuracies per max depth
    """
    training = []
    testing  = []

    impurity = calc_entropy
    gain_ratio = True
    
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = build_tree(data=X_train, impurity=impurity, gain_ratio=gain_ratio, chi=1, max_depth=max_depth)
        training.append(calc_accuracy(tree, X_train))
        testing.append(calc_accuracy(tree, X_test))

    return training, testing

def chi_pruning(X_train, X_test):

    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc  = []
    depth = []

    impurity = calc_entropy
    gain_ratio = True
    
    for chi in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        tree = build_tree(data=X_train, impurity=impurity, gain_ratio=gain_ratio, chi=chi)
        chi_training_acc.append(calc_accuracy(tree, X_train))
        chi_testing_acc.append(calc_accuracy(tree, X_test))

        # Calculate tree depth
        max_depth = 0
        queue = []
        queue.append(tree)
    
        while queue:
            node = queue.pop(0)
            
            if node.depth > max_depth:
                max_depth = node.depth
                
            for child in node.children:
                queue.append(child)
                
        depth.append(max_depth)

    return chi_training_acc, chi_testing_acc, depth

def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of nodes in the tree.
    """
    n_nodes = None

    if not node:
        return n_nodes
    else:
        n_nodes = 0
    
    queue = []
    queue.append(node)
    
    while queue:
        node = queue.pop(0)
        n_nodes += 1
        
        for child in node.children:
            queue.append(child)

    return n_nodes
