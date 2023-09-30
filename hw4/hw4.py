import numpy as np

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def compute_h(self, X):
        h = 1 + np.exp(-np.dot(X, self.theta))
        h = 1 / h

        return h

    def compute_cost(self, X, y, h):
        m = X.shape[0]
        example_cost = -1 * (y * np.log(h) + ((1 - y) * np.log(1 - h)))
        J = np.sum(example_cost) / m
        
        return J

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        self.theta = np.random.rand(X.shape[1]+1)
        self.thetas.append(self.theta)
        
        # apply bias trick
        X = np.c_[np.ones((X.shape[0], 1)), X]

        for i in range(self.n_iter):
            h = self.compute_h(X)
            J = self.compute_cost(X, y, h)
            
            # stop if convergence is reached
            if len(self.Js) > 0 and abs(self.Js[-1] - J) < self.eps:
                self.Js.append(J)
                break
                
            self.Js.append(J)
            self.theta -= self.eta * np.dot(X.T, h-y)
            self.thetas.append(self.theta)

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []

        # apply bias trick
        X = np.c_[np.ones((X.shape[0], 1)), X]
        
        h = self.compute_h(X)
        preds = np.where(h >= 0.5, 1, 0)    # Classify to one of two classes

        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """
    cv_accuracy = None
    
    # set random seed
    np.random.seed(random_state)

    accuracies = []
    indices = np.random.permutation(X.shape[0])
    fold = np.array_split(indices, folds)

    for i in range(folds):
        
        # split into train and test sets
        train_idx = np.concatenate(fold[:i] + fold[i+1:])
        test_idx = fold[i]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # use lor model on the train and test sets
        algo.fit(X_train, y_train)
        y_pred = algo.predict(X_test)

        accuracy = np.sum(y_pred == y_test) / y_test.size
        accuracies.append(accuracy)

    cv_accuracy = np.mean(accuracies)
    
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal density function for a given data,
    mean and standard deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None

    p = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (((data - mu) / sigma) ** 2))

    return p

class EM(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM process
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    def init_params(self, data):
        """
        Initialize distribution params
        """
        self.responsibilities = np.zeros((data.shape[0], self.k))
        self.weights = np.ones(self.k) / self.k
        splits = np.array_split(data, self.k)

        self.mus = np.array([np.mean(split) for split in splits])
        self.sigmas = np.array([np.std(split) for split in splits])
        
        self.costs = []

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        normal_responsibilities = [None for _ in range(self.k)]
        weight = 0

        for i in range(self.k):
            delta_weights = self.weights[i] * norm_pdf(data, self.mus[i], self.sigmas[i])
            weight += delta_weights

            normal_responsibilities[i] = delta_weights

        normal_responsibilities /= weight
        self.responsibilities = np.array(normal_responsibilities)

    def maximization(self, data):
        """
        M step - This function calculates and updates the model parameters
        """
        m = data.shape[0]
        new_weights = np.zeros(self.k)
        new_mus = np.zeros(self.k)
        new_sigmas = np.zeros(self.k)

        for i in range(self.k):
            new_weights[i] = np.sum(self.responsibilities[i]) / m
            new_mus[i] = np.sum(data*self.responsibilities[i]) / (new_weights[i]*m)
            new_sigmas[i] = np.sqrt(np.sum(self.responsibilities[i] * (data-new_mus[i])**2) / (new_weights[i]*m))
        
        # update fields
        self.weights = new_weights
        self.mus = new_mus
        self.sigmas = new_sigmas

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization functions to estimate the distribution parameters.
        Store the parameters in the attributes of the EM object.
        Stop the function when the difference between the previous cost and the current cost is less than the specified epsilon
        or when the maximum number of iterations is reached.

        Parameters:
        - data: The input data for training the model.
        """
        self.init_params(data)

        for i in range(self.n_iter):
            
            self.expectation(data)    # E step
            self.maximization(data)   # M step
            
            J = 0
            sum_pdf = np.zeros_like(data)

            for k in range(self.k):
                sum_pdf += self.weights[k] * norm_pdf(data, self.mus[k], self.sigmas[k])

            for k in range(self.k):
                J += (-np.log(np.sum(sum_pdf)))

            self.costs.append(J)

            # stop if convergence is reached
            if i > 0 and np.abs(self.costs[i-1] - J) < self.eps:
                break

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None

    pdf = np.sum(norm_pdf(data, mus, sigmas)*weights)

    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.classes = None
        self.priors = {}
        self.gmm_params = {}

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        self.classes = np.unique(y)
        
        for class_number in self.classes:
            class_data = X[y == class_number]
            self.priors[class_number] = class_data.shape[0] / X.shape[0]    # get prior probability of each class
            
            feature_models = []
            
            # build EM model for each feature
            for feature in range(X.shape[1]):
                model = EM(k=self.k, random_state=self.random_state)
                model.fit(class_data[:, feature])
                feature_models.append(model)

            self.gmm_params[class_number] = feature_models     

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """

        preds = []

        for instance in X:
            posterior_probs = []

            for class_number in self.classes:
                likelihood = 1
                feature_models = self.gmm_params[class_number]
                
                for j in range(len(feature_models)):
                    # assume independence
                    likelihood *= gmm_pdf(instance[j], feature_models[j].weights, feature_models[j].mus, feature_models[j].sigmas)

                posterior_probs.append(likelihood*self.priors[class_number])

            preds.append(np.argmax(posterior_probs))

        return np.asarray(preds)

# Decision boundaries plot - taken from the notebook
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):
    
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap

    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = np.array(Z)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    plt.show()

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    '''
    
    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None
    
    from matplotlib import pyplot as plt
    
    # Logistic Regression
    lor_model = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor_model.fit(x_train, y_train)

    y_pred_train_lor = lor_model.predict(x_train)
    y_pred_test_lor = lor_model.predict(x_test)
    
    # calculate train and test accuracies
    lor_train_acc = np.mean(y_train == y_pred_train_lor)
    lor_test_acc = np.mean(y_test == y_pred_test_lor)

    # plot
    plt.figure()
    plot_decision_regions(x_train, y_train, classifier=lor_model, title="Logistic Regression Decision Boundaries")

    # Naive Bayes Gaussian
    naive_model = NaiveBayesGaussian(k=k)
    naive_model.fit(x_train, y_train)

    y_pred_train_naive = naive_model.predict(x_train)
    y_pred_test_naive = naive_model.predict(x_test)

    # calculate train and test accuracies
    bayes_train_acc = np.mean(y_train == y_pred_train_naive)
    bayes_test_acc = np.mean(y_test == y_pred_test_naive)

    # plot
    plt.figure()
    plot_decision_regions(x_train, y_train, classifier=naive_model, title="Naive Bayes Decision Boundaries")

    # Plot cost vs iteration number for Logistic Regression
    plt.figure(figsize=(10, 10))
    plt.plot(range(len(lor_model.Js)), lor_model.Js)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title("Cost VS Iteration Number for Logistic Regression Model")
    plt.grid(True)
    plt.show()

    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None

    np.random.seed(1)

    a1 = np.concatenate((np.random.normal(8, 0.5, 200), np.random.normal(-8, 0.5, 200)))
    b1 = np.concatenate((np.random.normal(-8, 0.5, 200), np.random.normal(8, 0.5, 200)))

    a2 = np.concatenate((np.random.normal(-2 , 0.5 , 200), np.random.normal(-6 , 0.5 , 200)))
    b2 = np.concatenate((np.random.normal(-6 , 0.5 , 200), np.random.normal(-2 , 0.5 , 200)))

    a3 = np.concatenate((np.random.normal(-6 , 0.5 , 200) , np.random.normal(3 , 0.5 , 200)))
    b3 = np.concatenate((np.random.normal(-6 , 0.5 , 200) , np.random.normal(3 , 0.5 , 200)))
    
    dataset_a_features = (a1, a2, a3, b1, b2, b3)
    dataset_a_labels = (0, 0, 0, 1, 1, 1)
    
    c1 = np.concatenate((np.random.normal(0, 5, 200), np.random.normal(1, 5, 200)))
    d1 = np.concatenate((np.random.normal(-3, 5, 200), np.random.normal(-4, 5, 200)))

    c2 = c1 * 2
    d2 = d1 * 2

    c3 = np.concatenate((np.random.normal(0, 1, 200), np.random.normal(1, 1, 200)))
    d3 = np.concatenate((np.random.normal(-6, 1, 200), np.random.normal(-7, 1, 200)))
    
    dataset_b_features = (c1, c2, c3, d1, d2, d3)
    dataset_b_labels = (0, 0, 0, 1, 1, 1)
    
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }