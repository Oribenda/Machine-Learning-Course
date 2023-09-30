import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.1,
            (0, 1): 0.2,
            (1, 0): 0.2,
            (1, 1): 0.5
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.3,
            (0, 1): 0.2,
            (1, 0): 0.2,
            (1, 1): 0.3
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.4,
            (0, 1): 0.15,
            (1, 0): 0.1,
            (1, 1): 0.35
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.24,
            (0, 0, 1): 0.06,
            (0, 1, 0): 0.06,
            (0, 1, 1): 0.14,
            (1, 0, 0): 0.16,
            (1, 0, 1): 0.09,
            (1, 1, 0): 0.04,
            (1, 1, 1): 0.21,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y

        for x in X:
            for y in Y:
                if np.isclose(X[x]*Y[y], X_Y[(x, y)]):
                    return False
        
        return True

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C

        for x in X:
            for y in Y:
                for c in C:
                    X_Y_given_C = X_Y_C[(x, y, c)] / C[c]
                    X_given_C = X_C[(x, c)] / C[c]
                    Y_given_C = Y_C[(y, c)] / C[c]
                    
                    if not np.isclose(X_given_C*Y_given_C, X_Y_given_C):
                        return False
                    
        return True

def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None

    pmf = ((rate**k) * np.exp(0-rate)) / np.math.factorial(k)
    log_p = np.log2(pmf)

    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = None

    likelihoods = [np.sum([poisson_log_pmf(sample, rate) for sample in samples]) for rate in rates]

    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates) # might help

    rate = rates[np.argmax(likelihoods)]

    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None

    mean = np.sum(samples) / len(samples)

    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None

    denominator = np.sqrt(2*np.pi*(std**2))
    numerator = np.exp(0 - ((x-mean)**2 / (2*(std**2))))
    p = numerator / denominator

    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """

        self.dataset = dataset
        self.class_value = class_value
        self.dataset_of_class = dataset[dataset[:,-1] == class_value]
        self.mean = np.mean(self.dataset_of_class[:,:-1], axis=0)
        self.std = np.std(self.dataset_of_class[:,:-1], axis=0)
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None

        prior = self.dataset_of_class.shape[0] / self.dataset.shape[0]

        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = None

        likelihood = 1
        
        for i in range(len(self.mean)):
            likelihood *= normal_pdf(x[i], self.mean[i], self.std[i])

        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None

        posterior = self.get_prior() * self.get_instance_likelihood(x)

        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """

        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None

        posterior_ccd0 = self.ccd0.get_instance_posterior(x)
        posterior_ccd1 = self.ccd1.get_instance_posterior(x)
        
        if posterior_ccd0 > posterior_ccd1:
            pred = 0
        else:
            pred = 1

        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None

    count_correct = 0
    
    for x in test_set:
        if map_classifier.predict(x) == x[-1]:
            count_correct += 1
            
    acc = count_correct / test_set.shape[0]

    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None

    x = np.delete(x, -1)
    d = len(x)
    det = np.linalg.det(cov)
    cov_inverse = np.linalg.inv(cov)
    transpose = (x - mean).T
    
    pdf = ((2*np.pi)**(0-(d/2))) * (det**-0.5) * np.exp(-0.5*transpose @ cov_inverse @ (x - mean))

    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        
        self.dataset = dataset
        self.class_value = class_value
        self.dataset_of_class = dataset[dataset[:,-1] == class_value]
        self.mean = np.mean(self.dataset_of_class[:,:-1], axis=0)
        self.cov = np.cov(self.dataset_of_class[:,:-1], rowvar=False)
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None

        prior = self.dataset_of_class.shape[0] / self.dataset.shape[0]

        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = None

        likelihood = multi_normal_pdf(x, self.mean, self.cov)

        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None

        posterior = self.get_prior() * self.get_instance_likelihood(x)

        return posterior

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """

        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None

        prior_ccd0 = self.ccd0.get_prior()
        prior_ccd1 = self.ccd1.get_prior()
        
        if prior_ccd0 > prior_ccd1:
            pred = 0
        else:
            pred = 1

        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """

        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None

        likelihood_ccd0 = self.ccd0.get_instance_likelihood(x)
        likelihood_ccd1 = self.ccd1.get_instance_likelihood(x)
        
        if likelihood_ccd0 > likelihood_ccd1:
            pred = 0
        else:
            pred = 1

        return pred

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """

        self.dataset = dataset
        self.class_value = class_value
        self.dataset_of_class = dataset[dataset[:,-1] == class_value]
    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = None

        prior = self.dataset_of_class.shape[0] / self.dataset.shape[0]

        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = None

        v = []
        n_i = self.dataset_of_class.shape[0]
        
        for j in range(self.dataset.shape[1]-1):
            values, counts = np.unique(self.dataset[:,j], return_counts=True)
            v.append(len(values))
            
        likelihood = 1
            
        for j in range(len(x)-1):
            
            if x[j] not in self.dataset[:,j]:
                likelihood *= EPSILLON
                
            else:
                n_i_j = self.dataset_of_class[self.dataset_of_class[:,j]==x[j]].shape[0]
                likelihood *= ((n_i_j + 1) / (n_i + v[j]))

        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None

        posterior = self.get_prior() * self.get_instance_likelihood(x)

        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """

        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None

        posterior_ccd0 = self.ccd0.get_instance_posterior(x)
        posterior_ccd1 = self.ccd1.get_instance_posterior(x)
        
        if posterior_ccd0 > posterior_ccd1:
            pred = 0
        else:
            pred = 1

        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None

        count_correct = 0
    
        for x in test_set:
            if self.predict(x) == x[-1]:
                count_correct += 1
            
        acc = count_correct / test_set.shape[0]

        return acc