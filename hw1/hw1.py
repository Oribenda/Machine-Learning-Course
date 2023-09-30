###### Your ID ######
# ID1: 207163783
# ID2: 208711739
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """

    X = X - X.mean(axis = 0, keepdims = True)
    deltaX = X.max(axis = 0, keepdims = True) - X.min(axis = 0, keepdims = True)
    X = X / deltaX
    
    y = y - y.mean()
    deltaY = y.max() - y.min()
    y = y / deltaY
    
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """

    m = X.shape[0]
    ones = np.ones(m)
    X = np.c_[ones, X]

    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.
    
    h = np.dot(X, theta)
    difference = h - y
    differenceSquared = np.dot(difference, difference)
    sum = np.sum(differenceSquared)
    J = sum / (2 * X.shape[0])
    
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    
    for k in range(num_iters):
        
        h = np.dot(X, theta)
        difference = h - y
        derivative = np.dot(X.T, difference) / X.shape[0]
        theta -= alpha * derivative
        J_history.append(compute_cost(X, y, theta))
    
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    
    Xtranspose = X.T
    XtransposeX = np.dot(Xtranspose, X)
    pinv_X = np.dot(np.linalg.inv(XtransposeX), Xtranspose)
    pinv_theta = np.dot(pinv_X, y)
    
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration

    DIFF = 10 ** (-8)
    
    for k in range(num_iters):
        
        h = np.dot(X, theta)
        difference = h - y
        derivative = np.dot(X.T, difference) / X.shape[0]
        theta -= alpha * derivative
        J_history.append(compute_cost(X, y, theta))
        
        if k > 0 and (abs(J_history[k] - J_history[k-1]) < DIFF):
            break

    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}

    for alpha in alphas:
        
        np.random.seed(42)
        theta = np.random.random(size = X_train.shape[1])
        theta, j = efficient_gradient_descent(X_train, y_train, theta, alpha, iterations)
        loss = compute_cost(X_val, y_val, theta)
        alpha_dict[alpha] = loss
        
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    X_selected_train = np.ones(X_train.shape[0])
    X_selected_val = np.ones(X_val.shape[0])
    np.random.seed(42)
    
    for i in range(5):
        
        minJ = float('inf')
        theta = np.random.random(size = i + 2)
        
        for feature in range(X_train.shape[1]):
            
            if feature not in selected_features:
                
                X_selected_train = np.c_[X_selected_train, X_train[:,feature]]
                theta, J_history = efficient_gradient_descent(X_selected_train, y_train, theta, best_alpha, iterations)
                X_selected_val = np.c_[X_selected_val, X_val[:,feature]]
                J = compute_cost(X_selected_val, y_val, theta)
                
                if J < minJ:
                    minJ = J
                    minFeature = feature
                    
                X_selected_train = X_selected_train[:,:-1]
                X_selected_val = X_selected_val[:,:-1]
                
        X_selected_train = np.c_[X_selected_train, X_train[:,minFeature]]
        X_selected_val = np.c_[X_selected_val, X_val[:,minFeature]]
        
        selected_features.append(minFeature)

    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    n = df.shape[1]
    names = df.keys()
    
    for i in range(n):
        
        for j in range(i, n):
            
            new_col = pd.DataFrame({f'{names[i]}*{names[j]}': (df.iloc[:, i] * df.iloc[:, j])})
            df_poly = pd.concat((df_poly, new_col), axis = 1)

    return df_poly