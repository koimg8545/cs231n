from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]

    # print(num_classes)
    # print(num_train)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    df_dw = np.zeros((3073,num_classes))
    for i in range(num_train):
      scores = X[i].dot(W)
      loss -= np.log(np.exp(scores[y[i]])/np.sum(np.exp(scores)))
      
      dy_dw = np.repeat(X[i].reshape(-1,1),repeats = num_classes, axis = 1)



      df_dy_1d = []
      for k in range(num_classes):
        # print(num_classes)
        # print(k)
        if k == y[i]:
          df_dy_1d.append(-1+(np.exp(scores[k]))/(np.sum(np.exp(scores))))
        else:
          df_dy_1d.append((np.exp(scores[k]))/np.sum(np.exp(scores)))

      df_dw += dy_dw * np.array(df_dy_1d).reshape(1,-1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    loss /= num_train
    df_dw /= num_train
    loss += reg * np.sum(W * W)
    df_dw_reg = 2*reg*W

    dW = df_dw + df_dw_reg

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = np.matmul(X,W)
    one_hot = np.eye(num_train)[y][:,:num_classes].astype(int)
    right_score_1d = scores[one_hot > 0]
    sigma = np.sum(np.exp(scores),axis = 1)
    incorrects = np.exp(scores) / sigma.reshape(-1,1)
    incorrects -= one_hot
    df_dy = incorrects.reshape(-1,num_train*num_classes)

    dy_dw = np.repeat(X,repeats = num_classes, axis = 0).T

    df_dw = np.sum((dy_dw * df_dy).reshape(-1,num_train,num_classes), axis = 1)
    loss -= np.sum(np.log(np.exp(right_score_1d)/sigma))/num_train
    
    loss += reg * np.sum(W * W)
    df_dw_reg = 2*reg*W

    dW = df_dw/num_train + df_dw_reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
