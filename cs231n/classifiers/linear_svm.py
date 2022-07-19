from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    n = 3073
    m = num_classes
    loss = 0.0
    df_dw = np.zeros((n,m))
    for i in range(500):
        scores = X[i].dot(W)
        # print(X[i].shape)
        # print(W.shape)
        # print(scores.shape)
        correct_class_score = scores[y[i]]
        dy_dw = np.repeat(X[i].reshape(-1,1),repeats = m, axis = 1)
        # print(dy_dw)

        df_dy_1d = []
        # print(y[i])
        for k in range(m):
          if k == y[i]:
            df_dy_1d.append(1-m)
          else:
            df_dy_1d.append(1)

        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                # print(1223)
                loss += margin
            else:
              # print(123123)
              df_dy_1d[j] = 0
              df_dy_1d[y[i]] += 1
        
        # print(df_dy_1d)
        df_dy = np.diag(df_dy_1d)
        # print(df_dy)
        # print(df_dy.shape)
        # print(dy_dw.shape)

        df_dw += np.matmul(dy_dw, df_dy)



    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    df_dw /= num_train
    

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    df_dw_reg = 2*reg*W

    dW = df_dw + df_dw_reg
    # print(dW)
    # dW = df_dw

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_train = X.shape[0]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = np.matmul(X,W)
    one_hot = np.eye(num_train)[y][:,:10].astype(int)
    right_score_1d = scores[one_hot > 0]
    prev_max = (scores - right_score_1d.reshape(-1,1) + 1) * (1-one_hot)
    loss = np.sum(prev_max[prev_max>0])/num_train
    loss += reg * np.sum(W * W)
    dy_dw = np.repeat(X,repeats = 10, axis = 0).T
    num_margin = np.sum((prev_max>0), axis = 1)
    df_dy = np.repeat(-num_margin.reshape(-1,1),10,axis = 1) * one_hot + (prev_max>0)
    df_dy = df_dy.reshape(1,-1)
    df_dw = np.sum((dy_dw * df_dy).reshape(-1,num_train,10), axis = 1)

    dW = df_dw/num_train + 2*reg*W
    # print(dW)
    # print(dW)
    # np.zeros()
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
