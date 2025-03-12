import torch

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, D = X.shape
    C = W.shape[1]

    for i in range(N):
        scores = X[i] @ W  # shape (C,)
        # Shift for numeric stability
        scores -= torch.max(scores)
        exp_scores = torch.exp(scores)
        sum_exp = torch.sum(exp_scores)
        correct_exp = exp_scores[y[i]]

        # Compute loss for this sample
        loss += -torch.log(correct_exp / sum_exp)

        # Compute gradient
        for j in range(C):
            p_j = exp_scores[j] / sum_exp
            if j == y[i]:
                dW[:, j] += (p_j - 1) * X[i]
            else:
                dW[:, j] += p_j * X[i]

    # Average loss and gradient
    loss /= N
    dW /= N

    # Add regularization to loss and gradient
    loss += reg * torch.sum(W * W)
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]

    scores = X @ W  # shape (N, C)
    # Shift for numeric stability
    scores -= torch.max(scores, dim=1, keepdim=True)[0]
    exp_scores = torch.exp(scores)
    sum_exp_scores = torch.sum(exp_scores, dim=1, keepdim=True)  # shape (N, 1)

    # Probabilities
    probs = exp_scores / sum_exp_scores  # shape (N, C)

    # Loss
    correct_probs = probs[torch.arange(N), y]
    loss = -torch.sum(torch.log(correct_probs)) / N
    loss += reg * torch.sum(W * W)

    # Gradient
    dscores = probs.clone()
    dscores[torch.arange(N), y] -= 1  # subtract 1 from the correct class
    dW = X.t() @ dscores
    dW /= N
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
