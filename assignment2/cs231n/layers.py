from builtins import range
import numpy as np
import pprint

pp = pprint.pprint


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_reshape = x.reshape(x.shape[0], -1)

    out = x_reshape.dot(w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = x.shape[0]

    # print(x.shape)

    x_reshape = x.reshape(N, -1)  # NxD

    # print(x_reshape.shape, x.shape, w.shape)

    dx_reshape = dout.dot(w.T)  # NxM * MxD -> NxD
    dx = dx_reshape.reshape(*x.shape)  # -> (N, d1, ..., d_k)

    dw = x_reshape.T.dot(dout)  # DxN * NxM -> DxM
    dw /= N

    db = np.ones(N).T.dot(dout)

    db /= N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.copy(x)
    out[x < 0] = 0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = np.copy(dout)
    dx[x < 0] = 0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = x.shape[0]

    softmax = np.exp(x)

    # Normalize to get probabilities
    softmax /= np.array([np.sum(softmax, axis=1)]).T

    # Calculate loss
    loss = -np.log(softmax[range(N), y])
    loss = np.mean(loss)

    # Calculate gradient of the lost with respect tox
    dx = softmax.copy()
    dx[range(N), y] -= 1

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # q = x - mean  # 1
        # qsq = q * q  # 2
        # var = np.sum(qsq, axis=0) / N
        # vareps = var + eps  # 3
        # den = np.sqrt(vareps)  # 4
        # invden = 1.0 / den  # 5
        # x_norm = q * invden  # 6
        # out = x_norm * gamma + beta  # 7
        #
        # running_mean = momentum * running_mean + (1.0 - momentum) * mean
        # running_var = momentum * running_var + (1.0 - momentum) * var
        #
        # cache = (x, gamma, beta, q, qsq, vareps, den, invden, x_norm)

        m = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        # q = x - mean  # 1
        # qsq = q * q  # 2
        # var = np.sum(qsq, axis=0) / N

        # Normalize
        # Trick for division float to increase precision!!
        vareps = var + eps
        inden = 1.0 / np.sqrt(vareps)
        x_hat = (x - m) * inden

        # Allow distribution to shift
        out = gamma * x_hat + beta

        running_mean = running_mean * momentum + (1.0 - momentum) * m
        running_var = running_var * momentum + (1.0 - momentum) * var

        cache = [x, x_hat, m, var, gamma, vareps]
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_hat = (x - running_mean) / np.sqrt(running_var)

        out = gamma * x_hat + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    [x, x_hat, m, var, gamma, vareps] = cache

    sigma = np.sqrt(vareps)
    r = (x - m)
    q = 1.0 / sigma

    N, D = x.shape

    dL_dy = dout

    # Beta
    # dy_dbeta = np.ones(D, )  # Do this for clarity
    dL_dbeta = np.sum(dL_dy, axis=0)

    # Gamma
    dy_dgamma = x_hat
    dL_dgamma = np.sum(dL_dy * dy_dgamma, axis=0)

    # X hat
    dy_dx_hat = gamma
    dL_dx_hat = dL_dy * dy_dx_hat

    # Upstream gradient: dL/dx_hat (N,D)
    # Node variables: x_hat = r*q
    # Local gradient: dx_hat/dq = r (N,D)
    # Local gradient: dx_hat/dr = q (N,D)
    dx_hat_dq = r
    dx_hat_dr = np.ones((N, D)) * q

    dL_dq = np.sum(dL_dx_hat * dx_hat_dq, axis=0)
    dL_dr1 = dL_dx_hat * dx_hat_dr  # r1 since dL_dr is the sum of two gradients

    # Upstream gradient: dL/dq (D,)
    # Node variable: q = 1/sigma (sigma is the standard deviation)
    # Local gradient: dq/dsigma = - 1 / sigma**2 (D,)
    dq_dsigma = - 1.0 / sigma ** 2.0
    dL_dsigma = dL_dq * dq_dsigma

    # Upstream gradient: dL/dsigma (D,)
    # Node variable: sigma = sqrt(var)
    # Local gradient: dsigma/dvar = 1 / (2*sqrt(var)) (D,)
    dsigma_dvar = 1.0 / (2.0 * np.sqrt(vareps))
    dL_dvar = dL_dsigma * dsigma_dvar

    # Upstream gradient: dL/dvar (D,)
    # Node variable: var = z*1/N
    # Local gradient: dvar/dz =1/N*ones(D) (D,)
    dvar_dz = 1.0 / N * np.ones(D)
    dL_dz = dL_dvar * dvar_dz

    # Upstream gradient: dL/dz (D,)
    # Node variable: z = r^2 dim = (N,D)
    #                where r=x-m
    # Local gradient: dz/dr2 = 2r (N,D)
    dz_dr2 = 2.0 * r
    dL_dr2 = dL_dz * dz_dr2  # (N,D)

    # Upstream gradienst: dL/dr1 (N,D) and dL/dr2 (N,D)
    # Node variable: r = x-mu (N,D)
    # Local gradient: dr/dx1 = 1 (N,D)
    # Local gradient: dr/dx2 = -1/N (N,D)
    dr_dx1 = 1.0 * np.ones((N, D))
    dr_mu = -1.0 * np.ones((N, D))

    dL_dx1 = (dL_dr1 + dL_dr2) * dr_dx1
    dL_mu = np.sum((dL_dr1 + dL_dr2) * dr_mu, axis=0)

    # Upstream gradienst: dL/dmu
    # Node variable: m = sum(x)/N (D,)
    # Local gradient: dm/dx2 = 1/N (D,)
    dmu_dx2 = np.ones((N, D)) * 1.0 / N
    dL_dx2 = dL_mu * dmu_dx2

    # Upstream gradient: dL/dx1 (D,N) and dL/dx1 (D,N)
    # Node variable: x=x (N,D)
    # Local gradient: 1
    dL_dx = dL_dx1 + dL_dx2

    dx = dL_dx
    dgamma = dL_dgamma
    dbeta = dL_dbeta

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, x_hat, m, var, gamma, vareps = cache

    sigma = np.sqrt(vareps)

    N, D = x.shape

    dL_dy = dout

    # X hat
    dy_dx_hat = gamma

    # Beta
    # dy_dbeta = np.ones(D, )  # Do this for clarity
    dL_dbeta = np.sum(dL_dy, axis=0)

    # Gamma
    dy_dgamma = x_hat
    dL_dgamma = np.sum(dL_dy * dy_dgamma, axis=0)

    dL_dx_hat = dL_dy * dy_dx_hat
    dL_dx_mu = np.sum(dL_dx_hat, axis=0)
    dL_dx_sigma = x_hat * np.sum(dL_dx_hat * x_hat, axis=0)
    dL_dx_x = N * dL_dx_hat

    dL_dx = dL_dx_x - dL_dx_mu - dL_dx_sigma
    dL_dx = (1.0 / (N * sigma)) * dL_dx
    # dx_hat_dx = (1 / (N * sigma)) * ((1 / N) * y * np.sum(y, axis=0) - y * np.sum(y, axis=0) - 1)

    # dL_dx = dL_dy * dy_dx_hat * dx_hat_dx

    dx = dL_dx
    dgamma = dL_dgamma / N
    dbeta = dL_dbeta / N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x = x.T

    m = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    # q = x - mean  # 1
    # qsq = q * q  # 2
    # var = np.sum(qsq, axis=0) / N

    # Normalize
    # Trick for division float to increase precision!!
    vareps = var + eps
    inden = 1.0 / np.sqrt(vareps)
    x_hat = (x - m) * inden

    # Allow distribution to shift
    out = gamma * x_hat.T + beta

    # out = out.T

    cache = [x, x_hat, m, var, gamma, vareps]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    [x, x_hat, m, var, gamma, vareps] = cache

    sigma = np.sqrt(vareps)
    r = (x - m)
    q = 1.0 / sigma

    N, D = x.shape

    dL_dy = dout

    # Beta
    # dy_dbeta = np.ones(D, )  # Do this for clarity
    dL_dbeta = np.sum(dL_dy, axis=0)

    # Gamma
    dy_dgamma = x_hat.T
    dL_dgamma = np.sum(dL_dy * dy_dgamma, axis=0)

    # X hat
    dy_dx_hat = gamma
    dL_dx_hat = dL_dy * dy_dx_hat
    dL_dx_hat = dL_dx_hat.T

    # Upstream gradient: dL/dx_hat (N,D)
    # Node variables: x_hat = r*q
    # Local gradient: dx_hat/dq = r (N,D)
    # Local gradient: dx_hat/dr = q (N,D)
    dx_hat_dq = r
    dx_hat_dr = np.ones((N, D)) * q

    dL_dq = np.sum(dL_dx_hat * dx_hat_dq, axis=0)
    dL_dr1 = dL_dx_hat * dx_hat_dr  # r1 since dL_dr is the sum of two gradients

    # Upstream gradient: dL/dq (D,)
    # Node variable: q = 1/sigma (sigma is the standard deviation)
    # Local gradient: dq/dsigma = - 1 / sigma**2 (D,)
    dq_dsigma = - 1.0 / sigma ** 2.0
    dL_dsigma = dL_dq * dq_dsigma

    # Upstream gradient: dL/dsigma (D,)
    # Node variable: sigma = sqrt(var)
    # Local gradient: dsigma/dvar = 1 / (2*sqrt(var)) (D,)
    dsigma_dvar = 1.0 / (2.0 * np.sqrt(vareps))
    dL_dvar = dL_dsigma * dsigma_dvar

    # Upstream gradient: dL/dvar (D,)
    # Node variable: var = z*1/N
    # Local gradient: dvar/dz =1/N*ones(D) (D,)
    dvar_dz = 1.0 / N * np.ones(D)
    dL_dz = dL_dvar * dvar_dz

    # Upstream gradient: dL/dz (D,)
    # Node variable: z = r^2 dim = (N,D)
    #                where r=x-m
    # Local gradient: dz/dr2 = 2r (N,D)
    dz_dr2 = 2.0 * r
    dL_dr2 = dL_dz * dz_dr2  # (N,D)

    # Upstream gradienst: dL/dr1 (N,D) and dL/dr2 (N,D)
    # Node variable: r = x-mu (N,D)
    # Local gradient: dr/dx1 = 1 (N,D)
    # Local gradient: dr/dx2 = -1/N (N,D)
    dr_dx1 = 1.0 * np.ones((N, D))
    dr_mu = -1.0 * np.ones((N, D))

    dL_dx1 = (dL_dr1 + dL_dr2) * dr_dx1
    dL_mu = np.sum((dL_dr1 + dL_dr2) * dr_mu, axis=0)

    # Upstream gradienst: dL/dmu
    # Node variable: m = sum(x)/N (D,)
    # Local gradient: dm/dx2 = 1/N (D,)
    dmu_dx2 = np.ones((N, D)) * 1.0 / N
    dL_dx2 = dL_mu * dmu_dx2

    # Upstream gradient: dL/dx1 (D,N) and dL/dx1 (D,N)
    # Node variable: x=x (N,D)
    # Local gradient: 1
    dL_dx = dL_dx1 + dL_dx2

    dx = dL_dx.T
    dgamma = dL_dgamma
    dbeta = dL_dbeta

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Drop mask
        mask = np.random.rand(*x.shape) < p

        # Drop
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x * p

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, _, H, W = x.shape
    F, C, HH, WW = w.shape

    pad, stride = conv_param['pad'], conv_param['stride']

    H_prime = int(1 + (H + 2 * pad - HH) / stride)
    W_prime = int(1 + (W + 2 * pad - WW) / stride)

    out = np.zeros((N, F, H_prime, W_prime))

    npad = ((0, 0), (0, 0), (1, 1), (1, 1))  # pad around x's 3rd and 4th dimensions
    x_pad = np.pad(x, pad_width=npad, mode='constant', constant_values=0)

    for n in range(N):
        for f in range(F):
            for h_stride in range(H_prime):
                for w_stride in range(W_prime):
                    x_region = x_pad[
                               n,
                               :,
                               h_stride * stride:h_stride * stride + HH,
                               w_stride * stride:w_stride * stride + WW
                               ]

                    out[n, f, h_stride, w_stride] = np.sum(w[f] * x_region) + b[f]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, w, b, conv_param = cache

    N, C, H, W = np.shape(x)
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    _, _, H_next, W_next = dout.shape

    npad = ((0, 0), (0, 0), (1, 1), (1, 1))  # pad around x's 3rd and 4th dimensions
    padded_x = np.pad(x, pad_width=npad, mode='constant', constant_values=0)

    db = np.sum(dout, axis=(0, 2, 3))
    dw = np.zeros(w.shape)
    dpadded_x = np.zeros(padded_x.shape)

    for n in range(N):
        for f in range(F):
            for h_stride in range(H_next):
                for v_stride in range(W_next):
                    dw[f] += padded_x[n, :,
                             h_stride * stride: h_stride * stride + HH,
                             v_stride * stride: v_stride * stride + WW] * dout[n, f, h_stride, v_stride]
                    dpadded_x[n, :, h_stride * stride: h_stride * stride + HH,
                    v_stride * stride: v_stride * stride + WW] += w[f] * dout[n, f, h_stride, v_stride]
    dx = dpadded_x[:, :, 1:-1, 1:-1]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    vertical_strides = int(1 + (H - pool_height) / stride)
    horizontal_strides = int(1 + (W - pool_width) / stride)

    out = np.zeros((N, C, vertical_strides, horizontal_strides))
    switches = {}

    for n in range(N):
        for c in range(C):
            for i in range(vertical_strides):
                for j in range(horizontal_strides):
                    region_of_x = x[n,
                                  c,
                                  i * stride: i * stride + pool_height,
                                  j * stride: j * stride + pool_width]
                    out[n, c, i, j] = np.max(region_of_x)
                    switches[n, c, i, j] = np.unravel_index(region_of_x.argmax(), region_of_x.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param, switches)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param, switches = cache

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    _, _, HH, WW = dout.shape

    dx = np.zeros(x.shape)

    for n in range(N):
        for c in range(C):
            for i in range(HH):
                for j in range(WW):
                    local_index_of_max = switches[n, c, i, j]
                    i_of_max = local_index_of_max[0] + i * stride
                    j_of_max = local_index_of_max[1] + j * stride
                    dx[n, c, i_of_max, j_of_max] += dout[n, c, i, j]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = np.shape(x)

    # https://www.reddit.com/r/cs231n/comments/443y2g/hints_for_a2 has hint about reshaping.
    # To use batchnorm_forward which expects dimensions of (N',D'), treat C as D'
    # and N*H*W as N'. Need to order values correctly with transpose.

    x_flattened = x.transpose(0, 2, 3, 1).reshape(-1, C)  # dim=(N*H*W, C)
    out_flattened, cache = batchnorm_forward(x_flattened, gamma, beta, bn_param)  # dim=(N*H*W, C)
    out = out_flattened.reshape(N, H, W, C).transpose(0, 3, 1, 2)  # dim=N,C,H,W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = np.shape(dout)

    dout_flattened = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx_flattened, dgamma, dbeta = batchnorm_backward(dout_flattened, cache)
    dx = dx_flattened.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.

    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = np.shape(x)

    # https://www.reddit.com/r/cs231n/comments/443y2g/hints_for_a2 has hint about reshaping.
    # To use batchnorm_forward which expects dimensions of (N',D'), treat C as D'
    # and N*H*W as N'. Need to order values correctly with transpose.

    group_width = int(C / G)

    if group_width != C / G:
        print("G needs to be a divisor of C")
        raise

    out = np.zeros((N, C, H, W))
    cache = []

    x_flattened = x.transpose(0, 2, 3, 1).reshape(-1, C)  # dim=(N*H*W, C)
    gamma_flatten = gamma.transpose(0, 2, 3, 1).reshape(-1, C)  #
    beta_flatten = beta.transpose(0, 2, 3, 1).reshape(-1, C)  #

    for g in range(G):
        x_flattened_group = x_flattened[:, g * group_width:(g + 1) * group_width]
        gamma_group = gamma_flatten[:, g * group_width:(g + 1) * group_width]
        beta_group = beta_flatten[:, g * group_width:(g + 1) * group_width]

        out_flattened_group, cache_group = layernorm_forward(x_flattened_group, gamma_group, beta_group,
                                                             gn_param)  # dim=(N*H*W, C)

        out[:, g * group_width:(g + 1) * group_width, :, :] = out_flattened_group.reshape(N, H, W,
                                                                                          group_width).transpose(0, 3,
                                                                                                                 1, 2)

        cache.append(cache_group)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = np.shape(dout)

    dx = np.zeros((N, C, H, W))
    dgamma = np.zeros((1, C, 1, 1))
    dbeta = np.zeros((1, C, 1, 1))

    G = len(cache)
    group_width = int(C / G)

    dout_flattened = dout.transpose(0, 2, 3, 1).reshape(-1, C)  # dim=(N*H*W, C)

    for g in range(G):
        dout_flattened_group = dout_flattened[:, g * group_width:(g + 1) * group_width]

        dx_group, dgamma_group, dbeta_group = layernorm_backward(dout_flattened_group, cache[g])  # dim=(N*H*W, C)

        dgamma[:, g * group_width:(g + 1) * group_width, :, :] = dgamma_group.reshape(1, 1, 1,
                                                                                      group_width).transpose(0, 3,
                                                                                                             1, 2)
        dbeta[:, g * group_width:(g + 1) * group_width, :, :] = dbeta_group.reshape(1, 1, 1,
                                                                                      group_width).transpose(0, 3,
                                                                                                             1, 2)
        dx[:, g * group_width:(g + 1) * group_width, :, :] = dx_group.reshape(N, H, W,
                                                                                          group_width).transpose(0, 3,
                                                                                                                 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
