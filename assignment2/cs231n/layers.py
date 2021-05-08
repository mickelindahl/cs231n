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

        mean = np.mean(x, axis=0)

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
        x_hat = (x - m)*inden

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

    # dbeta = dout.sum(axis=0)
    # dgamma = np.sum(x_norm * dout, axis=0)

    # dx_norm = gamma * dout  # 7
    # dq = invden * dx_norm  # 6
    # dinvden = np.sum(q * dx_norm, axis=0)  # 6
    # dden = (-1.0 / (den ** 2)) * dinvden  # 5
    # dvareps = (1.0) / (2.0 * np.sqrt(vareps)) * dden  # 4
    # dqsq = 1.0 / N * dvareps  # 3
    # dq += 2.0 * q * dqsq  # 2
    # dmean = np.sum(dq, axis=0) / N
    # dx = dq - dmean
    # #################

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta

# def batchnorm_forward(x, gamma, beta, bn_param):
#     """
#     Forward pass for batch normalization.
#     During training the sample mean and (uncorrected) sample variance are
#     computed from minibatch statistics and used to normalize the incoming data.
#     During training we also keep an exponentially decaying running mean of the
#     mean and variance of each feature, and these averages are used to normalize
#     data at test-time.
#     At each timestep we update the running averages for mean and variance using
#     an exponential decay based on the momentum parameter:
#     running_mean = momentum * running_mean + (1 - momentum) * sample_mean
#     running_var = momentum * running_var + (1 - momentum) * sample_var
#     Note that the batch normalization paper suggests a different test-time
#     behavior: they compute sample mean and variance for each feature using a
#     large number of training images rather than using a running average. For
#     this implementation we have chosen to use running averages instead since
#     they do not require an additional estimation step; the torch7
#     implementation of batch normalization also uses running averages.
#     Input:
#     - x: Data of shape (N, D)
#     - gamma: Scale parameter of shape (D,)
#     - beta: Shift paremeter of shape (D,)
#     - bn_param: Dictionary with the following keys:
#       - mode: 'train' or 'test'; required
#       - eps: Constant for numeric stability
#       - momentum: Constant for running mean / variance.
#       - running_mean: Array of shape (D,) giving running mean of features
#       - running_var Array of shape (D,) giving running variance of features
#     Returns a tuple of:
#     - out: of shape (N, D)
#     - cache: A tuple of values needed in the backward pass
#     """
#     mode = bn_param['mode']
#     eps = bn_param.get('eps', 1e-5)
#     momentum = bn_param.get('momentum', 0.9)
#
#     N, D = x.shape
#     running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
#     running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
#
#     out, cache = None, None
#     if mode == 'train':
#         #######################################################################
#         # TODO: Implement the training-time forward pass for batch norm.      #
#         # Use minibatch statistics to compute the mean and variance, use      #
#         # these statistics to normalize the incoming data, and scale and      #
#         # shift the normalized data using gamma and beta.                     #
#         #                                                                     #
#         # You should store the output in the variable out. Any intermediates  #
#         # that you need for the backward pass should be stored in the cache   #
#         # variable.                                                           #
#         #                                                                     #
#         # You should also use your computed sample mean and variance together #
#         # with the momentum variable to update the running mean and running   #
#         # variance, storing your result in the running_mean and running_var   #
#         # variables.                                                          #
#         #######################################################################
#         mean = np.mean(x, axis=0)
#
#         q = x - mean  # 1
#         qsq = q * q  # 2
#         var = np.sum(qsq, axis=0) / N
#         vareps = var + eps  # 3
#         den = np.sqrt(vareps)  # 4
#         invden = 1.0 / den  # 5
#         x_norm = q * invden  # 6
#         out = x_norm * gamma + beta  # 7
#
#         running_mean = momentum * running_mean + (1.0 - momentum) * mean
#         running_var = momentum * running_var + (1.0 - momentum) * var
#
#         cache = (x, gamma, beta, q, qsq, vareps, den, invden, x_norm)
#         #######################################################################
#         #                           END OF YOUR CODE                          #
#         #######################################################################
#     elif mode == 'test':
#         #######################################################################
#         # TODO: Implement the test-time forward pass for batch normalization. #
#         # Use the running mean and variance to normalize the incoming data,   #
#         # then scale and shift the normalized data using gamma and beta.      #
#         # Store the result in the out variable.                               #
#         #######################################################################
#         out = (x - running_mean) / np.sqrt(running_var + eps)
#         out = out * gamma + beta
#         #######################################################################
#         #                          END OF YOUR CODE                           #
#         #######################################################################
#     else:
#         raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
#
#     # Store the updated running means back into bn_param
#     bn_param['running_mean'] = running_mean
#     bn_param['running_var'] = running_var
#
#     return out, cache
#
#
# def batchnorm_backward(dout, cache):
#     """
#     Backward pass for batch normalization.
#     For this implementation, you should write out a computation graph for
#     batch normalization on paper and propagate gradients backward through
#     intermediate nodes.
#     Inputs:
#     - dout: Upstream derivatives, of shape (N, D)
#     - cache: Variable of intermediates from batchnorm_forward.
#     Returns a tuple of:
#     - dx: Gradient with respect to inputs x, of shape (N, D)
#     - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
#     - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
#     """
#     dx, dgamma, dbeta = None, None, None
#     ###########################################################################
#     # TODO: Implement the backward pass for batch normalization. Store the    #
#     # results in the dx, dgamma, and dbeta variables.                         #
#     ###########################################################################
#     N, D = dout.shape
#     x, gamma, beta, q, qsq, vareps, den, invden, x_norm = cache
#
#     dbeta = dout.sum(axis=0)
#     dgamma = np.sum(x_norm * dout, axis=0)
#
#     dx_norm = gamma * dout  # 7
#     dq = invden * dx_norm  # 6
#     dinvden = np.sum(q * dx_norm, axis=0)  # 6
#     dden = (-1.0 / (den ** 2)) * dinvden  # 5
#     dvareps = (1.0) / (2.0 * np.sqrt(vareps)) * dden  # 4
#     dqsq = 1.0 / N * dvareps  # 3
#     dq += 2.0 * q * dqsq  # 2
#     dmean = np.sum(dq, axis=0) / N
#     dx = dq - dmean
#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
#
#     return dx, dgamma, dbeta

# array([[-0.00310319,  0.00305468, -0.00156246,  0.17251307,  0.01388029],
#        [ 0.01147762, -0.10800884, -0.01112564, -0.02021632, -0.02098085],
#        [-0.01682492, -0.01106847, -0.00384286,  0.13581055, -0.04108612],
#        [ 0.00845049,  0.11602263,  0.01653096, -0.2881073 ,  0.04818669]])
# array([[-0.00310319,  0.00305468, -0.00156246,  0.17251307,  0.01388029],
#        [ 0.01147762, -0.10800884, -0.01112564, -0.02021632, -0.02098085],
#        [-0.01682492, -0.01106847, -0.00384286,  0.13581055, -0.04108612],
#        [ 0.00845049,  0.11602263,  0.01653096, -0.2881073 ,  0.04818669]])
# array([2.29048278, 1.39248907, 2.93350569, 0.98234546, 2.08326113])
# array([2.29048278, 1.39248907, 2.93350569, 0.98234546, 2.08326113])
# array([ 0.08461601,  0.59073617,  1.2668311 , -1.75428014, -0.80317214])
# array([ 0.08461601,  0.59073617,  1.2668311 , -1.75428014, -0.80317214])
# dx error:  1.6674604875341426e-09
# dgamma error:  7.417225040694815e-13
# dbeta error:  2.379446949959628e-12

# array([[-0.00310319,  0.00305468, -0.00156246,  0.17251307,  0.01388029],
#        [ 0.01147762, -0.10800884, -0.01112564, -0.02021632, -0.02098085],
#        [-0.01682492, -0.01106847, -0.00384286,  0.13581055, -0.04108612],
#        [ 0.00845049,  0.11602263,  0.01653096, -0.2881073 ,  0.04818669]])
# array([[-0.0031032 ,  0.00305466, -0.00156246,  0.17251311,  0.0138803 ],
#        [ 0.01147763, -0.10800888, -0.01112564, -0.02021631, -0.02098084],
#        [-0.01682493, -0.01106847, -0.00384286,  0.13581057, -0.04108615],
#        [ 0.00845049,  0.11602269,  0.01653096, -0.28810737,  0.04818669]])
# array([2.29048278, 1.39248907, 2.93350569, 0.98234546, 2.08326113])
# array([2.29048278, 1.39248907, 2.93350569, 0.98234546, 2.08326113])
# array([ 0.08461601,  0.59073617,  1.2668311 , -1.75428014, -0.80317214])
# array([ 0.08461601,  0.59073617,  1.2668311 , -1.75428014, -0.80317214])
# dx error:  3.6789253252687406e-06
# dgamma error:  7.417225040694815e-13
# dbeta error:  2.379446949959628e-12

# Before batch normalization:
#   means: [ -2.3814598  -13.18038246   1.91780462]
#   stds:  [27.18502186 34.21455511 37.68611762]
#
# After batch normalization (gamma=1, beta=0)
#   means: [5.99520433e-17 6.93889390e-17 8.32667268e-19]
#   stds:  [0.99999999 1.         1.        ]
#
# After batch normalization (gamma= [1. 2. 3.] , beta= [11. 12. 13.] )
#   means: [11. 12. 13.]
#   stds:  [0.99999999 1.99999999 2.99999999]

# def batchnorm_backward_alt(dout, cache):
#     """
#     Alternative backward pass for batch normalization.
#     For this implementation you should work out the derivatives for the batch
#     normalizaton backward pass on paper and simplify as much as possible. You
#     should be able to derive a simple expression for the backward pass.
#     Note: This implementation should expect to receive the same cache variable
#     as batchnorm_backward, but might not use all of the values in the cache.
#     Inputs / outputs: Same as batchnorm_backward
#     """
#     dx, dgamma, dbeta = None, None, None
#     ###########################################################################
#     # TODO: Implement the backward pass for batch normalization. Store the    #
#     # results in the dx, dgamma, and dbeta variables.                         #
#     #                                                                         #
#     # After computing the gradient with respect to the centered inputs, you   #
#     # should be able to compute gradients with respect to the inputs in a     #
#     # single statement; our implementation fits on a single 80-character line.#
#     ###########################################################################
#     N, D = dout.shape
#     x, gamma, beta, q, qsq, vareps, den, invden, x_norm = cache
#
#     dbeta = dout.sum(axis=0)
#     dgamma = np.sum(x_norm * dout, axis=0)
#
#     dx_norm = gamma * dout
#     dldvar = np.sum(dx_norm*q*(invden**3), axis=0)*(-1.0/2.0)
#     dldmean = np.sum(dx_norm, axis=0)*(-1.0/den) + dldvar * (-2.0/N * np.sum(q, axis=0))
#     dx = dx_norm * invden + dldvar * 2.0 * q / N + dldmean / N
#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
#
#     return dx, dgamma, dbeta

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
    r = (x - m)
    q = 1.0 / sigma

    N, D = x.shape

    dx_hat = x

    dL_dy = dout

    # X hat
    dy_dx_hat = gamma

    # Beta
    #dy_dbeta = np.ones(D, )  # Do this for clarity
    dL_dbeta = np.sum(dL_dy, axis=0)

    # Gamma
    dy_dgamma = x_hat
    dL_dgamma = np.sum(dL_dy * dy_dgamma, axis=0)

    Y = dx_hat
    X = x

    dL_dx_hat = dL_dy * dy_dx_hat
    dL_dx_mu = np.sum(dL_dx_hat, axis=0)
    dL_dx_sigma = x_hat * np.sum(dL_dx_hat * x_hat, axis=0)
    dL_dx_x = N * dL_dx_hat

    dL_dx =  dL_dx_x - dL_dx_mu - dL_dx_sigma
    dL_dx = (1.0 / (N * sigma)) * dL_dx
    # dx_hat_dx = (1 / (N * sigma)) * ((1 / N) * y * np.sum(y, axis=0) - y * np.sum(y, axis=0) - 1)

    # dL_dx = dL_dy * dy_dx_hat * dx_hat_dx

    dx = dL_dx
    dgamma = dL_dgamma
    dbeta = dL_dbeta

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

    pass

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

    pass

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

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

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

        pass

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

    pass

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

    pass

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

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
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

    pass

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

    pass

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

    pass

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

    pass

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

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta

# array([[-0.02960701, -0.04014999, -0.0137734 , ..., -0.01035652,
#         -0.06595021,  0.02386322],
#        [ 0.03417437, -0.00763451, -0.01391406, ...,  0.01649333,
#          0.00942428,  0.0223573 ],
#        [-0.01341895,  0.05564852,  0.02808692, ..., -0.0048158 ,
#          0.02397009, -0.01430126],
#        ...,
#        [-0.02294035,  0.00099297, -0.0082513 , ...,  0.01252861,
#         -0.00837394,  0.01874283],
#        [-0.00735982, -0.0200003 ,  0.01613967, ..., -0.00858063,
#          0.00171868, -0.0560393 ],
#        [-0.00327935, -0.02256672, -0.01453427, ...,  0.01332806,
#          0.01768391, -0.00632291]])
# array([[-0.02959813, -0.04013853, -0.0137686 , ..., -0.01035517,
#         -0.06593016,  0.02385825],
#        [ 0.03416299, -0.00763265, -0.01390921, ...,  0.0164905 ,
#          0.00942333,  0.02235497],
#        [-0.0134143 ,  0.05563422,  0.02807613, ..., -0.0048141 ,
#          0.02396556, -0.01430169],
#        ...,
#        [-0.02293491,  0.00099238, -0.00824874, ...,  0.01252725,
#         -0.00837345,  0.01873625],
#        [-0.00735852, -0.01999094,  0.0161346 , ..., -0.00857972,
#          0.0017192 , -0.05602468],
#        [-0.00327846, -0.02256088, -0.01453091, ...,  0.01332299,
#          0.01767755, -0.00632105]])
# array([[-8.88123495e-06, -1.14577644e-05, -4.79527212e-06, ...,
#         -1.34837530e-06, -2.00468161e-05,  4.96476793e-06],
#        [ 1.13792874e-05, -1.86045111e-06, -4.84949484e-06, ...,
#          2.82801311e-06,  9.50877344e-07,  2.33566612e-06],
#        [-4.65370145e-06,  1.42937617e-05,  1.07925459e-05, ...,
#         -1.70069138e-06,  4.53215983e-06,  4.23613115e-07],
#        ...,
#        [-5.43791235e-06,  5.84137396e-07, -2.55814148e-06, ...,
#          1.36110385e-06, -4.95509176e-07,  6.57652126e-06],
#        [-1.30112940e-06, -9.35723288e-06,  5.07196005e-06, ...,
#         -9.07293422e-07, -5.24505621e-07, -1.46220191e-05],
#        [-8.83003612e-07, -5.83173014e-06, -3.35995805e-06, ...,
#          5.07807911e-06,  6.35971786e-06, -1.85778139e-06]])