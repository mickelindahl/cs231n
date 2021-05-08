from builtins import range
from builtins import object
import numpy as np
import pprint

from ..layers import *
from ..layer_utils import *

pp = pprint.pprint

# def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
#     a_out, a_cache = affine_forward(x, w, b)
#     b_out, b_cache = batchnorm_forward(a_out, gamma, beta, bn_param)
#     out, r_cache = relu_forward(b_out)
#     cache = (a_cache, b_cache, r_cache)
#     return out, cache
#
#
# def affine_batchnorm_relu_backward(dout, cache):
#     a_cache, b_cache, r_cache = cache
#     dx1 = relu_backward(dout, r_cache)
#     dx2, dgamma, dbeta = batchnorm_backward(dx1, b_cache)
#     dx, dw, db = affine_backward(dx2, a_cache)
#     return dx, dw, db, dgamma, dbeta
#
# class FullyConnectedNet(object):
#     """
#     A fully-connected neural network with an arbitrary number of hidden layers,
#     ReLU nonlinearities, and a softmax loss function. This will also implement
#     dropout and batch normalization as options. For a network with L layers,
#     the architecture will be
#     {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
#     where batch normalization and dropout are optional, and the {...} block is
#     repeated L - 1 times.
#     Similar to the TwoLayerNet above, learnable parameters are stored in the
#     self.params dictionary and will be learned using the Solver class.
#     """
#
#     def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
#                  dropout=0, normalization=False, reg=0.0,
#                  weight_scale=1e-2, dtype=np.float32, seed=None):
#         """
#         Initialize a new FullyConnectedNet.
#         Inputs:
#         - hidden_dims: A list of integers giving the size of each hidden layer.
#         - input_dim: An integer giving the size of the input.
#         - num_classes: An integer giving the number of classes to classify.
#         - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
#           the network should not use dropout at all.
#         - use_batchnorm: Whether or not the network should use batch normalization.
#         - reg: Scalar giving L2 regularization strength.
#         - weight_scale: Scalar giving the standard deviation for random
#           initialization of the weights.
#         - dtype: A numpy datatype object; all computations will be performed using
#           this datatype. float32 is faster but less accurate, so you should use
#           float64 for numeric gradient checking.
#         - seed: If not None, then pass this random seed to the dropout layers. This
#           will make the dropout layers deteriminstic so we can gradient check the
#           model.
#         """
#         use_batchnorm = normalization
#         self.use_batchnorm = normalization
#         self.use_dropout = dropout > 0
#         self.reg = reg
#         self.num_layers = 1 + len(hidden_dims)
#         self.dtype = dtype
#         self.params = {}
#
#         ############################################################################
#         # TODO: Initialize the parameters of the network, storing all values in    #
#         # the self.params dictionary. Store weights and biases for the first layer #
#         # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
#         # initialized from a normal distribution with standard deviation equal to  #
#         # weight_scale and biases should be initialized to zero.                   #
#         #                                                                          #
#         # When using batch normalization, store scale and shift parameters for the #
#         # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
#         # beta2, etc. Scale parameters should be initialized to one and shift      #
#         # parameters should be initialized to zero.                                #
#         ############################################################################
#         all_dims = [input_dim] + hidden_dims + [num_classes]
#
#         for idx in range(self.num_layers):
#             in_d, out_d = all_dims[idx:(idx + 2)]
#             self.params["W%d" % (idx + 1)] = np.random.normal(0.0, weight_scale, (in_d, out_d))
#             self.params["b%d" % (idx + 1)] = np.zeros(out_d, dtype=float)
#
#         if use_batchnorm:
#             for idx, dim in enumerate(hidden_dims):
#                 self.params["gamma%d" % (idx + 1)] = np.ones(dim, dtype=float)
#                 self.params["beta%d" % (idx + 1)] = np.zeros(dim, dtype=float)
#
#         ####################### ######################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################
#
#         # When using dropout we need to pass a dropout_param dictionary to each
#         # dropout layer so that the layer knows the dropout probability and the mode
#         # (train / test). You can pass the same dropout_param to each dropout layer.
#         self.dropout_param = {}
#         if self.use_dropout:
#             self.dropout_param = {'mode': 'train', 'p': dropout}
#             if seed is not None:
#                 self.dropout_param['seed'] = seed
#
#         # With batch normalization we need to keep track of running means and
#         # variances, so we need to pass a special bn_param object to each batch
#         # normalization layer. You should pass self.bn_params[0] to the forward pass
#         # of the first batch normalization layer, self.bn_params[1] to the forward
#         # pass of the second batch normalization layer, etc.
#         self.bn_params = []
#         if self.use_batchnorm:
#             self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
#
#         # Cast all parameters to the correct datatype
#         for k, v in self.params.items():
#             self.params[k] = v.astype(dtype)
#
#     def loss(self, X, y=None):
#         """
#         Compute loss and gradient for the fully-connected net.
#         Input / output: Same as TwoLayerNet above.
#         """
#         X = X.astype(self.dtype)
#         mode = 'test' if y is None else 'train'
#
#         # Set train/test mode for batchnorm params and dropout param since they
#         # behave differently during training and testing.
#         if self.dropout_param is not None:
#             self.dropout_param['mode'] = mode
#         if self.use_batchnorm:
#             for bn_param in self.bn_params:
#                 bn_param[mode] = mode
#
#         scores = None
#         ############################################################################
#         # TODO: Implement the forward pass for the fully-connected net, computing  #
#         # the class scores for X and storing them in the scores variable.          #
#         #                                                                          #
#         # When using dropout, you'll need to pass self.dropout_param to each       #
#         # dropout forward pass.                                                    #
#         #                                                                          #
#         # When using batch normalization, you'll need to pass self.bn_params[0] to #
#         # the forward pass for the first batch normalization layer, pass           #
#         # self.bn_params[1] to the forward pass for the second batch normalization #
#         # layer, etc.                                                              #
#         ############################################################################
#         cachelist = []
#         out = X
#         for idx in range(self.num_layers):
#             w = self.params["W%d" % (idx + 1)]
#             b = self.params["b%d" % (idx + 1)]
#             if idx == self.num_layers - 1:
#                 out, cache = affine_forward(out, w, b)
#             elif self.use_batchnorm:
#                 gamma = self.params["gamma%d" % (idx + 1)]
#                 beta = self.params["beta%d" % (idx + 1)]
#                 out, cache = affine_batchnorm_relu_forward(out, w, b, gamma, beta,
#                     self.bn_params[idx])
#             else:
#                 out, cache = affine_relu_forward(out, w, b)
#
#             if not idx == self.num_layers - 1 and self.use_dropout:
#                 out, cache_do = dropout_forward(out,self.dropout_param)
#                 cache = (cache, cache_do)
#
#             cachelist.append(cache)
#
#         scores = out
#         ############################################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################
#
#         # If test mode return early
#         if mode == 'test':
#             return scores
#
#         loss, grads = 0.0, {}
#         ############################################################################
#         # TODO: Implement the backward pass for the fully-connected net. Store the #
#         # loss in the loss variable and gradients in the grads dictionary. Compute #
#         # data loss using softmax, and make sure that grads[k] holds the gradients #
#         # for self.params[k]. Don't forget to add L2 regularization!               #
#         #                                                                          #
#         # When using batch normalization, you don't need to regularize the scale   #
#         # and shift parameters.                                                    #
#         #                                                                          #
#         # NOTE: To ensure that your implementation matches ours and you pass the   #
#         # automated tests, make sure that your L2 regularization includes a factor #
#         # of 0.5 to simplify the expression for the gradient.                      #
#         ############################################################################
#         loss, dscores = softmax_loss(scores, y)
#
#         dout = dscores
#         for idx in reversed(range(self.num_layers)):
#             cache = cachelist[idx]
#             if idx == self.num_layers - 1:
#                 dout, dw, db = affine_backward(dout, cache)
#             else:
#                 if self.use_dropout:
#                     cache, cache_do = cache
#                     dout = dropout_backward(dout, cache_do)
#
#                 if self.use_batchnorm:
#                     dout, dw, db, dgamma, dbeta = affine_batchnorm_relu_backward(dout, cache)
#                     grads["gamma%d" % (idx + 1)] = dgamma
#                     grads["beta%d" % (idx + 1)] = dbeta
#                 else:
#                     dout, dw, db = affine_relu_backward(dout, cache)
#
#             grads["W%d" % (idx + 1)] = dw
#             grads["b%d" % (idx + 1)] = db
#
#         for idx in range(self.num_layers):
#             w = "W%d" % (idx + 1)
#             if self.reg > 0:
#                 loss += 0.5 * self.reg * (self.params[w] ** 2).sum()
#                 grads[w] += self.reg * self.params[w]
#
#         ############################################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################
#
#         return loss, grads

class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
            self,
            hidden_dims,
            input_dim=3 * 32 * 32,
            num_classes=10,
            dropout_keep_ratio=1,
            normalization=None,
            reg=0.0,
            weight_scale=1e-2,
            dtype=np.float32,
            seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dimensions = [input_dim] + hidden_dims + [num_classes]

        self.params = {}

        for i in range(1, len(dimensions)):
            self.params['W' + str(i)] = weight_scale * np.random.randn(dimensions[i - 1], dimensions[i])
            self.params['b' + str(i)] = np.zeros(dimensions[i])

            if self.normalization == "batchnorm":
                self.params['gamma' + str(i)] = np.ones(dimensions[i])
                self.params['beta' + str(i)] = np.zeros(dimensions[i])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        caches = {
            'affine': {},
            'relu': {},
            'batchnorm': {}
        }

        out = X
        for i in range(1, self.num_layers + 1):

            W_key = 'W' + str(i)
            b_key = 'b' + str(i)
            gamma_key = 'gamma' + str(i)
            beta_key = 'beta' + str(i)

            out, cache = affine_forward(
                out,
                self.params[W_key],
                self.params[b_key]
            )

            caches['affine'][i] = cache

            # No relu for last activation
            if i == self.num_layers:
                # print('last')
                continue

            if self.normalization == "batchnorm":
                out, cache = batchnorm_forward(
                    out,
                    self.params[gamma_key],
                    self.params[beta_key],
                    self.bn_params[i - 1]
                )
                caches['batchnorm'][i] = cache

            out, cache = relu_forward(out)
            caches['relu'][i] = cache

        scores = out

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test" or y is None:
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Use last activation
        loss, dx = softmax_loss(scores, y)

        for i in range(self.num_layers, 0, -1):

            W_key = 'W' + str(i)
            b_key = 'b' + str(i)
            gamma_key = 'gamma' + str(i)
            beta_key = 'beta' + str(i)

            # Add relu if hidden layer
            if i != self.num_layers:
                dx = relu_backward(dx, caches['relu'][i])

                if self.normalization == "batchnorm":
                    dx, dgamma, dbeta = batchnorm_backward_alt(dx, caches['batchnorm'][i])
                    grads[gamma_key] = dgamma
                    grads[beta_key] = dbeta

            dx, dw, db = affine_backward(dx, caches['affine'][i])

            loss += np.sum(self.reg * self.params[W_key] ** 2)  # regularization
            grads[W_key] = dw + self.reg * 2 * self.params[W_key]
            grads[b_key] = db  # + self.reg * np.ones(db.shape)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

#