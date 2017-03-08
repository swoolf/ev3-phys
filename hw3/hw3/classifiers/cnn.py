import numpy as np

from hw3.layers import *
from hw3.conv_layers import *
from hw3.fast_layers import *
from hw3.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels. In this convnet, the convolutional layer doesn't change the
  image size.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, batchnorm=False):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.batchnorm=batchnorm
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C,H,W = input_dim
    F=num_filters
    #Conv Layer
    stride = 1
    Hout = H #b/c convnet doesn't change size
    Wout = W
    
    #in (N,C,H,W)
    #out (N,F,Hout,Wout)
    W1 = weight_scale*np.random.randn(F,C,filter_size,filter_size)
    b1 = np.zeros(F)
    
    #2x2 Max Pool
    #in (N,F,Hout,Wout)
    #out (N,F,Hout/2,Wout/2)
    
    #Affine1
    #convert to lin vector
    #in (N,F*Hout*Wout/4)
    #out (N,hidden_dim)
    W2= weight_scale*np.random.randn(F*Hout*Wout/4,hidden_dim)
    b2=np.zeros(hidden_dim)
    
    #Affine2
    #in (N, hidden_dim)
    #out (N, C)
    
    W3=weight_scale*np.random.randn(hidden_dim,num_classes)
    b3=np.zeros(num_classes)
    
    self.params.update({'W1':W1,'W2':W2,'W3':W3,'b1':b1,'b2':b2,'b3':b3})
    
    if self.batchnorm:
        print 'using batchnorm'
        #for conv layer
        bn1 = {'mean':np.zeros(F),'var':np.zeros(F)}
        gamma1 = np.ones(F)
        beta1 = np.zeros(F)
        
        #for affine layer
        bn2 = {'mean':np.zeros(F),'var':np.zeros(F)}
        gamma2 = np.ones(F)
        beta2 = np.zeros(F)
    
        self.params.update({'beta1':beta1,'gamma1':gamma1,'beta2':beta2,'gamma2':gamma2,'bn1':bn1,'bn2':bn2})
    
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    if self.batchnorm:
        bn1, gamma1, beta1 = self.params['bn1'], self.params['gamma1'], self.params['beta1']
        bn2, gamma2, beta2 = self.params['bn2'], self.params['gamma2'], self.params['beta2']
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    #conv layer
#    print X.shape, W1.shape, b1.shape
    if self.batchnorm:
        -----------------------------
    else:
    out, conv_cache = conv_relu_pool_forward(X,W1,b1, conv_param, pool_param)
#    conv_cache, relu_cache, pool_cache=cache
#    l1_cache = conv_cache, relu_cache
    #affine layer1
    N,F,H,W = out.shape
    X2 = out.reshape(N, F*H*W)
    
    X3, X3_cache = affine_relu_forward(X2,W2,b2)
    
    #affine layer2
    scores, scores_cache = affine_forward(X3, W3, b3)


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    
    #Calculate Loss
    loss, dscores = softmax_loss(scores, y)
    reg_loss = .5*self.reg*(np.sum(W1**2)+np.sum(W2**2)+np.sum(W3**2))
    loss+=reg_loss
    
    #calculate gradient
    dX3, dW3, db3 = affine_backward(dscores, scores_cache)
    dW3 += self.reg*W3
    
    dX2, dW2, db2 = affine_relu_backward(dX3, X3_cache)
    dW2+=self.reg*W2
    
    dX2 = dX2.reshape(N,F,H,W)
    dX1,dW1,db1= conv_relu_pool_backward(dX2, conv_cache)
    dW1+=self.reg*W1
    
    grads.update({'W1':dW1, 'W2':dW2,'W3':dW3,'b1':db1,'b2':db2,'b3':db3})
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads














