from hw3.layers import *
from hw3.conv_layers import *
from hw3.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db

#Referenced http://cthorey.github.io./ for the following:
def conv_bn_relu_pool_forward(x, w, b, conv_param, pool_param, beta, gamma, bn_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.
    
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    
    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    norm, norm_cache = spatial_batchnorm_forward(a,gamma,beta, bn_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, norm_cache, relu_cache, pool_cache)
    return out, cache

def conv_bn_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, norm_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dnorm, dgamma, dbeta = spatial_batchnorm_backward(da, norm_cache)
    dx, dw, db = conv_backward_fast(dnorm, conv_cache)
    return dx, dw, db, dgamma, dbeta

def affine_bn_relu_forward(x, w, b, bn_params=False, dr_params=False):
    """
    Convenience layer that perorms an batchnorm transform followed by a ReLU
    
    Inputs:
    - x: Data of shape (N, D)
    - w, b: Weights for the affine layer
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_params: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var: Array of shape (D,) giving running variance of features
    - dr_params: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - seed: If not None, then pass this random seed to the dropout layers. This
    will make the dropout layers deterministic so we can gradient check the
    model.
    - p: Scalar between 0 and 1 giving dropout strength. If equal to 0 then
    the network should not use dropout at all.
    
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
        
    a, fc_cache = affine_forward(x, w, b)
        
    if bn_params:
        gamma, beta, bn_param = bn_params
        a, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
            
    out, relu_cache = relu_forward(a)
        
    if dr_params:
        out, dr_cache = dropout_forward(out, dr_params)
            
    cache = {}
    cache['fc'] = fc_cache
    cache['relu'] = relu_cache
        
    if bn_params:
        cache['bn'] = bn_cache
    if dr_params:
        cache['drop'] = dr_cache
                        
    return out, cache


def affine_bn_relu_backward(dout, cache, bn_params=False, dr_params=False):
    """
    Backward pass for the bn-relu convenience layer
    """
        
    fc_cache = cache['fc']
    relu_cache = cache['relu']
        
    if dr_params:
        dr_cache = cache['drop']
        dout = dropout_backward(dout, dr_cache)
            
    da = relu_backward(dout, relu_cache)
        
    opts = None
    if bn_params:
        bn_cache = cache['bn']
        da, dgamma, dbeta = batchnorm_backward(da, bn_cache)
        opts = dgamma, dbeta
            
    dx, dw, db = affine_backward(da, fc_cache)
        
    return dx, dw, db, opts
