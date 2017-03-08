import numpy as np


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width WW.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  stride, pad = conv_param['stride'], conv_param['pad']
  
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant')
  N,C,H,W = x.shape
  F,C,HH,WW = w.shape
  
  #output width and height
  Hout = 1+( H+2*pad - HH)/stride
  Wout = 1+( W+2*pad - WW)/stride
  
#  print Hout, Wout
  out = np.zeros( (N,F,Hout,Wout) )
  
  
  
  for n in range(N): #for each sample
      for f in range(F): #for each filter
  
          for i in range(Hout):
              for j in range(Wout):
                  box = xpad[n,:, i*stride:i*stride+ HH, j*stride: j*stride+WW]
                  out[n,f,i,j]=np.sum( box*w[f,:]) + b[f]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  #Get Values
  x,w,b,conv_param = cache
  N,C,H,W = x.shape
  F,C,HH,WW = w.shape
  N,F,Hout,Wout = dout.shape
  pad = conv_param['pad']
  stride = conv_param['stride']
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant')
  doutpad=np.pad(dout, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant')
  Hout = 1+( H+2*pad - HH)/stride
  Wout = 1+( W+2*pad - WW)/stride

  db=np.zeros_like(b)
  dxpad=np.zeros_like(xpad)
  dw=np.zeros_like(w)


  for f in range(F):
      db[f]=np.sum(dout[:,f,:,:])
      for n in range(N):
          for i in range(Hout):
              for j in range(Wout):
                  dw[f,:,:,:] += xpad[n,:,i*stride:i*stride+HH, j*stride:j*stride+WW]*dout[n,f,i,j]
                  dxpad[n,:,i*stride:i*stride+HH, j*stride:j*stride+WW] += dout[n,f,i,j]*w[f,:,:,:]

  dx = dxpad[:,:,pad:H+pad,pad:W+pad]


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param, switches)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  ph = pool_param['pool_height']
  pw = pool_param['pool_width']
  stride = pool_param['stride']
  N,C,H,W = x.shape
  Hout = 1+( H - ph)/stride
  Wout = 1+( W - pw)/stride
  
  out=np.zeros( (N,C,Hout,Wout))
  
  for n in range(N):
      for c in range(C):
          for i in range(Hout):
              for j in range(Wout):
                  out[n,c,i,j]=np.max(x[n,c,i*stride:i*stride+ph,j*stride:j*stride+pw])
  switches = None
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param, switches)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param, switches) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param, switches = cache
  ph = pool_param['pool_height']
  pw = pool_param['pool_width']
  stride = pool_param['stride']
  N,C,H,W = x.shape
  
  dx = np.zeros_like(x)
  for n in range(N):
      for c in range(C):
          for i in range(dout.shape[2]):
              for j in range(dout.shape[3]):
                  
                  max = np.max(x[n,c, i*stride:i*stride+ph, j*stride:j*stride+pw])
                  maxmat = (x[n,c, i*stride:i*stride+ph, j*stride:j*stride+pw ]==max)
#                  print maxmat

                  dx[n,c, i*stride:i*stride+ph, j*stride:j*stride+pw] += dout[n,c,i,j]*maxmat

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
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
    - running_mean: Array of shape (C,) giving running mean of features
    - running_var Array of shape (C,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined in hw2. Your implementation should #
  # be very short; ours is less than five lines.                              #
  #############################################################################
   #Referenced: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

  N,C,H,W = x.shape
  eps = bn_param.get('eps',0)
  mu = np.sum(x, axis=(0,2,3) ).reshape(1,C,1,1)/(1.*N*H*W)
  var = np.sum((x-mu)**2, axis=(0,2,3) ).reshape(1,C,1,1)/(1.*N*H*W)
  xhat = (x-mu)/np.sqrt(var+eps)
  out = gamma.reshape(1,C,1,1)*xhat+beta.reshape(1,C,1,1)
  cache = mu,var,xhat,x,beta,gamma,bn_param

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined in hw2. Your implementation should #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  #referenced http://cthorey.github.io./backpropagation/
  mu,var,xhat,x,beta,gamma,bn_param = cache
  
  eps = bn_param.get('eps',0)
  N,C,H,W = x.shape
  
  dbeta = np.sum(dout,axis=(0,2,3))
  dgamma=np.sum(np.multiply(xhat,dout), axis=(0,2,3) )
  
  dx = (1. / (N*H*W)) * gamma.reshape(1,C,1,1) * (var + eps)**(-1. / 2.) * (
         N*H*W * dout
         - np.sum(dout, axis=(0,2,3)).reshape(1,C,1,1)
         - (x - mu) * (var + eps)**(-1.0) * np.sum(dout * (x - mu), axis=(0,2,3)).reshape(1,C,1,1)
           )

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  
