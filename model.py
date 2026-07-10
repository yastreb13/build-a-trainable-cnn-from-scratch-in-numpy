"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
def argmax_rows(matrix):
    result=[]
    for rows in matrix:
        result.append(np.argmax(rows))
    return np.array(result)

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    return np.array([matrix.max(axis=1)]).transpose()

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    return np.array(matrix.sum(axis=1,keepdims=True))

# Step 4 - exp_shifted
import numpy as np

def exp_shifted(logits):
    return np.exp(logits-row_max(logits))

# Step 5 - stable_softmax
def stable_softmax(logits):
   return exp_shifted(logits)/exp_shifted(logits).sum(axis=1,keepdims=True)

# Step 6 - one_hot
def one_hot(labels, num_classes):
    seen={}
    result=np.zeros((len(labels),num_classes))
    # j=0
    for i in range(len(labels)):
        j = labels[i]
        if not (labels[i] in seen):
            seen[labels[i]]=max
            result[i][j]=1.0
            # j+=1
        else:
            result[i][j]=1.0
    return result

# Step 7 - gather_true_class_probs
def gather_true_class_probs(probs, labels):
    result=[]
    for i in range(len(labels)):
        result.append(probs[i][labels[i]])
    return np.array(result)

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    gathered_probs = gather_true_class_probs(probs, labels)
    gathered_probs = np.clip(gathered_probs, eps, 1.0)
    return -np.mean(np.log(gathered_probs))

# Step 9 - accuracy
def accuracy(logits_or_probs, labels):
    # TODO: return the fraction of rows whose argmax matches the integer label.
    res=argmax_rows(logits_or_probs)
    return np.mean(res == labels)

# Step 10 - he_std
def he_std(fan_in):
    # TODO: return the He initialization standard deviation sqrt(2 / fan_in).
    return np.sqrt(2.0 / fan_in)

# Step 11 - he_init
def he_init(shape, fan_in, seed):
    np.random.seed(seed)
    std=he_std(fan_in)
    weights=np.random.normal(loc=0.0, scale=std, size=shape)
    return weights

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    return np.zeros((length,)).T

# Step 13 - pad_2d
def pad_2d(images, pad):
    if pad == 0:
        return images
    return np.pad(images,((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant')

# Step 14 - output_spatial_size
def output_spatial_size(input_size, kernel, stride=1, padding=0):
    return int((input_size-kernel+2*padding)/stride+1)

# Step 15 - im2col
def im2col(images, kernel_h, kernel_w, stride, padding):
    N,C,H,W=images.shape
    images=pad_2d(images,padding)
    H_out=output_spatial_size(H,kernel_h,stride,padding)
    W_out=output_spatial_size(W,kernel_w,stride,padding)
    out = np.zeros((N * H_out * W_out, C * kernel_h * kernel_w), dtype=images.dtype)
    row_idx = 0
    for n in range(N):
        for y in range(H_out):
                for x in range(W_out):
                    y_start=y*stride
                    x_start=x*stride
                    y_end=y_start+kernel_h
                    x_end=x_start+kernel_w
                    patch=images[n,:,y_start:y_end,x_start:x_end]
                    out[row_idx, :] = patch.flatten()
                    row_idx += 1
    return out

# Step 16 - col2im
def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    N, C, H, W = input_shape
    H_out=output_spatial_size(H,kernel_h,stride,padding)
    W_out=output_spatial_size(W,kernel_w,stride,padding)
    padded_images = np.zeros((N, C, H + 2 * padding, W + 2 * padding))
    row_idx = 0
    for n in range(N):
        for y in range(H_out):
                for x in range(W_out):
                    y_start = y * stride
                    y_end = y_start + kernel_h
                    x_start = x * stride
                    x_end = x_start + kernel_w
                    patch = cols[row_idx, :].reshape(C, kernel_h, kernel_w)
                    padded_images[n, :, y_start:y_end, x_start:x_end] += patch
                
                    row_idx += 1
    if padding > 0:
        return padded_images[:, :, padding:-padding, padding:-padding]
    else:
        return padded_images

# Step 17 - conv2d_forward
import numpy as np

def conv2d_forward(x, weights, bias, stride, padding):
    N, C, H, W = x.shape
    F, C_w, kernel_h, kernel_w = weights.shape
    out_h = (H + 2 * padding - kernel_h) // stride + 1
    out_w = (W + 2 * padding - kernel_w) // stride + 1
    cols = im2col(x, kernel_h, kernel_w, stride, padding)
    weights_flat = weights.reshape(F, -1)
    out_2d = np.dot(cols, weights_flat.T) 
    out_2d += bias
    out_4d = out_2d.reshape(N, out_h, out_w, F)
    out = out_4d.transpose(0, 3, 1, 2)
    cache = {
        'x_shape': x.shape,
        'weights': weights,
        'cols': cols,
        'stride': stride,
        'padding': padding,
        'kernel_h': kernel_h,
        'kernel_w': kernel_w
    }
    
    return out, cache

# Step 18 - conv2d_grad_input
def conv2d_grad_input(d_out, cache):
    omega = cache['weights']
    x_shape = cache['x_shape']
    stride = cache['stride']
    padding = cache['padding']
    kernel_h = cache['kernel_h']
    kernel_w = cache['kernel_w']
    
    N, C_in, H, W = x_shape
    C_out, _, _, _ = omega.shape

    w_row = omega.reshape(C_out, -1)
    d_out_reshaped = d_out.transpose(0, 2, 3, 1).reshape(-1, C_out)
    dx_col = np.dot(d_out_reshaped, w_row)
    dx = col2im(dx_col, x_shape, kernel_h, kernel_w, stride, padding)
    
    return dx

# Step 19 - conv2d_grad_weights
def conv2d_grad_weights(d_out, cache):
    # TODO: return dL/dW shaped (C_out, C_in, kH, kW) from d_out and the im2col cache.
    weights = cache['weights']
    cols = cache['cols']
    C_out, _, _, _ = weights.shape
    
    d_out_reshaped = d_out.transpose(0, 2, 3, 1).reshape(-1, C_out)
    dw_flat = np.dot(d_out_reshaped.T, cols)
    dw = dw_flat.reshape(weights.shape)
    
    return dw

# Step 20 - conv2d_grad_bias
def conv2d_grad_bias(d_out):
    # TODO: return a length C_out gradient by reducing d_out over batch and spatial axes
    return np.sum(d_out, axis=(0, 2, 3))

# Step 21 - conv2d_backward
def conv2d_backward(d_out, cache):
    dx = conv2d_grad_input(d_out, cache)
    dW = conv2d_grad_weights(d_out, cache)
    db = conv2d_grad_bias(d_out)
    
    return dx, dW, db

# Step 22 - maxpool2d_forward
def maxpool2d_forward(images, kernel, stride):
    N, C, H, W = images.shape
    out_h=output_spatial_size(H, kernel, stride, padding=0)
    out_w=output_spatial_size(W, kernel, stride, padding=0)
    out=np.zeros((N,C,out_h,out_w))
    argmax=np.zeros((N,C,out_h,out_w),dtype=int)
    for y in range(out_h):
        for x in range(out_w):
            x_start=stride*x
            x_end=stride*x+kernel
            y_start=stride*y
            y_end=stride*y+kernel
            temp=images[:,:,y_start:y_end,x_start:x_end,].reshape(N,C,-1)
            out[:, :, y, x] = np.max(temp, axis=2)
            argmax[:, :, y, x] = np.argmax(temp, axis=2)
    cache={
        'x_shape': images.shape,
        'argmax': argmax,
        'kernel': kernel,
        'stride': stride
    }
    return out, cache

# Step 23 - scatter_grad_window
def scatter_grad_window(grad_value, argmax_index, kernel):
    out=np.zeros((kernel,kernel))
    y=argmax_index//kernel
    x=argmax_index%kernel
    out[y][x]=grad_value
    return out

# Step 24 - maxpool2d_backward
def maxpool2d_backward(d_out, cache):
    N,C,H,W=cache['x_shape']
    argmax=cache['argmax']
    kernel=cache['kernel']
    stride=cache['stride']
    out_h=output_spatial_size(H, kernel, stride, padding=0)
    out_w=output_spatial_size(W, kernel, stride, padding=0)
    dx = np.zeros((N,C,H,W))
    for n in range(N):
        for c in range(C):
            for y in range(out_h):
                for x in range(out_w):
                    y_start = y * stride
                    y_end = y_start + kernel
                    x_start = x * stride
                    x_end = x_start + kernel
                    grad_val = d_out[n, c, y, x]
                    idx = argmax[n, c, y, x]
                    window_grad = scatter_grad_window(grad_val, idx, kernel)
                    dx[n, c, y_start:y_end, x_start:x_end] += window_grad        
    return dx

# Step 25 - relu_forward
def relu_forward(x):
    cache={
        'x': x.copy()
    }
    mask=x>0
    x[~mask]=0
    return x, cache

# Step 26 - relu_backward
def relu_backward(d_out, cache):
    # TODO: mask the upstream gradient by the positive entries of the cached input.
    x = cache['x']
    dx = np.array(d_out, copy=True)
    dx[x <= 0] = 0
    return dx

# Step 27 - flatten_forward
def flatten_forward(x):
    cache={
        'x_shape': x.shape
    }
    N, _, _, _=x.shape
    return x.reshape(N,-1), cache

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    N,C,H,W=cache['x_shape']
    return d_out.reshape(N,C,H,W)

# Step 29 - linear_forward
def linear_forward(x, weights, bias):
    cache={
        'x': x.copy(),
        'weights': weights.copy()
    }
    return x@weights+bias, cache

# Step 30 - linear_grad_input
import numpy as np

def linear_grad_input(d_out, cache):
    """Gradient of a linear layer w.r.t. its input X."""
    # TODO: return dL/dX given d_out (N, D_out) and cache['weights'] (D_in, D_out)
    X=cache['x']
    W=cache['weights']
    b=cache['bias']
    dX = d_out @ W.T
    
    return dX

# Step 31 - linear_grad_weights
import numpy as np

def linear_grad_weights(x, dout):
    """Gradient of loss wrt linear-layer weights W of shape (D_in, D_out)."""
    # TODO: Compute the gradient of a linear layer's loss wrt its weight matrix W.
    return x.T @ dout

# Step 32 - linear_grad_bias
import numpy as np

def linear_grad_bias(dout):
    return dout.sum(axis=0)

# Step 33 - linear_backward
def linear_backward(dout, cache):
    X=cache['x']
    dx=linear_grad_input(dout, cache)
    dW=linear_grad_weights(X, dout)
    db=linear_grad_bias(dout)
    return (dx,dW,db)

# Step 34 - softmax_cross_entropy_forward
def softmax_cross_entropy_forward(logits, y):
    probs = stable_softmax(logits)
    loss = cross_entropy_loss(probs, y)
    return float(loss)+0.0

# Step 35 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(logits, y):
    N, C =logits.shape
    probs = stable_softmax(logits)
    y_one_hot = one_hot(y, C)
    dlogits = (probs - y_one_hot) / N
    
    return dlogits

# Step 36 - sgd_step
import numpy as np

def sgd_step(param, grad, lr):
    return param-grad*lr

# Step 37 - adam_update_m
import numpy as np

def adam_update_m(m, grad, beta_one):
    return beta_one*m+(1-beta_one)*grad

# Step 38 - adam_update_v
import numpy as np

def adam_update_v(v, grad, beta_two):
    return beta_two*v+(1-beta_two)*grad**2

# Step 39 - adam_bias_correct
def adam_bias_correct(moment, beta, t):
    # TODO: return moment divided by (1 - beta**t) to undo Adam's zero-init bias.
    return moment/(1-beta**t)

# Step 40 - adam_param_step
import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
    # TODO: apply one Adam parameter update using bias-corrected moments
    return param-lr*m_hat/(v_hat**0.5+eps)

# Step 41 - adam_step
import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    m=adam_update_m(m, grad, beta_one)
    v=adam_update_v(v, grad, beta_two)
    m_hat=adam_bias_correct(m, beta_one, t)
    v_hat=adam_bias_correct(v, beta_two, t)
    param=adam_param_step(param, m_hat, v_hat, lr, eps)
    return param, m, v

# Step 42 - init_conv_layer
def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    b=init_zero_bias(out_channels)
    fan_in = in_channels * kernel_size * kernel_size
    W=he_init((out_channels, in_channels, kernel_size, kernel_size),fan_in,seed=seed)
    dicti={
        'W':W,
        'b':b
    }
    return dicti

# Step 43 - init_linear_layer
def init_linear_layer(in_features, out_features, seed=0):
    # TODO: return {'W': He-init matrix (in_features, out_features), 'b': zero bias (out_features,)}
    b=init_zero_bias(out_features)
    fan_in = in_features
    W=he_init((in_features, out_features),fan_in,seed=seed)
    dicti={
        'W':W,
        'b':b
    }
    return dicti

# Step 44 - init_lenet
def init_lenet(in_channels, num_classes, seed=0):
    # TODO: build conv1, conv2, fc1, fc2 with the right shapes and return them in a dict.
    conv1=init_conv_layer(6, in_channels, 5, seed)
    conv2=init_conv_layer(16, 6, 5, seed)
    fc1=init_linear_layer(16*4*4, 120, seed=0)
    fc2=init_linear_layer(120, num_classes, seed=0)
    result={
        'conv1': conv1,
        'conv2': conv2,
        'fc1': fc1,
        'fc2': fc2
    }
    return result

# Step 45 - forward_conv_block
def forward_conv_block(x, W, b, pool_size, stride, pad):
    # TODO: run conv2d -> relu -> maxpool2d and return (out, cache_dict)
    conv_out, conv_cache = conv2d_forward(x, W, b, stride, pad)
    relu_out, relu_cache = relu_forward(conv_out)
    pool_out, pool_cache = maxpool2d_forward(relu_out, pool_size, pool_size)
    cache = {
        'conv_cache': conv_cache,
        'relu_cache': relu_cache,
        'pool_cache': pool_cache
    }
    
    return pool_out, cache

# Step 46 - forward_classifier_block
def forward_classifier_block(x, fc1, fc2):
    # TODO: flatten -> linear1 -> relu -> linear2, return (out, cache_dict)
    flat_out, flatten_cache = flatten_forward(x)
    fc1_out, fc1_cache = linear_forward(flat_out, fc1['W'], fc1['b'])
    relu_out, relu_cache = relu_forward(fc1_out)
    logits, fc2_cache = linear_forward(relu_out, fc2['W'], fc2['b'])
    cache = {
        'flatten_cache': flatten_cache,
        'fc1_cache': fc1_cache,
        'relu_cache': relu_cache,
        'fc2_cache': fc2_cache
    }
    return logits, cache

# Step 47 - lenet_forward
def lenet_forward(x, params):
    # TODO: run two conv blocks then the classifier block and return (logits, caches).
    x,cache1=forward_conv_block(x, params['conv1']['W'], params['conv1']['b'], 2, 1, 0)
    x,cache2=forward_conv_block(x, params['conv2']['W'], params['conv2']['b'], 2, 1, 0)
    x,cache3=forward_classifier_block(x, params['fc1'], params['fc2'])
    caches = {
        'block1': cache1,
        'block2': cache2,
        'classifier': cache3
    }
    return x, caches

# Step 48 - backward_conv_block
def backward_conv_block(dout, cache):
    # TODO: backprop dout through the cached pool, relu, and conv layers in reverse order.
    pool_cache = cache['pool_cache']
    relu_cache = cache['relu_cache']
    conv_cache = cache['conv_cache']
    dout_pool = maxpool2d_backward(dout, pool_cache)
    dout_relu = relu_backward(dout_pool, relu_cache)
    dx, dW, db = conv2d_backward(dout_relu, conv_cache)
    
    return dx, dW, db

# Step 49 - backward_classifier_block
# ── Step 049  backward_classifier_block ──
def backward_classifier_block(dout, cache):
    # TODO: walk fc2 -> relu -> fc1 -> flatten backward, return gradients dict
    fc2_cache = cache['fc2_cache']
    relu_cache = cache['relu_cache']
    fc1_cache = cache['fc1_cache']
    flatten_cache = cache['flatten_cache']
    fc2_cache['bias'] = None
    fc1_cache['bias'] = None
    dout, dW2, db2 = linear_backward(dout, fc2_cache)
    dout = relu_backward(dout, relu_cache)
    dout, dW1, db1 = linear_backward(dout, fc1_cache)
    dx = flatten_backward(dout, flatten_cache)
    
    return {
        'dx': dx,
        'fc1': {
            'dW': dW1,
            'db': db1
        },
        'fc2': {
            'dW': dW2,
            'db': db2
        }
    }

# Step 50 - lenet_backward
def lenet_backward(dlogits, cache):
    # TODO: walk classifier and conv block caches in reverse to assemble all gradients
    classifier=backward_classifier_block(dlogits, cache['classifier'])
    dx2, dW2, db2 = backward_conv_block(classifier['dx'], cache['block2'])
    dx1, dW1, db1 = backward_conv_block(dx2, cache['block1'])
    return {
        'conv1': {
            'dW': dW1,
            'db': db1
        },
        'conv2': {
            'dW': dW2,
            'db': db2
        },
        'fc1': classifier['fc1'],
        'fc2': classifier['fc2']
    }

# Step 51 - lenet_predict
def lenet_predict(x, params):
    # TODO: Return the argmax class index per sample from a LeNet forward pass.
    logits, _ = lenet_forward(x, params)
    return argmax_rows(logits)

# Step 52 - build_synthetic_image_dataset (not yet solved)
# TODO: implement

# Step 53 - shuffle_indices
import numpy as np

def shuffle_indices(n, seed=0):
    # TODO: return a reproducible permutation of [0, n) as an int ndarray of shape (n,).
    x=np.linspace(0, n - 1, num=n, dtype=int)
    rng = np.random.RandomState(seed)
    rng.shuffle(x)
    
    return x

# Step 54 - train_test_split
def train_test_split(x, y, test_fraction=0.2, seed=0):
    # TODO: partition x and y into train and test halves using a shared shuffled order.
    n=len(x)
    index=shuffle_indices(n, seed)
    split_point=int(n*test_fraction)
    test_indices=index[:split_point]
    train_indices=index[split_point:]
    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]
    
    return x_train, y_train, x_test, y_test

# Step 55 - iterate_minibatches
def iterate_minibatches(x, y, batch_size, seed=0):
    # TODO: yield shuffled mini-batches of features and labels for one epoch of training.
    N = len(x)
    indices=shuffle_indices(N, seed=seed)
    for start in range(0, N, batch_size):
        end=start+batch_size
        if end>N:
            break
        batch_indices=indices[start:end]
        yield x[batch_indices], y[batch_indices]

# Step 56 - train_step (not yet solved)
# TODO: implement

# Step 57 - train_one_epoch (not yet solved)
# TODO: implement

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate
def evaluate(params, x, y):
    # TODO: return the fraction of samples whose predicted class equals the label.
    y_hat=lenet_predict(x, params)  # Already returns argmax class indices
    return float(np.mean(y_hat == y))

