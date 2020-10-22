def zero_pad(X, pad):
    '''
    :param X: (m, n_H, n_W, n_C)
    :param pad: integer
    :return: X_pad (m, n_H + 2*pad, n_W + 2*pad, n_C)
    '''
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    '''
    :param a_slice_prev: slice of input data (f, f, n_C_prev)
    :param W: (f, f, n_C_prev)
    :param b: (1, 1, 1)
    :return: scalar value
    '''
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)
    Z += float(b)
    return Z

def conv_forward(A_prev, W, b, hparameters):
    '''
    :param A_prev: (m, n_H_prev, n_W_prev, n_C_prev)
    :param W: (f, f, n_C_prev, n_C)
    :param b: (1, 1, 1, n_C)
    :param hparameters: stride / pad dict
    :return: Z (m, n_H, n_W, n_C) / cache: for back-prop
    '''
    (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)
    (f, f, n_C_prev, n_C) = np.shape(W)
    stride, pad = hparameters['stride'], hparameters['pad']

    n_H = int((n_H_prev + 2 * pad - f) / stride) + 1
    n_W = int((n_W_prev + 2 * pad - f) / stride) + 1

    # Initialize output volume
    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i, :, :, :]
        for h in range(n_H):
            vert_start = h * stride
            vert_end = vert_start + f

            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = horiz_start + f

                for c in range(n_C):
                    a_slice_prev = a_prev_pad[vert_start: vert_end, horiz_start: horiz_end, :]

                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)

    assert(Z.shape == (m, n_H, n_W, n_C))

    cache = (A_prev, W, b, hparameters)
    return Z, cache

def pool_forward(A_prev, hparameters, mode="max"):
    '''
    :param A_prev: (m, n_H_prev, n_W_prev, n_C_prev)
    :param hparameters: f / stride dict
    :param mode: "max" or "average"
    :return: A (m, n_H, n_W, n_C) cache()
    '''

    (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)
    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            vert_start = stride * h
            vert_end = vert_start + f

            for w in range(n_W):
                horiz_start = stride * w
                horiz_end = horiz_start + f

                for c in range(n_C):
                    a_prev_slice = A_prev[i, vert_start: vert_end, horiz_start: horiz_end, :]
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)

    cache = (A_prev, hparameters)
    assert(A.shape == (m, n_H, n_W, n_C))
    return A, cache

def conv_backward(dZ, cache):
    '''
    :param dZ:
    :param cache:
    :return: dA_prev, dW, db
    '''

    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)
    (f, f, n_C_prev, n_C) = np.shape(W)

    stride = hparameters["stride"]
    pad = hparameters["pad"]

    (m, n_H, n_W, n_C) = np.shape(dZ)

    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dw = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    A_prev_pad = zero_pad(A, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i, :, :, :]
        da_prev_pad = dA_prev_pad[i, :, :, :]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = h * stride
                    horiz_end = horiz_start + f

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    dW[:, :, :, c] += np.multiply(a_slice, dZ[i, h, w, c])
                    db[:, :, :, c] += dZ[i, h, w, c]

        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        # 이 부분 아직 제대로 이해 못함 컨볼루션 백프롭은 너무 어려워

    return dA_prev, dW, db