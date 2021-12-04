import tensorflow as tf


DT = 0.5  # detection/classification threshold

a_pos = 1.0  # Explanation mask composition factors
a_neg = 1.0
a_bg = 1.0


def normalize(x, axis=(-2, -1), reduce_min=True, reduce_max=True):
  if reduce_min: x -= tf.reduce_min(x, axis=axis, keepdims=True)
  if reduce_max:
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    x = tf.math.divide_no_nan(x, x_max)

  return x


@tf.function
def minmaxcam(
    model,
    x,
    sW
):
  """
  model: tf.keras.Model: the CNN classification model that outputs the logits vector and A_ij
  x: tf.Tensor[batch, height, width, 3]: the input, preprocessed data tensor
  sW: tf.Tensor[kernels, labels]: the weights of the classifying layer,
                                  usually be obtained from `model.layers[-1].weights[0]`
  """
  l, a = model(x, training=False)
  p = tf.nn.sigmoid(l)                            # predictions              (batch, labels)
  d = tf.cast(p > DT, tf.float32)                 # detection mask           (batch, labels)
  c = tf.reduce_sum(d, axis=-1)                   # count of detected labels (batch, 1)
  c = tf.reshape(c, (-1, 1, 1))

  w = d[:, tf.newaxis, :] * sW[tf.newaxis, ...]   # weights from detected labels (batch, kernels, labels)
  w_n = tf.reduce_sum(w, axis=-1, keepdims=True)  # added contributions          (batch, kernels)
  w_n = w_n - w                                   # minimizing contributions     (batch, kernels, labels)

  w = a_pos*sW - a_neg*w_n / tf.maximum(c-1, 1)   # kernel contributions         (batch, kernels, labels)

  maps = tf.einsum('bhwk,bku->buhw', a, w)        # explaining maps              (batch, labels, height, width)
  maps = tf.nn.relu(maps)
  maps = normalize(maps)

  return l, maps


@tf.function
def d_minmaxcam(
    model,
    x,
    sW
):
  l, a = model(x, training=False)
  p = tf.nn.sigmoid(l)

  d = tf.cast(p > DT, tf.float32)
  c = tf.reshape(tf.reduce_sum(d, axis=-1), (-1, 1, 1))

  w = d[:, tf.newaxis, :] * sW[tf.newaxis, ...]
  wa = tf.reduce_sum(w, axis=-1, keepdims=True)
  wn = wa - w

  w = (  a_pos * tf.nn.relu(sW)
       - a_neg * tf.nn.relu(wn) / tf.maximum(c-1, 1)
       + a_bg  * tf.minimum(0., wa) / tf.maximum(c, 1))

  maps = tf.einsum('bhwk,bku->buhw', a, w)
  maps = tf.nn.relu(maps)
  maps = normalize(maps)

  return l, maps


def minmax_j(s, p):
  d = tf.cast(p > DT, tf.float32)                  # detection mask           (batch, labels)
  c = tf.reduce_sum(d, axis=-1, keepdims=True)     # count of detected labels (batch, 1)

  sd = s*d                                         # masked detected scores   (batch, labels)
  s_n = tf.reduce_sum(sd, axis=-1, keepdims=True)  # added scores             (batch, 1)
  s_n = s_n - sd                                   # minimizing scores        (batch, labels)

  return a_pos*s - a_neg*s_n / tf.maximum(c-1, 1)

@tf.function
def minmax_gradcam(x, model):
  with tf.GradientTape(watch_accessed_variables=False) as t:
    t.watch(x)
    l, a = model(x, training=False)
    p = tf.nn.sigmoid(l)
    loss = minmax_j(l, p)

  dlda = t.batch_jacobian(loss, a)                 # partials contributions   (batch, labels, width, height, kernels)

  weights = tf.reduce_sum(dlda, axis=(-3, -2))     # kernel contributions     (batch, labels, kernels)
  maps = tf.einsum('bhwc,buc->buhw', a, weights)   # explaining maps          (batch, labels, width, height)
  maps = tf.nn.relu(maps)
  maps = normalize(maps)

  return l, maps


def d_minmax_j(s):
  p = tf.nn.sigmoid(s)
  d = tf.cast(p > DT, tf.float32)
  c = tf.reduce_sum(d, axis=-1, keepdims=True)

  sd = s*d                                        # only detections
  sa = tf.reduce_sum(sd, axis=-1, keepdims=True)  # sum logits detected (b, 1)
  sn = sa - sd

  return tf.stack((
    a_pos * s,
    a_neg * sn / tf.maximum(c-1, 1),
    a_bg  * (sn+sd) / tf.maximum(c, 1)
  ), axis=1)

@tf.function
def d_minmax_cam(x, model):
  with tf.GradientTape(watch_accessed_variables=False) as t:
    t.watch(x)
    
    l, a = model(x, training=False)
    loss = d_minmax_j(l)

  dlda = t.batch_jacobian(loss, a)

  w, wn, wa = dlda[:, 0], dlda[:, 1], dlda[:, 2]
  
  w = (  tf.nn.relu(w)
       - tf.nn.relu(wn)
       + tf.minimum(0., wa))

  weights = tf.reduce_sum(w, axis=(-3, -2))
  maps = tf.einsum('bhwc,buc->buhw', a, weights)
  maps = tf.nn.relu(maps)
  maps = normalize(maps)

  return l, maps
