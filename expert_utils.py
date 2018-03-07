import tensorflow as tf


if ffn_layer == "dense_relu_dense":
    # In simple convolution mode, use `pad_remover` to speed up processing.
    if pad_remover:
        original_shape = common_layers.shape_list(x)
        # Collapse `x` across examples, and remove padding positions.
        x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))
        x = tf.expand_dims(pad_remover.remove(x), axis=0)
    conv_output = common_layers.dense_relu_dense(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        dropout=hparams.relu_dropout,
        dropout_broadcast_dims=relu_dropout_broadcast_dims)
    if pad_remover:
        # Restore `conv_output` to the original shape of `x`, including padding.
        conv_output = tf.reshape(
            pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)


class PadRemover(object):
  """Helper to remove padding from a tensor before sending to the experts.

  The padding is computed for one reference tensor containing the padding mask
  and then can be applied to any other tensor of shape [dim_origin,...].

  Ex:
      input = [
        [tok1, tok2],
        [tok3, tok4],
        [0, 0],
        [0, 0],
        [tok5, tok6],
        [0, 0],
      ]
      output = [
        [tok1, tok2],
        [tok3, tok4],
        [tok5, tok6],
      ]
  """

  def __init__(self, pad_mask):
    """Compute and store the location of the padding.

    Args:
      pad_mask (tf.Tensor): Reference padding tensor of shape
        [batch_size,length] or [dim_origin] (dim_origin=batch_size*length)
        containing non-zeros positive values to indicate padding location.
    """
    self.nonpad_ids = None
    self.dim_origin = None

    with tf.name_scope("pad_reduce/get_ids"):
      pad_mask = tf.reshape(pad_mask, [-1])  # Flatten the batch
      # nonpad_ids contains coordinates of zeros rows (as pad_mask is
      # float32, checking zero equality is done with |x| < epsilon, with
      # epsilon=1e-9 as standard, here pad_mask only contains positive values
      # so tf.abs would be redundant)
      self.nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))
      self.dim_origin = tf.shape(pad_mask)[:1]

  def remove(self, x):
    """Remove padding from the given tensor.

    Args:
      x (tf.Tensor): of shape [dim_origin,...]

    Returns:
      a tensor of shape [dim_compressed,...] with dim_compressed <= dim_origin
    """
    with tf.name_scope("pad_reduce/remove"):
      x_shape = x.get_shape().as_list()
      x = tf.gather_nd(
          x,
          indices=self.nonpad_ids,
      )
      if not context.in_eager_mode():
        # This is a hack but for some reason, gather_nd return a tensor of
        # undefined shape, so the shape is set up manually
        x.set_shape([None] + x_shape[1:])
    return x

  def restore(self, x):
    """Add padding back to the given tensor.

    Args:
      x (tf.Tensor): of shape [dim_compressed,...]

    Returns:
      a tensor of shape [dim_origin,...] with dim_compressed >= dim_origin. The
      dim is restored from the original reference tensor
    """
    with tf.name_scope("pad_reduce/restore"):
      x = tf.scatter_nd(
          indices=self.nonpad_ids,
          updates=x,
          shape=tf.concat([self.dim_origin, tf.shape(x)[1:]], axis=0),
      )
    return x