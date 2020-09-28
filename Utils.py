import tensorflow as tf


def cal_binary_cross_entropy_loss(score, annotation, useRegularization=False):

    if useRegularization:
        l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))


def cal_cross_entropy_loss(score, annotation, useRegularization=False):
    # score: [H,W,2],[fg_score,bg_score]; annotation: binary ground truth, [H,W,1]
    # cross entropy loss

    annotation = tf.image.resize_bilinear(annotation, score.get_shape()[1:3], True)
    annotation = tf.concat([1.0 - annotation, annotation], axis=3)
    loss = -annotation*tf.log(tf.clip_by_value(score, 1e-8, 1.0))
    loss = tf.reduce_mean(loss)
    if useRegularization:
        l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = loss + l2_loss
    return loss


def cal_focal_loss(score, annotation, useAlpha, useRegularization=False):
    # score: [H,W,2],[fg_score,bg_score]; annotation: binary ground truth, [H,W,1]
    # Focal Loss https://arxiv.org/pdf/1708.02002.pdf

    # prob = tf.nn.softmax(score) # Here are the reason that causes NaN values
    prob = tf.clip_by_value(score, 1e-10, 1.0)
    gamma = 2.0
    annotation = tf.cast(annotation, dtype=tf.float32)
    fl = tf.pow(1 - prob, gamma) * tf.log(prob)

    if useAlpha:
        alpha = 0.25
        loss = alpha * fl[:, :, :, 1] * annotation[:, :, :, 0] + \
               (1 - alpha) * fl[:, :, :, 0] * (1 - annotation[:, :, :, 0])
    else:
        loss = fl[:, :, :, 1] * annotation[:, :, :, 0] + \
               fl[:, :, :, 0] * (1 - annotation[:, :, :, 0])

    loss = tf.reduce_mean(-loss, name="focal_loss")

    if useRegularization:
        l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = loss + l2_loss
    return loss


def cal_mIoU(alpha,labels):
    labels = tf.to_float(labels)
    a = tf.round(alpha)

    U = tf.reduce_sum(tf.round((labels+a+0.5)/2))
    I = tf.reduce_sum(tf.round((labels+a)/2))
    return I/U