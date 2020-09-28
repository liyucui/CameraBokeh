import tensorflow as tf
import Tiramisu as trms
import BatchDataSet as datasetf
import Utils
import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

MAX_ITERATION = int(18e4+1)
IMAGE_SIZE = (144, 192)  # horizontal
# IMAGE_SIZE = (192, 144)  # vertical
data_dir = 'train_144'
# data_dir = '/home/data01_disk/liyucui/'

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "32", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/1021_aspp/", "path to logs directory")
tf.flags.DEFINE_float("learning_rate", '0.0001', "learning rate of adam optimizer")

RESTORE = False

# restore_dir = '0814/model.ckpt-100000.meta'
dropout = 0.8


# def train(loss_val, var_list):
#     optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss_val)
    # grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    #
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #     return optimizer.apply_gradients(grads), grads, update_ops


def main(argv=None):
    image = tf.placeholder(
        tf.float32, shape=[None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3], name='input_image')
    annotation = tf.placeholder(
        tf.float32, shape=[None, IMAGE_SIZE[0], IMAGE_SIZE[1], 1], name='annotation')  # 0/1
    # dropout_num = tf.placeholder(tf.float32, name='dropout_num')

    # build network structure
    classes = trms.Tiramisu2(image)
    classes_1 = trms.Tiramisu2_1(image)
    # classes = trms.build_mobile_unet(image, preset_model="MobileUNet")
    train_loss = Utils.cal_cross_entropy_loss(classes, annotation, useRegularization=True)
    train_loss_1 = Utils.cal_cross_entropy_loss(classes_1, annotation, useRegularization=True)
    mask_f = tf.expand_dims(classes[:, :, :, 1], 3)
    val_iou = Utils.cal_mIoU(mask_f, annotation)
    trainable_var = tf.trainable_variables(scope='Net')
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(train_loss)

    # tensorboard
    with tf.name_scope("tensorboard"):
        tb_bokeh = image*tf.expand_dims(classes[:, :, :, 1], 3)
        tmp_anno = tf.to_float(annotation)
        tmp_d = tmp_anno-tf.expand_dims(classes[:, :, :, 1], 3)

        tmp_TF = tf.cast(tf.abs((-1)*tmp_d)-(-1)*tmp_d, tf.bool)
        tmp_TF = tf.to_float(tmp_TF) * tmp_d
        tmp_FT = tf.to_float(tmp_TF)*(-1)*tmp_d
        tmp_TF = tf.concat([tmp_TF, tmp_TF, tmp_TF], axis=3)
        tmp_FT = tf.concat([tmp_FT, tmp_FT, tmp_FT], axis=3)

        tf_255 = tf.ones(tf.shape(annotation)) * 255
        tf_0 = tf.zeros(tf.shape(annotation))
        tf_red = tf.concat([tf_255, tf_0, tf_0], axis=3)
        tf_blue = tf.concat([tf_0, tf_0, tf_255], axis=3)

        tb_delta = tmp_TF*tf_blue + tmp_FT*tf_red

        tf.summary.image("input_image", image, max_outputs=10)
        tf.summary.image("bokeh", tb_bokeh, max_outputs=10)
        tf.summary.image("delta", tb_delta, max_outputs=2)
        tf.summary.scalar("Loss", train_loss)
        tf.summary.scalar("IoU", val_iou)

        for t in trainable_var:
            tf.summary.histogram("Parameters~"+t.name, t)
        # for g in grads:
        #     tf.summary.histogram("Gradients~"+g[1].name, g[0])
        # for op in ops:
        #     tf.summary.histogram('Update~'+op.name, op)
        summary_op = tf.summary.merge_all()

    data = datasetf.BatchDataSet(data_dir)

    # train
    IoU_max = 0.0
    Loss_min = 999

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_summary_writer = tf.summary.FileWriter(FLAGS.logs_dir+'train', sess.graph)
    val_summary_writer = tf.summary.FileWriter(FLAGS.logs_dir+'val', sess.graph)

    if RESTORE:
        saver0 = tf.train.Saver()
        # saver0 = tf.train.Saver(var_list=tf.trainable_variables(scope='Net'))
        saver0.restore(sess, "logs/1019_aspp/model.ckpt-16000")

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=40)

    for itr in range(0, MAX_ITERATION):
        train_images, train_annotations = data.train_next_batch(FLAGS.batch_size)
        feed_dict = {image: train_images, annotation: train_annotations}
        sess.run(train_op, feed_dict=feed_dict)
        # 前向传播

        if itr % 50 == 0:
            train_batch_loss, summary_str = sess.run([train_loss, summary_op], feed_dict=feed_dict)

            if itr % 200 == 0:
                print("Step: %d, Train_loss:%g" % (itr, train_batch_loss), end='')

                val_images, val_annotations = data.val_random_batch(64)
                feed_dict_dev = {image: val_images, annotation: val_annotations}
                val_batch_loss, dev_summary_str, val_batch_iou = sess.run([train_loss_1, summary_op, val_iou], feed_dict=feed_dict_dev)
                val_summary_writer.add_summary(dev_summary_str, itr)

                # val_images = data.val_image_batch(10)
                # feed_dict_dev = {image: val_images, dropout_num: 1.0}
                # val_images1, dev_summary_str = sess.run([classes, summary_op], feed_dict=feed_dict_dev)
                # val_summary_writer.add_summary(dev_summary_str, itr)

                # feed_dict_bokeh = {image: val_images, dropout_num:1.0}
                # bokeh_image = sess.run(, feed_dict=feed_dict_bokeh)

                if val_batch_iou > IoU_max:
                    IoU_max = val_batch_iou
                    saver.save(sess, FLAGS.logs_dir + "topAccModel.ckpt", itr)
                else:
                    if val_batch_loss < Loss_min:
                        Loss_min = val_batch_loss
                        saver.save(sess, FLAGS.logs_dir + "minLossModel.ckpt", itr)
                print(" Val_loss:%g, IoU:%g" % (val_batch_loss, val_batch_iou))
            else:
                print("Step: %d, Train_loss:%g" % (itr, train_batch_loss))
            val_train_summary_writer.add_summary(summary_str, itr)

        if itr % 2000 == 0:
            saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
            FLAGS.learning_rate /= 2.0
            if FLAGS.learning_rate < 0.0000001:
                FLAGS.learning_rate = 0.0000001
            print(FLAGS.learning_rate)

    sess.close()
    print('end')


if __name__ == "__main__":
    tf.app.run()
