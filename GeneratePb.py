import tensorflow as tf
# import back_camera_train as train
import Tiramisu as model
from tensorflow.python.framework.graph_util import convert_variables_to_constants
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# IMAGE_SIZE = (96, 128)  # front horizontal
# IMAGE_SIZE = (128, 96)  # front vertical
# IMAGE_SIZE = (144, 192)  # back horizontal
# IMAGE_SIZE = (128, 96)  # back vertical
IMAGE_SIZE = (128, 128)  # back vertical
restore_dir = "logs/0814/model.ckpt-108000"

def main(argv=None):
    image = tf.placeholder(
        tf.float32, shape=[1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3], name='input_img')

    classes = model.Tiramisu2_no_dropout(image)
    mask = tf.expand_dims(classes[..., 1], axis=3, name='mask')

    sess = tf.Session()

    saver = tf.train.Saver(var_list=tf.global_variables())
    saver.restore(sess, restore_dir)
    graph = convert_variables_to_constants(sess, sess.graph_def, ["mask"])

    tf.train.write_graph(graph, '0826', 'origin1.pb', as_text=False)

    sess.close()
    print('end')


if __name__ == "__main__":
    tf.app.run()
