import tensorflow as tf
import tensorflow.contrib.slim as slim
import resnet_v1 as bottleneck


def separable_conv(input_tensor, depth_out, scope, stride=1, rate=1):
    with tf.variable_scope(scope):
        depth_in = input_tensor.get_shape()[-1]

        if depth_in > depth_out:
            depth_expand = depth_in
        else:
            depth_expand = depth_out

        conv = slim.conv2d(input_tensor, depth_expand, 1, activation_fn=tf.nn.relu6,
                           normalizer_fn=slim.batch_norm, scope='conv_expand')
        conv = slim.separable_conv2d(conv, None, 3, 1, rate=rate, stride=stride,
                                     activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm, scope='convs')
        conv = slim.conv2d(conv, depth_out, 1, activation_fn=None,
                           normalizer_fn=slim.batch_norm, scope = 'conv_shrink')

        return conv


def dense_block_3(input, outnum, scope, dropout_num):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            activation_fn=tf.nn.relu6,
                            weights_initializer=slim.variance_scaling_initializer()):
            with slim.arg_scope([slim.batch_norm], scale=True, is_training=True):
                d1 = slim.conv2d(input, outnum, 3, rate=2, scope='d1')
                d1 = tf.nn.dropout(d1, dropout_num)
                c1 = tf.concat([input, d1], axis=3)

                d2 = slim.conv2d(c1, outnum, 3, rate=3, scope='d2')
                d2 = tf.nn.dropout(d2, dropout_num)
                c2 = tf.concat([c1, d2], axis=3)

                d3 = slim.conv2d(c2, outnum, 3, rate=5, scope='d3')
                d3 = tf.nn.dropout(d3, dropout_num)
                c3 = tf.concat([d1, d2, d3], axis=3)
                c3 = slim.conv2d(c3, outnum, 1, scope='c3')
    return c3 # output channels is outnum
def dense_block_3_aspp(input, outnum, scope, dropout_num):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            activation_fn=tf.nn.relu6,
                            weights_initializer=slim.variance_scaling_initializer()):
            with slim.arg_scope([slim.batch_norm], scale=True, is_training=True):
                d0 = slim.conv2d(input, outnum, 1, scope='d0')
                d0 = tf.nn.dropout(d0, dropout_num)

                d1 = slim.conv2d(input, outnum, 3, rate=2, scope='d1')
                d1 = tf.nn.dropout(d1, dropout_num)
                # c1 = tf.concat([d0, d1], axis=3)

                d2 = slim.conv2d(input, outnum, 3, rate=3, scope='d2')
                d2 = tf.nn.dropout(d2, dropout_num)
                # c2 = tf.concat([c1, d2], axis=3)

                d3 = slim.conv2d(input, outnum, 3, rate=5, scope='d3')
                d3 = tf.nn.dropout(d3, dropout_num)
                c3 = tf.concat([d0, d1, d2, d3], axis=3)
                c3 = slim.conv2d(c3, outnum, 1, scope='c3')
    return c3 # output channels is outnum

def dense_block_3_no_dropout(input, outnum, scope):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            activation_fn=tf.nn.relu6,
                            weights_initializer=slim.variance_scaling_initializer()):
            with slim.arg_scope([slim.batch_norm], scale=True, is_training=True):
                d1 = slim.conv2d(input, outnum, 3, rate=2, scope='d1')
                # d1 = tf.nn.dropout(d1, dropout_num)
                c1 = tf.concat([input, d1], axis=3)

                d2 = slim.conv2d(c1, outnum, 3, rate=3, scope='d2')
                # d2 = tf.nn.dropout(d2, dropout_num)
                c2 = tf.concat([c1, d2], axis=3)

                d3 = slim.conv2d(c2, outnum, 3, rate=5, scope='d3')
                # d3 = tf.nn.dropout(d3, dropout_num)
                c3 = tf.concat([d1, d2, d3], axis=3)
                c3 = slim.conv2d(c3, outnum, 1, scope='c3')
    return c3 # output channels is outnum


def dense_block_3_separable(input, outnum, scope, dropout_num):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            activation_fn=tf.nn.relu6,
                            weights_initializer=slim.variance_scaling_initializer()):
            with slim.arg_scope([slim.batch_norm], scale=True, is_training=True):
                d1 = separable_conv(input, outnum, 'd1', rate=2)
                d1 = tf.nn.dropout(d1, dropout_num)
                c1 = tf.concat([input, d1], axis=3)

                d2 = separable_conv(c1, outnum, 'd2', rate=3)
                d2 = tf.nn.dropout(d2, dropout_num)
                c2 = tf.concat([c1, d2], axis=3)

                d3 = separable_conv(c2, outnum, 'd3', rate=5)
                d3 = tf.nn.dropout(d3, dropout_num)
                c3 = tf.concat([d1, d2, d3], axis=3)

                out = slim.conv2d(c3, outnum, 1, scope='out')
                return out


def dense_block_4(input, outnum, scope, dropout_num):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            activation_fn=tf.nn.relu6,
                            weights_initializer=slim.variance_scaling_initializer()):
            with slim.arg_scope([slim.batch_norm], scale=True, is_training=True):
                d0 = slim.conv2d(input, outnum, 3, scope='d0')
                d0 = tf.nn.dropout(d0, dropout_num)
                c0 = tf.concat([input, d0], axis=3)

                d1 = slim.conv2d(c0, outnum, 3, rate=2, scope='d1')
                d1 = tf.nn.dropout(d1, dropout_num)
                c1 = tf.concat([c0, d1], axis=3)

                d2 = slim.conv2d(c1, outnum, 3, rate=3, scope='d2')
                d2 = tf.nn.dropout(d2, dropout_num)
                c2 = tf.concat([c1, d2], axis=3)

                d3 = slim.conv2d(c2, outnum, 3, rate=5, scope='d3')
                d3 = tf.nn.dropout(d3, dropout_num)
                c3 = tf.concat([d0, d1, d2, d3], axis=3)

                out = slim.conv2d(c3, outnum, 1, scope='out')
                return out


def dense_block_5_regular_conv(input_tensor, outnum, scope, dropout_num):
    with tf.variable_scope(scope):
        d0 = slim.conv2d(input_tensor, outnum, 3, scope='d0')
        d0 = tf.nn.dropout(d0, dropout_num)
        c0 = tf.concat([input_tensor, d0], axis=3)

        d1 = slim.conv2d(c0, outnum, 3, scope='d1')
        d1 = tf.nn.dropout(d1, dropout_num)
        c1 = tf.concat([c0, d1], axis=3)

        d2 = slim.conv2d(c1, outnum, 3, scope='d2')
        d2 = tf.nn.dropout(d2, dropout_num)
        c2 = tf.concat([c1, d2], axis=3)

        d3 = slim.conv2d(c2, outnum, 3, scope='d3')
        d3 = tf.nn.dropout(d3, dropout_num)
        c3 = tf.concat([c2, d3], axis=3)

        d4 = slim.conv2d(c3, outnum, 3, scope='d4')
        d4 = tf.nn.dropout(d4, dropout_num)
        c4 = tf.concat([d0, d1, d2, d3, d4], axis=3)

        out = slim.conv2d(c4, outnum, 1, scope='out')
        return out


def dense_block_4_separable(input_tensor, out_num, scope, dropout_num):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            activation_fn=tf.nn.relu6,
                            weights_initializer=slim.variance_scaling_initializer()):
            with slim.arg_scope([slim.batch_norm], scale=True, is_training=True):
                d0 = separable_conv(input_tensor, out_num, 'd0')
                d0 = tf.nn.dropout(d0, dropout_num)
                c0 = tf.concat([input_tensor, d0], axis=3)

                d1 = separable_conv(c0, out_num, 'd1', rate=2)
                d1 = tf.nn.dropout(d1, dropout_num)
                c1 = tf.concat([c0, d1], axis=3)

                d2 = separable_conv(c1, out_num, 'd2', rate=3)
                d2 = tf.nn.dropout(d2, dropout_num)
                c2 = tf.concat([c1, d2], axis=3)

                d3 = separable_conv(c2, out_num, 'd3', rate=5)
                d3 = tf.nn.dropout(d3, dropout_num)
                c3 = tf.concat([d0, d1, d2, d3], axis=3)

                out = slim.conv2d(c3, out_num, 1, scope='out')
                return out


def shuffle_v1(input_tensor, num_outputs, scope, kernel_size=3, stride=1, num_groups=4, use_batch_norm=True, activation=tf.nn.relu6):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=None,
                            activation_fn=None):
            sz = input_tensor.get_shape()[-1].value // num_groups
            conv_side_layers = [slim.conv2d(input_tensor[..., i * sz: i * (sz + 1)],
                                            num_outputs // num_groups,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            scope='conv' + str(i))
                                for i in range(num_groups)]
            conv_g = tf.concat(conv_side_layers, axis=-1)
            if use_batch_norm:
                conv_g = slim.batch_norm(conv_g, scale=True, is_training=True)
            conv_g = activation(conv_g, name='activation')

            n, h, w, c = conv_g.shape.as_list()
            conv_g_reshaped = tf.reshape(conv_g, [-1, h, w, num_groups, c // num_groups])
            conv_g_transposed = tf.transpose(conv_g_reshaped, [0, 1, 2, 4, 3])
            output = tf.reshape(conv_g_transposed, [-1, h, w, c])

            if stride > 1:
                max_pool = slim.max_pool2d(input_tensor, stride, stride, padding='SAME')
                output = tf.concat([output, max_pool], axis=-1)

            return output


def shuffle_v2(input_tensor, scope, kernel_size=3, rate=1, stride=1):
    with tf.variable_scope(scope):
        input_depth = input_tensor.get_shape()[-1]
        if stride <= 1:
            num_branch_outputs = input_depth // 2
            x = input_tensor[..., :num_branch_outputs]
            y = input_tensor[..., num_branch_outputs:]
            x = slim.conv2d(x, num_branch_outputs, kernel_size=kernel_size, rate=rate)
        else:
            num_branch_outputs = input_depth
            x = slim.conv2d(input_tensor, num_branch_outputs, kernel_size=kernel_size, stride=stride)
            # y = slim.conv2d(input_tensor, num_branch_outputs, kernel_size=kernel_size, rate=2)  # tested not useful
            y = slim.max_pool2d(input_tensor, stride, stride, padding='SAME')

        n, h, w, c = x.get_shape()
        z = tf.stack([x, y], axis=3)
        z = tf.transpose(z, [0, 1, 2, 4, 3])
        z = tf.reshape(z, [-1, h, w, num_branch_outputs * 2])

        return z  # z is the same size as input(stride=1) or twice the depth(stride>=2)


def dense_block_3_shuffle_v2(input_tensor, scope, dropout_num):
    with tf.variable_scope(scope):
        d1 = shuffle_v2(input_tensor=input_tensor, scope='d1', rate=2)
        d2 = shuffle_v2(input_tensor=d1, scope='d2', rate=3)
        d3 = shuffle_v2(input_tensor=d2, scope='d3', rate=5)
    return tf.nn.dropout(d3, dropout_num)  # output size is the same as input


def dense_block_4_shuffle_v2(input_tensor, scope, dropout_num):
    with tf.variable_scope(scope):
        d1 = shuffle_v2(input_tensor=input_tensor, scope='d1', rate=2)
        d2 = shuffle_v2(input_tensor=d1, scope='d2', rate=3)
        d3 = shuffle_v2(input_tensor=d2, scope='d3', rate=5)
        d4 = shuffle_v2(input_tensor=d3, scope='d4', rate=9)
    return tf.nn.dropout(d4, dropout_num)  # output size is the same as input


# 使用降采样卷积/空洞卷积/densenet结构 非常有效
# 减薄 + 降采样卷积
def Tiramisu2(input_image, dropout_num):
    input_image /= 255.0
    with tf.variable_scope('Tiramisu'):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            normalizer_fn=slim.batch_norm,
                            activation_fn=tf.nn.relu6,
                            weights_regularizer=slim.l2_regularizer((0.0005)),
                            weights_initializer=slim.variance_scaling_initializer()):
            with slim.arg_scope([slim.batch_norm], scale=True, is_training=True):
                conv1 = slim.conv2d(input_image, 16, 3)
                conv1 = tf.nn.dropout(conv1, dropout_num)
                conv1 = slim.conv2d(conv1, 16, 3)

                conv2 = slim.conv2d(conv1, 32, 3, stride=2)
                conv2 = tf.nn.dropout(conv2, dropout_num)  # 96*72
                conv2 = slim.conv2d(conv2, 32, 3)

                conv3 = slim.conv2d(conv2, 48, 3, stride=2)  # 48*36
                conv3 = dense_block_3_aspp(conv3, 48, 'conv3', dropout_num)

                conv4 = slim.conv2d(conv3, 64, 3, stride=2)  # 24*18
                conv4 = dense_block_3_aspp(conv4, 64, 'conv4', dropout_num)

                conv5 = slim.conv2d(conv4, 128, 3, stride=2)  # 12*9
                conv5 = tf.nn.dropout(conv5, dropout_num)
                conv5 = slim.conv2d(conv5, 128, 3)

                up1 = tf.image.resize_bilinear(conv5, (conv4.get_shape()[1], conv4.get_shape()[2]))  # 24*18
                up1 = tf.concat([conv4, up1], axis=3)
                up1 = dense_block_3_aspp(up1, 64, 'up1', dropout_num)

                up2 = tf.image.resize_bilinear(up1, (conv3.get_shape()[1], conv3.get_shape()[2]))  # 48*36
                up2 = tf.concat([conv3, up2], axis=3)
                up2 = dense_block_3_aspp(up2, 48, 'up2', dropout_num)

                up3 = tf.image.resize_bilinear(up2, (conv2.get_shape()[1], conv2.get_shape()[2]))
                up3 = tf.concat([conv2, up3], axis=3)
                up3 = slim.conv2d(up3, 32, 3)
                up3 = tf.nn.dropout(up3, dropout_num)
                up3 = slim.conv2d(up3, 32, 3)

                up4 = tf.image.resize_bilinear(up3, (conv1.get_shape()[1], conv1.get_shape()[2]))
                up4 = tf.concat([conv1, up4], axis=3)
                up4 = slim.conv2d(up4, 16, 3)
                up4 = tf.nn.dropout(up4, dropout_num)
                up4 = slim.conv2d(up4, 16, 3)
                classes = slim.conv2d(up4, 2, 1, activation_fn=tf.nn.softmax, normalizer_fn=None, scope='classes')
    return classes
def Tiramisu2_with_ASPP(input_image, dropout_num, aspp=True, reuse=None):
    input_image /= 255.0
    depth = 50
    if aspp:
        multi_grid = (1, 2, 4)
    else:
        multi_grid = (1, 2, 1)
    scope = 'resnet_v1_{}'.format(depth)
    with tf.variable_scope(scope, [input_image], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
    with tf.variable_scope('Tiramisu'):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            normalizer_fn=slim.batch_norm,
                            activation_fn=tf.nn.relu6,
                            weights_regularizer=slim.l2_regularizer((0.0005)),
                            weights_initializer=slim.variance_scaling_initializer()):
            with slim.arg_scope([slim.batch_norm], scale=True, is_training=True):
                conv1 = slim.conv2d(input_image, 16, 3)
                conv1 = tf.nn.dropout(conv1, dropout_num)
                conv1 = slim.conv2d(conv1, 16, 3)

                conv2 = slim.conv2d(conv1, 32, 3, stride=2)
                conv2 = tf.nn.dropout(conv2, dropout_num)  # 96*72
                conv2 = slim.conv2d(conv2, 32, 3)

                conv3 = slim.conv2d(conv2, 48, 3, stride=2)  # 48*36
                conv3 = dense_block_3(conv3, 48, 'conv3', dropout_num)

                conv4 = slim.conv2d(conv3, 64, 3, stride=2)  # 24*18
                conv4 = dense_block_3(conv4, 64, 'conv4', dropout_num)

                conv5 = slim.conv2d(conv4, 128, 3, stride=2)  # 12*9
                conv5 = tf.nn.dropout(conv5, dropout_num)
                conv5 = slim.conv2d(conv5, 128, 3)

                up1 = tf.image.resize_bilinear(conv5, (conv4.get_shape()[1], conv4.get_shape()[2]))  # 24*18
                up1 = tf.concat([conv4, up1], axis=3)
                up1 = dense_block_3(up1, 64, 'up1', dropout_num)

                up2 = tf.image.resize_bilinear(up1, (conv3.get_shape()[1], conv3.get_shape()[2]))  # 48*36
                up2 = tf.concat([conv3, up2], axis=3)
                up2 = dense_block_3(up2, 48, 'up2', dropout_num)

                up3 = tf.image.resize_bilinear(up2, (conv2.get_shape()[1], conv2.get_shape()[2]))
                up3 = tf.concat([conv2, up3], axis=3)
                up3 = slim.conv2d(up3, 32, 3)
                up3 = tf.nn.dropout(up3, dropout_num)
                up3 = slim.conv2d(up3, 32, 3)

                up4 = tf.image.resize_bilinear(up3, (conv1.get_shape()[1], conv1.get_shape()[2]))
                up4 = tf.concat([conv1, up4], axis=3)
                up4 = slim.conv2d(up4, 16, 3)
                up4 = tf.nn.dropout(up4, dropout_num)
                up4 = slim.conv2d(up4, 16, 3)
                net = up4
                if aspp:
                    with tf.variable_scope('aspp', [net]) as sc:
                        aspp_list = []
                        branch_1 = slim.conv2d(net, 256, [1, 1], stride=1,
                                               scope='1x1conv')
                        branch_1 = slim.utils.collect_named_outputs(
                            end_points_collection, sc.name, branch_1)
                        aspp_list.append(branch_1)

                        for i in range(3):
                            branch_2 = slim.conv2d(net, 256, [3, 3], stride=1, rate=6*(i+1),
                                                   scope='rate{}'.format(6*(i+1)))
                            branch_2 = slim.utils.collect_named_outputs(end_points_collection, sc.name, branch_2)
                            aspp_list.append(branch_2)



                        # aspp = tf.add_n(aspp_list)
                        # aspp = slim.utils.collect_named_outputs(end_points_collection, sc.name, aspp)

                    # with tf.variable_scope('img_pool', [net]) as sc:
                    #     """Image Pooling
                    #     See ParseNet: Looking Wider to See Better
                    #     """
                    #     pooled = tf.reduce_mean(net, [1, 2], name='avg_pool',
                    #                             keep_dims=True)
                    #     pooled = slim.utils.collect_named_outputs(end_points_collection,
                    #                                               sc.name, pooled)
                    #
                    #     pooled = slim.conv2d(pooled, 256, [1, 1], stride=1, scope='1x1conv')
                    #     pooled = slim.utils.collect_named_outputs(end_points_collection,
                    #                                               sc.name, pooled)
                    #
                    #     pooled = tf.image.resize_bilinear(pooled, tf.shape(net)[1:3])
                    #     pooled = slim.utils.collect_named_outputs(end_points_collection,
                    #                                               sc.name, pooled)
                    #
                    # with tf.variable_scope('fusion', [aspp_list, pooled]) as sc:
                    #     aspp_list.append(pooled)
                    #     net = tf.concat(aspp_list, 3)
                    #     net = slim.utils.collect_named_outputs(end_points_collection,
                    #                                            sc.name, net)
                    #
                    #     net = slim.conv2d(net, 256, [1, 1], stride=1, scope='1x1conv')
                    #     net = slim.utils.collect_named_outputs(end_points_collection,
                    #                                            sc.name, net)
                else:
                    with tf.variable_scope('block5', [net]) as sc:
                        base_depth = 512

                        for i in range(3):
                            with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                                net = bottleneck(net, depth=base_depth * 4,
                                                 depth_bottleneck=base_depth, stride=1, rate=4 * multi_grid[i])
                        net = slim.utils.collect_named_outputs(end_points_collection,
                                                               sc.name, net)

                    with tf.variable_scope('block6', [net]) as sc:
                        base_depth = 512

                        for i in range(3):
                            with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                                net = bottleneck(net, depth=base_depth * 4,
                                                 depth_bottleneck=base_depth, stride=1, rate=8 * multi_grid[i])
                        net = slim.utils.collect_named_outputs(end_points_collection,
                                                               sc.name, net)

                    with tf.variable_scope('block7', [net]) as sc:
                        base_depth = 512

                        for i in range(3):
                            with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                                net = bottleneck(net, depth=base_depth * 4,
                                                 depth_bottleneck=base_depth, stride=1, rate=16 * multi_grid[i])
                        net = slim.utils.collect_named_outputs(end_points_collection,
                                                               sc.name, net)

                classes = slim.conv2d(net, 2, 1, activation_fn=tf.nn.softmax, normalizer_fn=None, scope='classes')
    return classes


def Tiramisu2_ASPP_2357(input_image, dropout_num, aspp=True, reuse=None):
    input_image /= 255.0
    depth = 50
    if aspp:
        multi_grid = (1, 2, 4)
    else:
        multi_grid = (1, 2, 1)
    scope = 'resnet_v1_{}'.format(depth)
    with tf.variable_scope(scope, [input_image], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
    with tf.variable_scope('Tiramisu'):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            normalizer_fn=slim.batch_norm,
                            activation_fn=tf.nn.relu6,
                            weights_regularizer=slim.l2_regularizer((0.0005)),
                            weights_initializer=slim.variance_scaling_initializer()):
            with slim.arg_scope([slim.batch_norm], scale=True, is_training=True):
                conv1 = slim.conv2d(input_image, 16, 3)
                conv1 = tf.nn.dropout(conv1, dropout_num)
                conv1 = slim.conv2d(conv1, 16, 3)

                conv2 = slim.conv2d(conv1, 32, 3, stride=2)
                conv2 = tf.nn.dropout(conv2, dropout_num)  # 96*72
                conv2 = slim.conv2d(conv2, 32, 3)

                conv3 = slim.conv2d(conv2, 48, 3, stride=2)  # 48*36
                conv3 = dense_block_3(conv3, 48, 'conv3', dropout_num)

                conv4 = slim.conv2d(conv3, 64, 3, stride=2)  # 24*18
                conv4 = dense_block_3(conv4, 64, 'conv4', dropout_num)

                conv5 = slim.conv2d(conv4, 128, 3, stride=2)  # 12*9
                conv5 = tf.nn.dropout(conv5, dropout_num)
                conv5 = slim.conv2d(conv5, 128, 3)

                up1 = tf.image.resize_bilinear(conv5, (conv4.get_shape()[1], conv4.get_shape()[2]))  # 24*18
                up1 = tf.concat([conv4, up1], axis=3)
                up1 = dense_block_3(up1, 64, 'up1', dropout_num)

                up2 = tf.image.resize_bilinear(up1, (conv3.get_shape()[1], conv3.get_shape()[2]))  # 48*36
                up2 = tf.concat([conv3, up2], axis=3)
                up2 = dense_block_3(up2, 48, 'up2', dropout_num)

                up3 = tf.image.resize_bilinear(up2, (conv2.get_shape()[1], conv2.get_shape()[2]))
                up3 = tf.concat([conv2, up3], axis=3)
                up3 = slim.conv2d(up3, 32, 3)
                up3 = tf.nn.dropout(up3, dropout_num)
                up3 = slim.conv2d(up3, 32, 3)

                up4 = tf.image.resize_bilinear(up3, (conv1.get_shape()[1], conv1.get_shape()[2]))
                up4 = tf.concat([conv1, up4], axis=3)
                up4 = slim.conv2d(up4, 16, 3)
                up4 = tf.nn.dropout(up4, dropout_num)
                up4 = slim.conv2d(up4, 16, 3)
                net = up4
                if aspp:
                    with tf.variable_scope('aspp', [net]) as sc:
                        branch_1 = slim.conv2d(net, 256, [1, 1], stride=1)
                        branch = slim.conv2d(net, 64, [1, 1])
                        branch_2 = slim.conv2d(branch, 64, [3, 3], stride=1, rate=2)
                        branch_2 = slim.conv2d(branch_2, 256, [1, 1])
                        branch_3 = slim.conv2d(branch, 64, [3, 3], stride=1, rate=3)
                        branch_3 = slim.conv2d(branch_3, 256, [1, 1])
                        branch_4 = slim.conv2d(branch, 64, [3, 3], stride=1, rate=5)
                        branch_4 = slim.conv2d(branch_4, 256, [1, 1])
                        net = tf.concat([branch_1, branch_2, branch_3, branch_4], 3)
                        net = slim.conv2d(net, 256, [1, 1], stride=1, scope='1x1conv')
                else:
                    with tf.variable_scope('block5', [net]) as sc:
                        base_depth = 512
                        for i in range(3):
                            with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                                net = bottleneck(net, depth=base_depth * 4,
                                                 depth_bottleneck=base_depth, stride=1, rate=4 * multi_grid[i])

                    with tf.variable_scope('block6', [net]) as sc:
                        base_depth = 512

                        for i in range(3):
                            with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                                net = bottleneck(net, depth=base_depth * 4,
                                                 depth_bottleneck=base_depth, stride=1, rate=8 * multi_grid[i])
                        net = slim.utils.collect_named_outputs(end_points_collection,
                                                               sc.name, net)

                    with tf.variable_scope('block7', [net]) as sc:
                        base_depth = 512

                        for i in range(3):
                            with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                                net = bottleneck(net, depth=base_depth * 4,
                                                 depth_bottleneck=base_depth, stride=1, rate=16 * multi_grid[i])
                        net = slim.utils.collect_named_outputs(end_points_collection,
                                                               sc.name, net)

                classes = slim.conv2d(net, 2, 1, activation_fn=tf.nn.softmax, normalizer_fn=None, scope='classes')
    return classes

def ConvBlock(inputs, n_filters, kernel_size=[3, 3]):
	"""
	Builds the conv block for MobileNets
	Apply successivly a 2D convolution, BatchNormalization relu
	"""
	# Skip pointwise by setting num_outputs=Non
	net = slim.conv2d(inputs, n_filters, kernel_size=[1, 1], activation_fn=None)
	net = slim.batch_norm(net, fused=True)
	net = tf.nn.relu(net)
	return net

def DepthwiseSeparableConvBlock(inputs, n_filters, kernel_size=[3, 3]):
	"""
	Builds the Depthwise Separable conv block for MobileNets
	Apply successivly a 2D separable convolution, BatchNormalization relu, conv, BatchNormalization, relu
	"""
	# Skip pointwise by setting num_outputs=None
	net = slim.separable_convolution2d(inputs, num_outputs=None, depth_multiplier=1, kernel_size=[3, 3], activation_fn=None)

	net = slim.batch_norm(net, fused=True)
	net = tf.nn.relu(net)
	net = slim.conv2d(net, n_filters, kernel_size=[1, 1], activation_fn=None)
	net = slim.batch_norm(net, fused=True)
	net = tf.nn.relu(net)
	return net

def conv_transpose_block(inputs, n_filters, kernel_size=[3, 3]):
	"""
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	"""
	net = slim.conv2d_transpose(inputs, n_filters, kernel_size=[3, 3], stride=[2, 2], activation_fn=None)
	net = tf.nn.relu(slim.batch_norm(net))
	return net

def build_mobile_unet(inputs, preset_model):
    with tf.variable_scope('Tiramisu'):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            normalizer_fn=slim.batch_norm,
                            activation_fn=tf.nn.relu6,
                            weights_regularizer=slim.l2_regularizer((0.0005)),
                            weights_initializer=slim.variance_scaling_initializer()):
            with slim.arg_scope([slim.batch_norm], scale=True, is_training=True):
                has_skip = False
                if preset_model == "MobileUNet":
                    has_skip = False
                elif preset_model == "MobileUNet-Skip":
                    has_skip = True
                else:
                    raise ValueError(
                        "Unsupported MobileUNet model '%s'. This function only supports MobileUNet and MobileUNet-Skip" % (
                            preset_model))

                   #####################
                # Downsampling path #
                #####################
                net = ConvBlock(inputs, 64)
                net = DepthwiseSeparableConvBlock(net, 64)
                net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
                skip_1 = net

                net = DepthwiseSeparableConvBlock(net, 128)
                net = DepthwiseSeparableConvBlock(net, 128)
                net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
                skip_2 = net

                net = DepthwiseSeparableConvBlock(net, 256)
                net = DepthwiseSeparableConvBlock(net, 256)
                net = DepthwiseSeparableConvBlock(net, 256)
                net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
                skip_3 = net

                net = DepthwiseSeparableConvBlock(net, 512)
                net = DepthwiseSeparableConvBlock(net, 512)
                net = DepthwiseSeparableConvBlock(net, 512)
                net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
                skip_4 = net

                net = DepthwiseSeparableConvBlock(net, 512)
                net = DepthwiseSeparableConvBlock(net, 512)
                net = DepthwiseSeparableConvBlock(net, 512)
                net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')


                #####################
                # Upsampling path #
                #####################
                net = conv_transpose_block(net, 512)
                net = DepthwiseSeparableConvBlock(net, 512)
                net = DepthwiseSeparableConvBlock(net, 512)
                net = DepthwiseSeparableConvBlock(net, 512)
                if has_skip:
                    net = tf.add(net, skip_4)

                net = conv_transpose_block(net, 512)
                net = DepthwiseSeparableConvBlock(net, 512)
                net = DepthwiseSeparableConvBlock(net, 512)
                net = DepthwiseSeparableConvBlock(net, 256)
                if has_skip:
                    net = tf.add(net, skip_3)

                net = conv_transpose_block(net, 256)
                net = DepthwiseSeparableConvBlock(net, 256)
                net = DepthwiseSeparableConvBlock(net, 256)
                net = DepthwiseSeparableConvBlock(net, 128)
                if has_skip:
                    net = tf.add(net, skip_2)

                net = conv_transpose_block(net, 128)
                net = DepthwiseSeparableConvBlock(net, 128)
                net = DepthwiseSeparableConvBlock(net, 64)
                if has_skip:
                    net = tf.add(net, skip_1)

                net = conv_transpose_block(net, 64)
                net = DepthwiseSeparableConvBlock(net, 64)
                net = DepthwiseSeparableConvBlock(net, 64)

                net = slim.conv2d(net, 2, 1, activation_fn=tf.nn.softmax, normalizer_fn=None, scope='classes')
    return net


# 使用降采样卷积/空洞卷积/densenet结构 非常有效
# 减薄 + 降采样卷积
def Tiramisu2_no_dropout(input_image):
    input_image /= 255.0
    with tf.variable_scope('Tiramisu'):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            normalizer_fn=slim.batch_norm,
                            activation_fn=tf.nn.relu6,
                            weights_regularizer=slim.l2_regularizer((0.0005)),
                            weights_initializer=slim.variance_scaling_initializer()):
            with slim.arg_scope([slim.batch_norm], scale=True, is_training=False):
                conv1 = slim.conv2d(input_image, 16, 3)
                conv1 = slim.conv2d(conv1, 16, 3)

                conv2 = slim.conv2d(conv1, 32, 3, stride=2)
                conv2 = slim.conv2d(conv2, 32, 3)

                conv3 = slim.conv2d(conv2, 48, 3, stride=2)  # 48*36
                conv3 = dense_block_3_no_dropout(conv3, 48, 'conv3')

                conv4 = slim.conv2d(conv3, 64, 3, stride=2)  # 24*18
                conv4 = dense_block_3_no_dropout(conv4, 64, 'conv4')

                conv5 = slim.conv2d(conv4, 128, 3, stride=2)  # 12*9
                conv5 = slim.conv2d(conv5, 128, 3)

                up1 = tf.image.resize_bilinear(conv5, (conv4.get_shape()[1], conv4.get_shape()[2]))  # 24*18
                up1 = tf.concat([conv4, up1], axis=3)
                up1 = dense_block_3_no_dropout(up1, 64, 'up1')

                up2 = tf.image.resize_bilinear(up1, (conv3.get_shape()[1], conv3.get_shape()[2]))  # 48*36
                up2 = tf.concat([conv3, up2], axis=3)
                up2 = dense_block_3_no_dropout(up2, 48, 'up2')

                up3 = tf.image.resize_bilinear(up2, (conv2.get_shape()[1], conv2.get_shape()[2]))
                up3 = tf.concat([conv2, up3], axis=3)
                up3 = slim.conv2d(up3, 32, 3)
                # up3 = tf.nn.dropout(up3, dropout_num)
                up3 = slim.conv2d(up3, 32, 3)

                up4 = tf.image.resize_bilinear(up3, (conv1.get_shape()[1], conv1.get_shape()[2]))
                up4 = tf.concat([conv1, up4], axis=3)
                up4 = slim.conv2d(up4, 16, 3)
                # up4 = tf.nn.dropout(up4, dropout_num)
                up4 = slim.conv2d(up4, 16, 3)

                classes = slim.conv2d(up4, 2, 1, activation_fn=tf.nn.softmax, normalizer_fn=None, scope='classes')
    return classes



# Tiramisu + 多分辨率监督(效果不明显）
def Tiramisu5(input_image, dropout_num):
    input_image /= 255.0
    with tf.variable_scope('Tiramisu'):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            normalizer_fn=slim.batch_norm,
                            activation_fn=tf.nn.relu6,
                            weights_regularizer=slim.l2_regularizer((0.0005)),
                            weights_initializer=slim.variance_scaling_initializer()):
            with slim.arg_scope([slim.batch_norm], scale=True, is_training=True):
                conv1 = slim.conv2d(input_image, 16, 3)
                conv1 = tf.nn.dropout(conv1, dropout_num)
                conv1 = slim.conv2d(conv1, 16, 3)

                down1 = slim.conv2d(conv1, 32, 3, stride=2)
                conv2 = tf.nn.dropout(down1, dropout_num)  # 96*72
                conv2 = slim.conv2d(conv2, 32, 3)

                down2 = slim.conv2d(conv2, 64, 3, stride=2)  # 48*36
                conv3 = dense_block_3(down2, 64, 'conv3', dropout_num)

                down3 = slim.conv2d(conv3, 128, 3, stride=2)  # 24*18
                conv4 = dense_block_3(down3, 128, 'conv4', dropout_num)

                predict0 = slim.conv2d(conv4, 2, 1, normalizer_fn=None, activation_fn=tf.nn.softmax, scope='predict0')

                up1 = tf.image.resize_bilinear(conv4, conv3.get_shape()[1:3], True)  # 48*36
                predict0_resize = tf.image.resize_bilinear(predict0, conv3.get_shape()[1:3], True)
                up1 = tf.concat([conv3, up1, predict0_resize], axis=3)
                up1 = dense_block_3(up1, 64, 'up1', dropout_num)

                predict1 = slim.conv2d(up1, 2, 1, normalizer_fn=None, activation_fn=tf.nn.softmax, scope='predict1')

                up2 = tf.image.resize_bilinear(up1, conv2.get_shape()[1:3], True)  # 96*72
                predict1_resize = tf.image.resize_bilinear(predict1, conv2.get_shape()[1:3], True)
                up2 = tf.concat([conv2, up2, predict1_resize], axis=3)
                up2 = slim.conv2d(up2, 32, 3)
                up2 = tf.nn.dropout(up2, dropout_num)
                up2 = slim.conv2d(up2, 32, 3)

                predict2 = slim.conv2d(up2, 2, 1, normalizer_fn=None, activation_fn=tf.nn.softmax, scope='predict2')

                up3 = tf.image.resize_bilinear(up2, conv1.get_shape()[1:3], True)
                predict2_resize = tf.image.resize_bilinear(predict2, conv1.get_shape()[1:3], True)
                up3 = tf.concat([conv1, up3, predict2_resize], axis=3)
                up3 = slim.conv2d(up3, 16, 3)
                up3 = tf.nn.dropout(up3, dropout_num)
                up3 = slim.conv2d(up3, 16, 3)

                predict = slim.conv2d(up3, 2, 1, activation_fn=tf.nn.softmax, normalizer_fn=None, scope='predict')

    return predict0, predict1, predict2, predict


# shuffle_v2
def Tiramisu6(input_image, dropout_num):
    input_image /= 255.0
    with tf.variable_scope('Tiramisu'):
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            activation_fn=tf.nn.relu6,
                            weights_regularizer=slim.l2_regularizer((0.0005)),
                            weights_initializer=slim.variance_scaling_initializer()):
            with slim.arg_scope([slim.batch_norm], scale=True, is_training=True):
                conv1 = slim.conv2d(input_image, 16, 3)
                conv1 = tf.nn.dropout(conv1, dropout_num)
                conv1 = slim.conv2d(conv1, 16, 3)

                conv2 = shuffle_v2(conv1, 'conv2_1', stride=2)
                conv2 = tf.nn.dropout(conv2, dropout_num)  # 32
                conv2 = shuffle_v2(conv2, 'conv2_2')

                conv3 = shuffle_v2(conv2, 'conv3_1', stride=2)  # 64
                conv3 = dense_block_3_shuffle_v2(conv3, 'conv3_2', dropout_num)

                conv4 = shuffle_v2(conv3, 'conv4_1', stride=2)  # 128
                conv4 = dense_block_3_shuffle_v2(conv4, 'conv4_2', dropout_num)

                conv5 = shuffle_v2(conv4, 'conv5_1', stride=2)  # 256
                conv5 = tf.nn.dropout(conv5, dropout_num)
                conv5 = shuffle_v2(conv5, 'conv5_2')

                up1 = tf.image.resize_bilinear(conv5, conv4.get_shape()[1:3], True)
                up1 = tf.concat([conv4, up1], axis=3)  # 256 + 128
                up1 = slim.conv2d(up1, 128, 1)
                up1 = dense_block_3_shuffle_v2(up1, 'up1', dropout_num)

                up2 = tf.image.resize_bilinear(up1, conv3.get_shape()[1:3], True)
                up2 = tf.concat([conv3, up2], axis=3)  # 128 + 64
                up2 = slim.conv2d(up2, 64, 1)
                up2 = dense_block_3_shuffle_v2(up2, 'up2', dropout_num)

                up3 = tf.image.resize_bilinear(up2, conv2.get_shape()[1:3], True)
                up3 = tf.concat([conv2, up3], axis=3)  # 64 + 32
                up3 = slim.conv2d(up3, 32, 1)
                up3 = shuffle_v2(up3, 'up3_1')
                up3 = tf.nn.dropout(up3, dropout_num)
                up3 = shuffle_v2(up3, 'up3_2')

                up4 = tf.image.resize_bilinear(up3, conv1.get_shape()[1:3], True)
                up4 = tf.concat([conv1, up4], axis=3)
                up4 = slim.conv2d(up4, 16, 3)
                up4 = tf.nn.dropout(up4, dropout_num)
                up4 = slim.conv2d(up4, 16, 3)

                classes = slim.conv2d(up4, 2, 1, normalizer_fn=None,
                                      activation_fn=tf.nn.softmax, scope='classes')
                return classes


# Tiramisu7对比  反卷积vs双线性插值(训练结果表明反卷积必要性不大，已删）
def Tiramisu7_3(input_image, dropout_num):
    input_image /= 255.0
    with tf.variable_scope('Tiramisu'):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            padding='SAME',
                            normalizer_fn=slim.batch_norm,
                            activation_fn=tf.nn.relu6,
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            weights_initializer=slim.variance_scaling_initializer()):
            with slim.arg_scope([slim.batch_norm], scale=True, is_training=True):
                with tf.variable_scope('encoder'):
                    conv1 = slim.conv2d(input_image, 16, 3)  # test5
                    conv1 = tf.nn.dropout(conv1, dropout_num)
                    conv1 = slim.conv2d(conv1, 16, 3)

                    conv2 = slim.conv2d(conv1, 32, 3, stride=2)
                    conv2 = tf.nn.dropout(conv2, dropout_num)  # 96*72
                    conv2 = slim.conv2d(conv2, 32, 3)

                    conv3 = slim.conv2d(conv2, 48, 3, stride=2)  # 48*36
                    conv3 = tf.nn.dropout(conv3, dropout_num)
                    conv3 = dense_block_4(conv3, 48, 'conv3', dropout_num)

                    conv4 = slim.conv2d(conv3, 64, 3, stride=2)  # 24*18
                    conv4 = tf.nn.dropout(conv4, dropout_num)
                    conv4 = dense_block_4(conv4, 64, 'conv4', dropout_num)

                    conv5 = slim.conv2d(conv4, 128, 3, stride=2)  # 12*9
                    conv5 = tf.nn.dropout(conv5, dropout_num)
                    conv5 = slim.conv2d(conv5, 128, 3)

                with tf.variable_scope('decoder'):
                    up1 = tf.image.resize_bilinear(conv5, conv4.get_shape()[1:3], True)
                    up1 = tf.concat([conv4, up1], axis=3)
                    up1 = dense_block_4(up1, 64, 'up1', dropout_num)

                    up2 = tf.image.resize_bilinear(up1, conv3.get_shape()[1:3], True)
                    up2 = tf.concat([conv3, up2], axis=3)
                    up2 = dense_block_4(up2, 48, 'up2', dropout_num)

                    up3 = tf.image.resize_bilinear(up2, conv2.get_shape()[1:3], True)
                    up3 = tf.concat([conv2, up3], axis=3)
                    up3 = slim.conv2d(up3, 32, 3)
                    up3 = tf.nn.dropout(up3, dropout_num)
                    up3 = slim.conv2d(up3, 32, 3)

                    up4 = tf.image.resize_bilinear(up3, conv1.get_shape()[1:3], True)
                    up4 = tf.concat([conv1, up4], axis=3)
                    up4 = slim.conv2d(up4, 16, 3)
                    up4 = tf.nn.dropout(up4, dropout_num)
                    up4 = slim.conv2d(up4, 16, 3)

                classes = slim.conv2d(up4, 2, 1, normalizer_fn=None,
                                      activation_fn=tf.nn.softmax, scope='classes')
                return classes


# Tiramisu7_3对比  seperable_block4（可分离卷积并没有速度提升，反而由于加深了网络，使得inference的速度变慢了10%左右)
def Tiramisu7_4(input_image, dropout_num):
    input_image /= 255.0
    with tf.variable_scope('Tiramisu'):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            normalizer_fn=slim.batch_norm,
                            activation_fn=tf.nn.relu6,
                            weights_regularizer=slim.l2_regularizer((0.0005)),
                            weights_initializer=slim.variance_scaling_initializer()):
            with slim.arg_scope([slim.batch_norm], scale=True, is_training=True):
                with tf.variable_scope('encoder'):
                    conv1 = slim.conv2d(input_image, 16, 3)  # test5
                    conv1 = tf.nn.dropout(conv1, dropout_num)
                    conv1 = slim.conv2d(conv1, 16, 3)

                    conv2 = separable_conv(conv1, 32, 'conv2_1', stride=2)
                    conv2 = tf.nn.dropout(conv2, dropout_num)  # 96*72
                    conv2 = separable_conv(conv2, 32, 'conv2_2')

                    conv3 = separable_conv(conv2, 48, 'conv3_1', stride=2)  # 48*36
                    conv3 = tf.nn.dropout(conv3, dropout_num)
                    conv3 = dense_block_4_separable(conv3, 48, 'conv3_2', dropout_num)

                    conv4 = separable_conv(conv3, 64, 'conv4_1', stride=2)  # 24*18
                    conv4 = tf.nn.dropout(conv4, dropout_num)
                    conv4 = dense_block_4_separable(conv4, 64, 'conv4_2', dropout_num)

                    conv5 = separable_conv(conv4, 80, 'conv5_1', stride=2)  # 12*9
                    conv5 = tf.nn.dropout(conv5, dropout_num)
                    conv5 = separable_conv(conv5, 80, 'conv5_2')

                with tf.variable_scope('decoder'):
                    up1 = tf.image.resize_bilinear(conv5, conv4.get_shape()[1:3], True)
                    up1 = tf.concat([conv4, up1], axis=3)
                    up1 = dense_block_4_separable(up1, 64, 'up1', dropout_num)

                    up2 = tf.image.resize_bilinear(up1, conv3.get_shape()[1:3], True)
                    up2 = tf.concat([conv3, up2], axis=3)
                    up2 = dense_block_4_separable(up2, 48, 'up2', dropout_num)

                    up3 = tf.image.resize_bilinear(up2, conv2.get_shape()[1:3], True)
                    up3 = tf.concat([conv2, up3], axis=3)
                    up3 = separable_conv(up3, 32, 'up3_1')
                    up3 = tf.nn.dropout(up3, dropout_num)
                    up3 = separable_conv(up3, 32, 'up3_2')

                    up4 = tf.image.resize_bilinear(up3, conv1.get_shape()[1:3], True)
                    up4 = tf.concat([conv1, up4], axis=3)
                    up4 = slim.conv2d(up4, 16, 3)  # first and last convolutions remain the same
                    up4 = tf.nn.dropout(up4, dropout_num)
                    up4 = slim.conv2d(up4, 16, 3)

                classes = slim.conv2d(up4, 2, 1, normalizer_fn=None,
                                      activation_fn=tf.nn.softmax, scope='classes')
                return classes


# Tiramisu dense_block4 gesture only（加入姿态点，暂时搁置）
def Tiramisu8(input_image, dropout_num):
    input_image /= 255.0
    with tf.variable_scope('Tiramisu'):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            normalizer_fn=slim.batch_norm,
                            activation_fn=tf.nn.relu6,
                            weights_regularizer=slim.l2_regularizer((0.0005)),
                            weights_initializer=slim.variance_scaling_initializer()):
            with slim.arg_scope([slim.batch_norm], scale=True, is_training=True):
                with tf.variable_scope('encoder'):
                    conv1 = slim.conv2d(input_image, 16, 3)
                    conv1 = tf.nn.dropout(conv1, dropout_num)
                    conv1 = slim.conv2d(conv1, 16, 3)

                    conv2 = slim.conv2d(conv1, 32, 3, stride=2)
                    conv2 = tf.nn.dropout(conv2, dropout_num)  # 96*72
                    conv2 = slim.conv2d(conv2, 32, 3)

                    conv3 = slim.conv2d(conv2, 48, 3, stride=2)  # 48*36
                    conv3 = dense_block_4(conv3, 48, 'conv3', dropout_num)

                    conv4 = slim.conv2d(conv3, 64, 3, stride=2)  # 24*18
                    conv4 = dense_block_4(conv4, 64, 'conv4', dropout_num)

                    conv5 = slim.conv2d(conv4, 80, 3, stride=2)  # 12*9
                    conv5 = tf.nn.dropout(conv5, dropout_num)
                    conv5 = slim.conv2d(conv5, 80, 3)

                with tf.variable_scope('decoder'):
                    up1 = tf.image.resize_bilinear(conv5, conv4.get_shape()[1:3], True)
                    up1 = tf.concat([conv4, up1], axis=3)
                    up1 = dense_block_4(up1, 64, 'up1', dropout_num)

                    up2 = tf.image.resize_bilinear(up1, conv3.get_shape()[1:3], True)
                    up2 = tf.concat([conv3, up2], axis=3)
                    up2 = dense_block_4(up2, 48, 'up2', dropout_num)

                    # up3 = tf.image.resize_bilinear(up2, conv2.get_shape()[1:3], True)
                    # up3 = tf.concat([conv2, up3], axis=3)
                    # up3 = slim.conv2d(up3, 32, 3)
                    # up3 = tf.nn.dropout(up3, dropout_num)
                    # up3 = slim.conv2d(up3, 32, 3)
                    #
                    # up4 = tf.image.resize_bilinear(up3, conv1.get_shape()[1:3], True)
                    # up4 = tf.concat([conv1, up4], axis=3)
                    # up4 = slim.conv2d(up4, 16, 3)
                    # up4 = tf.nn.dropout(up4, dropout_num)
                    # up4 = slim.conv2d(up4, 16, 3)

                with tf.variable_scope('gesture'):
                    fc1 = slim.flatten(up2, scope='fc1')

                    fc2 = slim.fully_connected(fc1, 1024, weights_regularizer=slim.l2_regularizer(0.0001), scope='fc2')
                    print(fc2.get_shape())
                    fc3 = slim.fully_connected(fc2, 1024, weights_regularizer=slim.l2_regularizer(0.0001), scope='fc3')
                    keypoints_scores = slim.fully_connected(fc3,
                                                            140,
                                                            activation_fn=None,
                                                            weights_regularizer=slim.l2_regularizer(0.0001),
                                                            biases_regularizer=slim.l2_regularizer(0.0001),
                                                            scope='keypoints_scores')
                return keypoints_scores