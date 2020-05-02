import os
import glob
import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def make_single_dataset(image_size=[256, 128], tfrecords_path="dataset_dir/mars_train_00000-of-00001.tfrecord", shuffle_buffer_size=2000, repeat=True, train=True):
    image_size = tf.cast(image_size, tf.int32)

    def _parse_function(example_proto):
        features = {'image/class/label': tf.io.FixedLenFeature((), tf.int64, default_value=1),
	    	'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=""),
	    	'image/height': tf.io.FixedLenFeature([], tf.int64),
	    	'image/width': tf.io.FixedLenFeature([], tf.int64),
	    	'image/format': tf.io.FixedLenFeature((), tf.string, default_value="")}

        parsed_features = tf.io.parse_single_example(example_proto, features)
        image_buffer = parsed_features['image/encoded']

        image = tf.image.decode_jpeg(image_buffer,channels=3)
        image = tf.cast(image, tf.float32)

        S = tf.stack([tf.cast(parsed_features['image/height'], tf.int32),
    		tf.cast(parsed_features['image/width'], tf.int32), 3])
        image = tf.reshape(image, S)

        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [256, 128])

        return image, parsed_features['image/class/label'], parsed_features['image/format']

    filenames = [tfrecords_path]

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(_parse_function, num_parallel_calls=8)
    return dataset

def create_filter_func(same_prob, diff_prob):
    def filter_func(left, right):
        _, right_label, _ = left
        _, left_label, _ = right

        label_cond = tf.equal(right_label, left_label)

        different_labels = tf.fill(tf.shape(label_cond), diff_prob)
        same_labels = tf.fill(tf.shape(label_cond), same_prob)

        weights = tf.where(label_cond, same_labels, different_labels)
        random_tensor = tf.random.uniform(shape=tf.shape(weights))

        return weights > random_tensor

    return filter_func

def combine_dataset(batch_size, image_size, same_prob, diff_prob, repeat=True, train=True):
    dataset_left = make_single_dataset(image_size, repeat=repeat, train=train)
    dataset_right = make_single_dataset(image_size, repeat=repeat, train=train)

    dataset = tf.data.Dataset.zip((dataset_left, dataset_right))

    if train:
        filter_func = create_filter_func(same_prob, diff_prob)
        dataset = dataset.filter(filter_func)
    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

def model(input, reuse=False):
    print(np.shape(input))
    with tf.name_scope("model"):
        with tf.variable_scope("conv1") as scope:
            net = tf.contrib.layers.conv2d(input, 256, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='valid')
            net = tf.nn.relu(net)
            net = tf.layers.batch_normalization(net, fused=True)
            print("1st", np.shape(net))

        with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='valid')
            net = tf.nn.relu(net)
            net = tf.layers.batch_normalization(net, fused=True)
            print("2nd", np.shape(net))

        with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, 64, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='valid')
            net = tf.nn.relu(net)
            net = tf.layers.batch_normalization(net, fused=True)
            print("3rd", np.shape(net))

        with tf.variable_scope("conv4") as scope:
            net = tf.contrib.layers.conv2d(net, 32, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='valid')
            net = tf.nn.relu(net)
            net = tf.layers.batch_normalization(net, fused=True)
            print("4rth", np.shape(net))

        net = tf.contrib.layers.flatten(net)
        print(np.shape(net))

        net = tf.layers.dense(net, 4096, activation=tf.sigmoid)
        print(np.shape(net))

    return net

def inference(left_input_image, right_input_image):
    margin = 0.2
    with tf.variable_scope('feature_generator', reuse=tf.AUTO_REUSE) as sc:

        left_features = model(tf.layers.batch_normalization(tf.divide(left_input_image, 255.0)))
        right_features = model(tf.layers.batch_normalization(tf.divide(right_input_image, 255.0)))

    # L1 distance
    merged_features = tf.abs(tf.subtract(left_features, right_features))
    logits = tf.contrib.layers.fully_connected(merged_features, num_outputs=1, activation_fn=None)
    logits = tf.reshape(logits, [-1])
    return logits, left_features, right_features

def loss(logits, left_label, right_label):
    label = tf.equal(left_label, right_label)
    label_float = tf.cast(label, tf.float64)

    logits = tf.cast(logits, tf.float64)
    cross_entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label_float))
    tf.losses.add_loss(cross_entropy_loss)
