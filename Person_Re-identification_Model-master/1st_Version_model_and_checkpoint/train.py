import os
import glob
import argparse
import numpy as np
import cv2
import tensorflow as tf 
from train_helper import *

parser = argparse.ArgumentParser(description='Training options.')
parser.add_argument('-d', '--data', metavar='--data', type=str, nargs='?', help='Training data path')
args = parser.parse_args()

'''
data_path = args.data
os.system('python3 create_tf_record.py --tfrecord_filename=mars --dataset_dir=%s' % (args.data))
'''

data_filename = os.path.join(args.data, 'data_summary.txt')
with tf.io.gfile.GFile(data_filename, 'r') as f:
    num_validatiaon = f.readline()
    num_dataset = f.readline()

    print('Found %d images in the training data' % (int(num_dataset) - int(num_validatiaon)))
    print('Found %d images in the validataion data' % (int(num_validatiaon)))
    training_data_num = int(num_dataset) - int(num_validatiaon)

def main(argv=None):
    BATCH_SIZE = 32
    num_epochs = 200

    train_dataset = combine_dataset(batch_size=BATCH_SIZE, image_size=[256, 128], same_prob=0.5, diff_prob=0.5, train=True)
    val_dataset = combine_dataset(batch_size=BATCH_SIZE, image_size=[256, 128], same_prob=0.5, diff_prob=0.5, train=False)
    handle = tf.compat.v1.placeholder(tf.string, shape=[])

    iterator = tf.data.Iterator.from_string_handle(
                handle, train_dataset.output_types, train_dataset.output_shapes)

    train_iterator = train_dataset.make_one_shot_iterator()
    val_iterator = val_dataset.make_one_shot_iterator()

    left, right = iterator.get_next()
    left_input_im, left_label, left_addr = left
    right_input_im, right_label, right_addr = right

    logits, model_left, model_right = inference(left_input_im, right_input_im)
    loss(logits, left_label, right_label)

    total_loss = tf.losses.get_total_loss()
    global_step = tf.Variable(0, trainable=False)

    params = tf.compat.v1.trainable_variables()
    gradients = tf.gradients(total_loss, params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00001)
    updates = optimizer.apply_gradients(zip(gradients, params), global_step=global_step)


    global_init = tf.compat.v1.variables_initializer(tf.compat.v1.global_variables())

    saver = tf.compat.v1.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(global_init)

        # setup tensorboard
        if not os.path.exists('train_log'):
            os.makedirs('train_log')
        tf.compat.v1.summary.scalar('step', global_step)
        tf.compat.v1.summary.scalar('loss', total_loss)
        for var in tf.compat.v1.trainable_variables():
            tf.compat.v1.summary.histogram(var.op.name, var)
        merged = tf.compat.v1.summary.merge_all()
        writer = tf.compat.v1.summary.FileWriter('train_log', sess.graph)

        training_handle = sess.run(train_iterator.string_handle())
        validation_handle = sess.run(val_iterator.string_handle())

        num_iterations = training_data_num // BATCH_SIZE

        for epoch in range(num_epochs):
            print('epoch : ', epoch, ' / ', num_epochs)
            for iteration in range(num_iterations):
                feed_dict_train = {handle:training_handle}

                loss_train, _, summary_str = sess.run([total_loss, updates, merged], feed_dict_train)
                writer.add_summary(summary_str, epoch)
                print("iteration : %d / %d - Loss : %f" % (iteration, num_iterations, loss_train))

            feed_dict_val = {handle: validation_handle}
            val_loss = sess.run([total_loss], feed_dict_val)
            print('========================================')
            print("epoch : %d - Validation Loss : %f" % (epoch, val_loss[0]))
            print('========================================')

            if not os.path.exists("MARS_model/"):
                os.makedirs("MARS_model/")
            saver.save(sess, "MARS_model/model.ckpt")

if __name__ == '__main__':
    tf.compat.v1.app.run()
