import tensorflow as tf
import time
import datetime
import os
import utils
import codecs
from classifier_cnn import CNNClassifier

dev_sample_percentage = .01

positive_data_file = "pos.txt"
negative_data_file = "neg.txt"

embedding_dim = 128
filter_sizes = "3,5,8"
num_filters = 128
dropout_keep_prob = 0.5
l2_reg_lambda = 0.0

batch_size = 32
num_epochs = 200
evaluate_every = 100
checkpoint_every = 100
num_checkpoints = 5

allow_soft_placement = True
log_device_placement = False

# Data Preparation
# ================

# Load data
print("Loading data...")
x_text, y, vocabulary, vocabulary_inv = utils.load_data(positive_data_file, negative_data_file)

# Split train/test set
dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
print('dev_sample_percentage: {}\nlen(y): {}\ndev_sample_index: {}'.format(dev_sample_percentage, len(y),
                                                                           dev_sample_index))
x_train, x_dev = x_text[:dev_sample_index], x_text[dev_sample_index:]
y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]

print("Vocabulary Size: {:d}".format(len(vocabulary)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Training
# ========

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=allow_soft_placement,
        log_device_placement=log_device_placement,)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.40
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        cnn = CNNClassifier(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocabulary),
            embedding_size=embedding_dim,
            filter_sizes=list(map(int, filter_sizes.split(","))),
            num_filters=num_filters,
            l2_reg_lambda=l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Visualization for embedding
        # Write meta
        with codecs.open(os.path.join(out_dir, "metadata.tsv"), 'w', encoding='utf-8') as tsv_file:
            for vocab in vocabulary_inv:
                tsv_file.write(vocab + "\n")

        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = cnn.W.name

        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = os.path.join(out_dir, 'metadata.tsv')
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(tf.summary.FileWriter(dev_summary_dir),
                                                                      config)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()

            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()

            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            if writer:
                writer.add_summary(summaries, step)


        batches = utils.batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs)

        # Training loop. For each batch...
        for batch in batches:

            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev[:100], y_dev[:100], writer=dev_summary_writer)
                print("")

            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)

                # Save the model for Embedding Visualization
                saver.save(sess, os.path.join(dev_summary_dir, "model.ckpt"), global_step=current_step)

                print("Saved model checkpoint to {}\n".format(path))
