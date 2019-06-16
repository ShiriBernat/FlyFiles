# -*- coding: utf-8 -*-
import sys
import tensorflow as tf
import utils
from flask import Flask, request, jsonify

# Server Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "C:/Users/bb02/Desktop/Listener/Checkpoints", "Checkpoint directory from training run")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))

positive_data_file = "./pos.txt"
negative_data_file = "./neg.txt"
x, y, vocabulary, vocabulary_inv = utils.load_data(positive_data_file, negative_data_file)

app = Flask(__name__)

def run_model(graph, sess, x, y, vocabulary, raw_x):
    # Load the saved meta graph and restore variables
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)

    # Get the placeholders from the graph by name
    input_x = graph.get_operation_by_name("input_x").outputs[0]
    dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
    predictions = graph.get_operation_by_name("output/predictions").outputs[0]

    predicted_result = sess.run(predictions, {input_x: raw_x, dropout_keep_prob: 1.0})
    return predicted_result

@app.route('/predict/', methods=['POST'])
def predict():
    text = request.get_json()
    print(text)

    if not request.json or not 'text' in request.json:
        exit(400)

    print('create graph')
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            try:
                text = request.json['text']
                print('request body:', text)
                raw_x = utils.sentence_to_index(text, vocabulary, x.shape[1])
                print('input row', raw_x)
                predicted_results = run_model(graph, sess, x, y, vocabulary, raw_x)
                print('predict res', predicted_results)
                if predicted_results[0] == 0:
                    return_res = "computer science"
                else:
                    return_res = "others"

                return jsonify({'result': return_res})
            except Exception as e:
                print(e)

# from client: {text: "this article is about robotics and vision"}
# to client: {result: 0}
if __name__ == '__main__':
    app.run(debug=True)
