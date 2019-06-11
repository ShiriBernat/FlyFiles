import tensorflow as tf
import argparse
import utils
import os
import shutil
import time
import random

from collections import Counter

def most_common(lst):
    return max(set(lst), key=lst.count)

def evaluate_preformance(filename, dir, res_list):
    print("file: ", filename, " classified to folder: ", dir)
    # TP
    if filename.startswith("1") and dir.endswith("CS"):
        res_list[0] = res_list[0]+1
    # FN
    if filename.startswith("1") and dir.endswith("OT"):
        res_list[1] = res_list[1]+1
    # TN
    if filename.startswith("0") and dir.endswith("OT"):
        res_list[2] = res_list[2]+1
    # FP
    if filename.startswith("0") and dir.endswith("CS"):
        res_list[3] = res_list[3]+1
    #EXP
    if(dir.endswith("Default")):
        res_list[4] = res_list[4]+1

    print(res_list)
    return res_list


def run_model(args, graph, sess, x, y, vocabulary, text):
    print('Loading data')
    # text = [list(text)]
    sentences_padded = utils.pad_sentences(text, maxlen=x.shape[1])
    raw_x, dummy_y = utils.build_input_data(sentences_padded, [0], vocabulary)

    checkpoint_file = tf.train.latest_checkpoint(args.checkpoint_dir)
    # graph = tf.Graph()
    # with graph.as_default():
    #     sess = tf.Session()
    #     with sess.as_default():
    # Load the saved meta graph and restore variables
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)

    # Get the placeholders from the graph by name
    input_x = graph.get_operation_by_name("input_x").outputs[0]
    # input_y = graph.get_operation_by_name("input_y").outputs[0]
    dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

    # Tensors we want to evaluate
    predictions = graph.get_operation_by_name("output/predictions").outputs[0]

    predicted_result = sess.run(predictions, {input_x: raw_x, dropout_keep_prob: 1.0})
    return predicted_result

def listener(args):
    path_to_watch = args.dir
    before = dict([(f, None) for f in os.listdir(path_to_watch)])
    #tp fn tn fp exp
    evaluate_res = [0, 0, 0, 0, 0]

    positive_data_file = "./pos.txt"
    negative_data_file = "./neg.txt"
    x, y, vocabulary, _ = utils.load_data(positive_data_file, negative_data_file)

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            while True:
                time.sleep(10)
                # iterate over the input folder in order to cllasify the files as cs/others
                after = dict([(f, None) for f in os.listdir(path_to_watch)])
                added = [f for f in after if not f in before]
                if added:

                    for filename in os.listdir(args.dir):
                        try:
                            res = 0
                            if filename.endswith(".pdf") or filename.endswith(".txt"):
                                if filename.endswith(".pdf"):
                                    text = utils.convert(os.path.join(args.dir, filename))
                                    text = text.splitlines()
                                else:
                                    with open(os.path.join(args.dir, filename), encoding="utf8") as f:
                                        text = f.readlines()
                                    text = [x.strip() for x in text]
                                    f.close()
                                # send the text to test
                                print("send the file:" + str(filename) + "to the model in order to classifies it")
                                random.shuffle(text)
                                test_text = text[:600][:100]
                                res = run_model(args, graph, sess, x, y, vocabulary, test_text)
                                # max_res = most_common(list(res))
                                c = Counter(list(res)).most_common(2)
                                print(c)
                                if len(c) > 1 and c[1][1] < 10:
                                    print("The file is classified as computer science\n-----move it to computer science folder-----")
                                    directory = args.cs_f
                                else:
                                    print("The file is classified as others\n-----move it to others folder-----")
                                    directory = args.otr_f
                            else:
                                print(str(filename) + " is not a text file\n------move it to all_files folder------")
                                directory = args.all_f
                        except:
                            print("an error occurred during classifing the file: %s" % filename)
                            directory = args.all_f

                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        if (os.path.isfile(os.path.join(directory, filename)) == False):
                            shutil.move(os.path.join(args.dir, filename), directory)
                        else:
                            print("the file already exists in the destination folder\nthe file removes from the downloads folder")
                            os.remove(os.path.join(args.dir, filename))
                        evaluate_res = evaluate_preformance(filename, directory, evaluate_res)

                before = after


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, default='/Downloads',
                        help='directory to Downloads folder')
    parser.add_argument('--cs_f', type=str, default='/CS',
                        help='directory to output folder of the cs files')
    parser.add_argument('--otr_f', type=str, default='/OT',
                        help='directory to output folder of the others files')
    parser.add_argument('--all_f', type=str, default='/Default',
                        help='directory to folder all the others files')
    parser.add_argument('--checkpoint_dir', type=str, default='/Checkpoints',
                        help='model directory to store checkpoints models')
    args = parser.parse_args()
    listener(args)


if __name__ == '__main__':
    main()
