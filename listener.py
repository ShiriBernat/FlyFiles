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
    if filename.startswith("0") and dir.endswith("CS"):
        res_list[0] = res_list[0]+1
    # FN
    if filename.startswith("0") and dir.endswith("OT"):
        res_list[1] = res_list[1]+1
    # TN
    if filename.startswith("1") and dir.endswith("OT"):
        res_list[2] = res_list[2]+1
    # FP
    if filename.startswith("1") and dir.endswith("CS"):
        res_list[3] = res_list[3]+1
    # EXP
    if dir.endswith("Default"):
        res_list[4] = res_list[4]+1

    print(res_list)
    return res_list

def run_model(args, graph, sess, x, y, vocabulary, text):
    sentences_padded = utils.pad_sentences(text, maxlen=x.shape[1])
    raw_x, dummy_y = utils.build_input_data(sentences_padded, [0], vocabulary)

    # Load the saved meta graph and restore variables
    checkpoint_file = tf.train.latest_checkpoint(args.checkpoint_dir)
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)

    # Get the placeholders from the graph by name
    input_x = graph.get_operation_by_name("input_x").outputs[0]
    dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
    predictions = graph.get_operation_by_name("output/predictions").outputs[0]

    predicted_result = sess.run(predictions, {input_x: raw_x, dropout_keep_prob: 1.0})
    return predicted_result

def load_file_text(dir, file):
    if file.endswith(".pdf"):
        text = utils.convert(os.path.join(dir, file))
        text = utils.clean_str(text)

    if file.endswith(".txt"):
        with open(os.path.join(dir, file), encoding="utf8") as f:
            text = f.read().replace('\n', '')
        f.close()

    for i in range(599, len(text), 600):
        text = text[:i] + '\n' + text[i:]
    text = text.splitlines()
    return text

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
                # iterate over the input folder in order to classify the files as cs/others
                after = dict([(f, None) for f in os.listdir(path_to_watch)])
                added = [f for f in after if not f in before]
                if added:
                    for filename in os.listdir(args.dir):
                        print('start ', filename)
                        try:
                            res = 0
                            if filename.endswith(".pdf") or filename.endswith(".txt"):
                                text = load_file_text(args.dir, filename)
                                random.shuffle(text)
                                test_text = text[:600][:100]

                                # send the text to test
                                res = run_model(args, graph, sess, x, y, vocabulary, test_text)
                                count = Counter(list(res)).most_common(2)
                                percentage = len(test_text)*0.75
                                print("File: "+str(filename)+"\n", "Count: "+str(count)+"\n",
                                      "75% Percentage: "+str(percentage))

                                if len(test_text) <= 80 and \
                                        (count[0][0] == 0 or (len(count) > 1 and float(count[0][1]) < percentage)):
                                        directory = args.cs_f
                                elif count[0][0] == 0 or (len(count) > 1 and count[0][1] == count[1][0]):
                                    directory = args.cs_f
                                else:
                                    directory = args.otr_f
                            else:
                                print(str(filename) + " is not a text file\n------move it to all_files folder------")
                                directory = args.all_f
                        except:
                            print("an error occurred during classifing the file: %s" % filename)
                            directory = args.all_f

                        print(" Dir: "+directory+"\n")
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        if (os.path.isfile(os.path.join(directory, filename)) == False):
                            shutil.move(os.path.join(args.dir, filename), directory)
                        else:
                            print("the file already exists in the destination folder\n"
                                  "the file removes from the downloads folder")
                            os.remove(os.path.join(args.dir, filename))
                        evaluate_res = evaluate_preformance(filename, directory, evaluate_res)

                before = after


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, default='/Listener/Downloads',
                        help='directory to Downloads folder')
    parser.add_argument('--cs_f', type=str, default='/Listener/CS',
                        help='directory to output folder of the cs files')
    parser.add_argument('--otr_f', type=str, default='/Listener/Others',
                        help='directory to output folder of the others files')
    parser.add_argument('--all_f', type=str, default='/Listener/Default',
                        help='directory to folder all the others files')
    parser.add_argument('--checkpoint_dir', type=str, default='/Listener/Checkpoints',
                        help='model directory to store checkpoints models')
    args = parser.parse_args()
    listener(args)


if __name__ == '__main__':
    main()
