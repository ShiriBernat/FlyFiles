import numpy as np
import re
import itertools
from collections import Counter

from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import os
import re


def keep_two_dup_chars(string):
    counter = 0
    last_letter = ''
    result = ""

    for curr_letter in string:
        if last_letter != curr_letter:
            counter = 0
            last_letter = curr_letter
        else:
            counter += 1

        if counter < 2:
            result += curr_letter

    return result


def clean_str(string):
    # Keep words with only two duplicated chars (for each char)
    string = keep_two_dup_chars(string)

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"http\S+", " ", string)
    string = re.sub(r"https\S+", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)

    # Signs
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_from_disk(positive_data_file, negative_data_file):
    # Load data from files
    positive_sentences = list(set(open(positive_data_file, "r", encoding="utf8").readlines()))
    positive_sentences = [s.strip() for s in positive_sentences]

    negative_sentences = list(set(open(negative_data_file, "r", encoding="utf8").readlines()))
    negative_sentences = [s.strip() for s in negative_sentences]

    # Make sure that data contains same size of negative and positive data (assume: negative < positive)
    positive_sentences = positive_sentences[:len(negative_sentences)]

    # Split by words
    X = positive_sentences + negative_sentences
    X = [clean_str(sent) for sent in X]

    positive_labels = [[1, 0] for _ in positive_sentences]
    negative_labels = [[0, 1] for _ in negative_sentences]

    Y = np.concatenate([positive_labels, negative_labels], 0)

    return [X, Y]


def pad_sentences(sentences, padding_word="<PAD/>", maxlen=0):
    """
    Pads all the sentences to the same length. The length is defined by the longest sentence.
     Returns padded sentences.
    """

    if maxlen > 0:
        sequence_length = maxlen
    else:
        sequence_length = max(len(s) for s in sentences)

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)

        replaced_newline_sentence = []
        for char in list(sentence):
            if char == "\n":
                replaced_newline_sentence.append("<NEWLINE/>")
            elif char == " ":
                replaced_newline_sentence.append("<SPACE/>")
            else:
                replaced_newline_sentence.append(char)

        new_sentence = replaced_newline_sentence + [padding_word] * num_padding

        # new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """

    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))

    # Map from index to word
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))

    # Map from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary
    """
    x = np.array([[vocabulary[word] if word in vocabulary else 0 for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def sentence_to_index(sentence, vocabulary, maxlen):
    sentence = clean_str(sentence)
    raw_input = [list(sentence)]
    sentences_padded = pad_sentences(raw_input, maxlen=maxlen)
    raw_x, dummy_y = build_input_data(sentences_padded, [0], vocabulary)
    return raw_x


def load_data(positive_data_file, negative_data_file):
    sentences, labels = load_data_from_disk(positive_data_file, negative_data_file)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)

    return [x, y, vocabulary, vocabulary_inv]


def convert(fname, pages=None):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)

    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    infile = open(fname, 'rb')
    try:
        for page in PDFPage.get_pages(infile, pagenums):
            interpreter.process_page(page)
    except:
        infile.close()
        raise ValueError('cannot convert pdf to text')
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close
    return text

# converts all pdfs in directory pdfDir, saves all resulting txt files to txtdir
def convertMultiple(pdfDir, txtDir):
    if pdfDir == "": pdfDir = os.getcwd() + "\\"  # if no pdfDir passed in
    for pdf in os.listdir(pdfDir):  # iterate through pdfs in pdf directory

        fileExtension = pdf.split(".")[-1]
        if fileExtension == "pdf":
            pdfFilename = os.path.join(pdfDir, pdf)
            print("Pre:  ", pdfFilename)
            text = convert(pdfFilename)  # get string of text content of pdf

            pdf = pdf[:-4] + ".txt"
            textFilename = os.path.join(txtDir, pdf)
            print("Post:  ", textFilename)

            textFile = open(textFilename, "wb")  # make text file
            text = text.encode('utf-8', 'ignore')
            # text=text.replace(chr(272)," ")

            textFile.write(text)
            textFile.close()
        # textFile.write(text) #write text to text file
