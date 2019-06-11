import os
from tqdm import tqdm

DATA_DIR = "data"
MAX_DOCS = 1000


def file_len(fname):
    with open(fname, encoding="utf8") as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def csv_data_to_text(data_dirs):
    pos_cnt = 0
    neg_cnt = 0
    with open(r'pos.txt', 'w') as cs_outfile:
        with open(r'neg.txt', 'w') as other_outfile:
            for root, dirs, files in os.walk(data_dirs):
                for dir in dirs:
                    for txt_file in tqdm(os.listdir('{}/{}'.format(root, dir))):
                        if dir.__contains__('CS'):
                            if txt_file.endswith(".txt") and pos_cnt <= MAX_DOCS:
                                with open('{}/{}/{}'.format(root, dir, txt_file)) as infile:
                                    pos_cnt += 1
                                    try:
                                        for line in infile:
                                            if len(line) > 100:
                                                for sen in line.splitlines():
                                                    cs_outfile.write(sen[:1000] + '\n')
                                    except:
                                        continue
                        else:
                            if txt_file.endswith(".txt") and neg_cnt <= MAX_DOCS:
                                with open('{}/{}/{}'.format(root, dir, txt_file)) as infile:
                                    neg_cnt += 1
                                    try:
                                        for line in infile:
                                            if len(line) > 100:
                                                for sen in line.splitlines():
                                                    other_outfile.write(sen[:1000] + '\n')
                                    except:
                                        continue
    #
    # print('pos len: ', file_len(r'pos.txt'))
    # print('neg len: ', file_len(r'neg.txt'))


if __name__ == '__main__':
    csv_data_to_text(DATA_DIR)
