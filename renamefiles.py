import tensorflow as tf
import argparse
import utils
import os
import shutil
import time
import utils
import re
import random
from collections import Counter

def rename_files(args):
    files_list = os.listdir(args.dir)
    for f in files_list:
        os.rename(os.path.join(args.dir, f), os.path.join(args.dir, "1_"+f))

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, default='/Downloads',
                        help='directory to Downloads folder')
    args = parser.parse_args()
    rename_files(args)

if __name__ == '__main__':
    main()
