from __future__ import division
from __future__ import print_function
import numpy as np
from datetime import datetime
import sys


def onehot_encoding(label, max_num_class):
    encoding = np.eye(max_num_class)
    encoding = encoding[label]
    return encoding


def calculate_acc(output, label):
    correct = np.sum(np.argmax(output, axis=1) == label)
    return correct / len(label) * 100


def LOG_INFO(msg):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-3]
    sys.stderr.write(display_now + ' ' + msg + '\n')
