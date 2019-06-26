# -*- coding: utf-8 -*-
# @Time    : 2019/5/14 11:40
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : test.py
# @Software: PyCharm

from CCMT.sohu import tokenization
import os
import csv
import tensorflow as tf


def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def get_train_examples(data_dir):
    lines = read_tsv(os.path.join(data_dir))
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "train-%d" % (i)
        text_a = tokenization.convert_to_unicode(line[0])
        print("line[0]:"+line[0])
        text_b = tokenization.convert_to_unicode(line[1])
        print("line[1]:"+line[1])
        label = tokenization.convert_to_unicode(line[2])
        print("line[2]:"+line[2])


get_train_examples(data_dir="E:\\chinese_L-12_H-768_A-12_fine_tune\\MRPC\\train.tsv")


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples