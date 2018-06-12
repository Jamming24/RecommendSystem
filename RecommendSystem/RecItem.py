# coding=utf-8

import numpy as np
from collections import defaultdict


valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occurances = defaultdict(int)


dataset_filename = "affinity_dataset.txt"
X = np.loadtxt(dataset_filename)
print(X[:5])

num_apple_purchases = 0
for sample in X:
    if sample[3] == 1:
        num_apple_purchases += 1

print("{0} people bought Apples".format(num_apple_purchases))


for sampl in X:
    for premise in range(4):
        if sample[premise] == 0:
            continue
        num_occurances[premise] += 1
        for conclusion in range(n_features):
            if premise == conclusion:
                continue
        if sample[conclusion] == 1:
            valid_rules[(premise, conclusion)] += 1
        else:
            invalid_rules[(premise, conclusion)] += 1
support = valid_rules

confidence = defaultdict(float)
for premise, conclusion in valid_rules.keys():
    rule = (premise, conclusion)
    confidence[rule] = valid_rules[rule] / num_occurances[premise]

