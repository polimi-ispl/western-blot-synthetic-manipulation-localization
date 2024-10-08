# Libraries import #
import csv
from datetime import datetime
from pprint import pprint
from torch import nn as nn


def make_train_tag(net_class: nn.Module,
                   lr: float,
                   aug: bool,
                   aug_p: float,
                   patch_size: int,
                   patch_number: int,
                   batch_size: int,
                   num_classes: int,
                   ):
    # Training parameters and tag
    tag_params = dict(net=net_class.__name__,
                      lr=lr,
                      aug=aug,
                      aug_p=aug_p,
                      patch_size=patch_size,
                      patch_number=patch_number,
                      batch_size=batch_size
                      )
    print('Parameters')
    pprint(tag_params)
    tag = ''
    tag += '_'.join(['-'.join([key, str(tag_params[key])]) for key in tag_params])
    print('Tag: {:s}'.format(tag))
    return tag


def store_results(train_tag, train_acc, val_acc):
    with open('results.csv', 'a', newline='') as csvfile:
        results = csv.writer(csvfile)
        results.writerow([datetime.now(), train_tag, train_acc, val_acc])
    csvfile.close()
    print('Results stored successfully')
    return
