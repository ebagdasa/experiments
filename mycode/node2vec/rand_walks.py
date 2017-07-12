from time import sleep
# from tqdm import tqdm
from multiprocessing import Manager, Process, Queue, JoinableQueue, Value, Lock
from multiprocessing.queues import Empty, Full
import logging
import time


import yaml
import os
import json
import gzip
import zlib
import datetime
from dateutil import parser
import multiprocessing
from subprocess import run, PIPE
import os
from pymongo import MongoClient
import pickle
import numpy as np
import scipy.sparse as sparse
client = MongoClient()

import multiprocessing as mp
# GET_PROC = 10
CPU_COUNT = mp.cpu_count()
FREQUENCY = 1000
#logger
logger = logging.getLogger('MyLogger')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('my.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

manager = Manager()
cpu_count = 16

def save_obj(obj, name ):
    with open('/home/ubuntu/projects/obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('/home/ubuntu/projects/obj/' + name + ('.pkl' if not '.pkl' in name else ''), 'rb') as f:
        return pickle.load(f)

def read_dicts(iterator, file_list):
    fl = [x for x in file_list if 'proc_{0}_'.format(iterator) in x]
    translation = dict()
    for x in fl:
        new_dict = load_obj('translation/{0}'.format(x))
        translation.update(new_dict)
    logger.debug(len(translation))
    return translation


if __name__ == '__main__':
    files = os.listdir('/home/ubuntu/projects/obj/translation')
    translation = read_dicts(1,file_list=files)
    logger.debug(len(translation))
