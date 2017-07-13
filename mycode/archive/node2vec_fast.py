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
rW = []

match_id = dict()
repo_pro = dict()
repo_dep = dict()

class idgiver:
    def __init__(self):
        self.iterr = 0
        self.repo_pro2 = dict()
        self.repo_dep2 = dict()

    def get_id(self, g_id):
        if not match_id.get(g_id, False):
            match_id[g_id] = self.iterr
            self.iterr += 1
        return match_id[g_id]


idgive = idgiver()
db = client.libs
curs = db.js6.find()
for entry in curs:
    p = entry['project']
    d = entry['depenendent']
    repo_pro[p]= repo_pro.get(p, 0) + 1
    repo_dep[d]= repo_dep.get(d, 0) + 1
curs = db.js6.find()

# for entry in curs:
#     if repo_pro[entry['project']] >= 20 and repo_dep[entry['depenendent']] >= 20:
#         p = entry['project']
#         d = entry['depenendent']
#         idgive.repo_pro2[p] = idgive.repo_pro2.get(p, 0) + 1
#         idgive.repo_dep2[d] = idgive.repo_dep2.get(d, 0) + 1

# curs = db.js6.find()
edges_list = list()
for entry in curs:
    if repo_pro.get(entry['project'],0)>=10 and repo_dep.get(entry['depenendent'],0)>=10:
        p = idgive.get_id(entry['project'])
        d = idgive.get_id(entry['depenendent'])
        edges_list.append('{0} {1}'.format(p,d))
        # edges_list.append((d,p))
print(idgive.iterr)
print(len(edges_list))


s = [k for k in sorted(match_id, key=match_id.get, reverse=False)]
f = open('/home/ubuntu/projects/tf.log/github.tsv', 'w')
f.write('\n'.join(s))
f.close()
f = open('/home/ubuntu/projects/obj/edges.list', 'w')
f.write('\n'.join(edges_list))
f.close()

    # f.writ