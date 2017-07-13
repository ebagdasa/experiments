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

translation = manager.dict()
cpu_count = 16

def save_obj(obj, name ):
    with open('/home/ubuntu/projects/obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('/home/ubuntu/projects/obj/' + name + ('.pkl' if not '.pkl' in name else ''), 'rb') as f:
        return pickle.load(f)

def read_dicts(file_list):
    fl = file_list #[x for x in file_list if 'proc_{0}_'.format(iterator) in x]
    # translation = dict()
    for iterator,x in enumerate(fl):
        logger.debug(iterator)
        new_dict = load_obj('translation/{0}'.format(x))
        translation.update(new_dict)
    logger.debug(len(translation))
    return translation

def generate_random_walks(iterator, queue, adj_mat_csr_sparse,random_walk_length):
    start = time.time()
    num_nodes=adj_mat_csr_sparse.shape[0]
    indices=adj_mat_csr_sparse.indices
    indptr=adj_mat_csr_sparse.indptr
    data=adj_mat_csr_sparse.data
    random_walks = list()
    counter = 0
    beg = time.time()
    #get random walks
    for u in range(num_nodes):
        if len(indices[indptr[u]:indptr[u+1]]) !=0:
            #first move is just depends on weight
            possible_next_node=indices[indptr[u]:indptr[u+1]]
            weight_for_next_move=data[indptr[u]:indptr[u+1]]#i.e  possible next ndoes from u
            weight_for_next_move=weight_for_next_move.astype(np.float32)/np.sum(weight_for_next_move)
            first_walk=np.random.choice(possible_next_node, 1, p=weight_for_next_move)
            random_walk=[u,first_walk[0]]
            for i in range(random_walk_length-2):
                cur_node = random_walk[-1]
                precious_node=random_walk[-2]
                (pi_vx_indices,pi_vx_values)=translation[precious_node,cur_node]
                next_node=np.random.choice(pi_vx_indices, 1, p=pi_vx_values)
                random_walk.append(next_node[0])
            counter += 1
            random_walks.append(random_walk)
            if counter%FREQUENCY==0:
                save_obj(random_walks, 'randwalks/proc_{0}_{1}'.format(iterator, counter))

                logger.debug('Saved proc_{0}_{1}  in {2} sec'.format(iterator, counter, time.time() - beg))
                beg = time.time()
                del random_walks
                random_walks = list()
    save_obj(random_walks, 'randwalks/proc_{0}_{1}'.format(iterator, counter))
    return



if __name__ == '__main__':
    logger.debug('QWERQWER')
    files = os.listdir('/home/ubuntu/projects/obj/translation/')
    sp_m = sparse.load_npz('/home/ubuntu/projects/obj/sparse_python.npz')
    beg = time.time()
    translation = read_dicts(file_list=files)
    logger.debug('read time: {0} sec'.format(time.time()-beg))
    logger.debug(len(translation))
    proc_list = list()
    beg = time.time()
    for entry in range(CPU_COUNT):
        new_proc = Process(target=generate_random_walks, args=(entry, None, sp_m, 100))
        proc_list.append(new_proc)
        new_proc.start()
    for entry in proc_list:
        entry.join()
    logger.debug('processing time time: {0} sec'.format(time.time() - beg))

