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

def calculate_sparse():
    idgive = idgiver()
    db = client.libs
    curs = db.js6.find()
    for entry in curs:
        p = entry['project']
        d = entry['depenendent']
        repo_pro[p]= repo_pro.get(p, 0) + 1
        repo_dep[d]= repo_dep.get(d, 0) + 1
    curs = db.js6.find()

    for entry in curs:
        if repo_pro[entry['project']] >= 20 and repo_dep[entry['depenendent']] >= 20:
            p = entry['project']
            d = entry['depenendent']
            idgive.repo_pro2[p] = idgive.repo_pro2.get(p, 0) + 1
            idgive.repo_dep2[d] = idgive.repo_dep2.get(d, 0) + 1

    curs = db.js6.find()
    edges_list = list()
    for entry in curs:

        if idgive.repo_pro2.get(entry['project'],0)>=20 and idgive.repo_dep2.get(entry['depenendent'],0)>=20:
            p = idgive.get_id(entry['project'])
            d = idgive.get_id(entry['depenendent'])
            edges_list.append((p,d))
            edges_list.append((d,p))
    print(idgive.iterr)


    s = [k for k in sorted(match_id, key=match_id.get, reverse=False)]
    f = open('/home/ubuntu/projects/tf.log/github.tsv', 'w')
    f.write('\n'.join(s))
    f.close()
    # col = [ k for (k,_) in edges_list]
    # row = [ w for (_,w) in edges_list]
    # data = len(col) * [1]
    # sha_pe = idgive.iterr
    # logger.info('edges: {0}'.format(len(edges_list)))
    # sp_m = sparse.coo_matrix((data, (row, col)), shape=(sha_pe, sha_pe)).tocsr()
    logger.info('sparse matrix is done')
    # sparse.save_npz('/home/ubuntu/projects/obj/sparse_python', sp_m)
    return None



# sp_m = sparse.load_npz('/home/ubuntu/projects/obj/sparse4.npz')
# iterator = tqdm()

def save_obj(obj, name ):
    with open('/home/ubuntu/projects/obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('/home/ubuntu/projects/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

class randomWalks:
    def __init__(self, sp_m, p, q):
        self.sp_m = sp_m
        self.p = p
        self.q = q
        self.random_walks = list()
        self.time_dict = manager.dict()
        self.count = Value('i', 0)
        self.total_entries = 0


    # def append_dict(self, t,v, entrye):
        # self.transition[t,v] = entrye


def update_dict(iterator, queue, lock):
    sd = time.time()
    transition = dict()
    internal_count = 0
    do_work = False if iterator!=0 else True

    while (rW.count.value != rW.total_entries):
        if queue.qsize()<10 and iterator!=0:
            sleep(1)
            continue
        try:
            trans = queue.get_nowait()
            # logger.info(trans)

            transition[trans[0]] = trans[1]
            with lock:
                rW.count.value += 1
            internal_count +=1
            # queue.task_done()
            array_len = len(transition)
            if array_len%10000==0 and array_len>0:
                logger.debug('Proc: {0}, Speed: {1} it/s, total: {2}'.format(iterator, int(10000/(time.time()-sd)), internal_count))
                sd = time.time()
                save_obj(transition, 'translation/test_{0}_{1}'.format(iterator, internal_count))
                # del rW.transition
                del transition
                transition = dict()

        except Empty:
            # logger.info('empty')
            pass
    logger.info('GetProc:{0} total: {1}.'.format(iterator, internal_count))
    save_obj(transition, 'translation/test_{0}_last'.format(iterator))
    return

def frequency_count(queue):

    while True:
        iterator, proc, timer = queue.get()
        if iterator%FREQUENCY != 0:
            iterator = 'last'
        if not rW.time_dict.get(iterator):
            rW.time_dict[iterator] = list()
        rW.time_dict[iterator].append({proc: timer})
        mylen = len(rW.time_dict[iterator])
        if mylen==CPU_COUNT:
            logger.debug('The overall frequency is {}'.format(np.average( list(rW.time_dict[iterator].values()))))

def compute_transition_prob(iterator, queue, interval, adj_mat_csr_sparse, p, q):

    mycount = 0
    logger.debug('Proc {0} started with interval: {1}'.format(iterator, interval))
    translation = dict()
    begin = time.time()
    num_nodes=adj_mat_csr_sparse.shape[0]
    indices=adj_mat_csr_sparse.indices
    indptr=adj_mat_csr_sparse.indptr
    data=adj_mat_csr_sparse.data
    #Precompute the transition matrix in advance
    # cbeg = int((kk/cpu_count)*num_nodes)
    # cend = int(((kk+1)/cpu_count)*num_nodes)
    beg = time.time()
    for t in range(interval[0], interval[1]):#t is row index
        # if (t-cbeg)%10==0:
            # logger.info('iteration: {0} Process {1}'.format(t-cbeg, kk))
        for v in indices[indptr[t]:indptr[t+1]]:#i.e  possible next ndoes from t
            pi_vx_indices=indices[indptr[v]:indptr[v+1]]#i.e  possible next ndoes from v
            pi_vx_values = np.array([alpha(p,q,t,x,adj_mat_csr_sparse) for x in pi_vx_indices])
            pi_vx_values=pi_vx_values*data[indptr[v]:indptr[v+1]]

            pi_vx_values=pi_vx_values/np.sum(pi_vx_values)
            # now, we have normalzied transion probabilities for v traversed from t
            # the probabilities are stored as a sparse vector.
            translation[(t,v)] = (pi_vx_indices, pi_vx_values)
            mycount += 1
            if mycount % FREQUENCY==0:
                save_obj(translation, 'translation/proc_{0}_{1}'.format(iterator, mycount))
                logger.debug('Proc {0}, completed {1} in {2} sec. Speed: {3}'.format(iterator, mycount, time.time() - beg, int(FREQUENCY / (time.time() - beg))))
                queue.put((mycount, iterator, time.time()-beg))
                del translation
                translation = dict()
                beg=time.time()
            # put_suc = True
            # while put_suc:
            #     try:
            #         mycount+=1
            #         queue.put_nowait(((t,v),  (pi_vx_indices,pi_vx_values)))
            #         put_suc = False
            #     except Full:
            #         put_suc = True
            #         logger.info('queue is full')
            # rW.append_dict(t,v, (pi_vx_indices,pi_vx_values))
    save_obj(translation, 'translation/proc_{0}_{1}'.format(iterator, mycount))
    logger.info('Proc: {0}. End in {1} min. Count {2} !!!!!!'.format(iterator, int((time.time()-begin)/60), mycount))
    return


def alpha(p,q,t,x,adj_mat_csr_sparse):
    if t==x:
        return 1.0/p
    elif adj_mat_csr_sparse[t,x]>0:
        return 1.0
    else:
        return 1.0/q

def calculate_intervals(adj_mat_csr_sparse):
    num_nodes=adj_mat_csr_sparse.shape[0]
    indices=adj_mat_csr_sparse.indices
    indptr=adj_mat_csr_sparse.indptr
    total = 0
    for t in range(num_nodes):
        for v in indices[indptr[t]:indptr[t+1]]:
            total+=len(indices[indptr[v]:indptr[v+1]])
    logger.debug('total: {0}'.format(total))
    entry_per_core = total / CPU_COUNT
    rW.total_entries = total
    logger.info(total)
    intervals = []
    temp = 0
    for t in range(num_nodes):
        for v in indices[indptr[t]:indptr[t+1]]:
            temp+=len(indices[indptr[v]:indptr[v+1]])
            if temp>int(entry_per_core):
                # logger.info(entry_per_core)
                if len(intervals)>0:
                    intervals.append((intervals[-1][1], t))
                else:
                    intervals.append((0, t))
                entry_per_core += total / CPU_COUNT
    intervals.append((intervals[-1][1], num_nodes))
    logger.info(intervals)
    return intervals

if __name__ == '__main__':
    calculate_sparse()
    exit(0)
    p = 1.0
    q = 0.5
    idgive = idgiver()
    lock = Lock()
    sp_m = sparse.load_npz('/home/ubuntu/projects/obj/sparse_python.npz')
    rW = randomWalks(sp_m, p, q)
    intervals = calculate_intervals(sp_m)
    logger.info('shape: {0}'.format(sp_m.shape[0]))

    queue = Queue()

    reader_list = []
    proc_list = []
    # for entry in range(GET_PROC):
    #     q_proc = Process(target=update_dict, args=(entry, queue,lock))
    #     q_proc.start()
    #     reader_list.append(q_proc)

    import time
    sd = time.time()
    q_proc = Process(target=frequency_count, args=(queue,))
    q_proc.start()
    for entry in range(CPU_COUNT):
        new_proc = Process(target=compute_transition_prob, args=(entry, queue, intervals[entry], sp_m, p, q))
        proc_list.append(new_proc)
        new_proc.start()
    for entry in proc_list:
        entry.join()

    # for entry in reader_list:
    #     entry.join()
    logger.info('time: {0} min'.format(int((time.time()-sd)/60.0)))
    # logger.info('len of last: {0}'.format(len(rW.transition)))
    logger.info('count: {0}'.format(rW.count.value))
    logger.info('average speed: {0} it/s'.format(int(rW.count.value/(time.time()-sd))))
    # queue.join()
    q_proc.terminate()
    logger.debug('Timings:')
    logger.debug(rW.time_dict)
    # while not queue.empty():
    #     item, value = queue.get()
    #     transition[item] = value
    #
    # transition = dict()
    # for x,y in rW.transition.items():
    #     transition[x] = y
    # save_obj(transition, 'translation/test_{0}'.format(rW.count.value))