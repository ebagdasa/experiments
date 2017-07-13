from pymongo import MongoClient
import numpy as np
client = MongoClient()
db = client.libs

class IdGiver:
    def __init__(self):
        self.iterr = 0
        self.match_id = dict()
        self.repo_pro = dict()
        self.repo_dep = dict()
        self.final_repo_pro = dict()
        self.final_repo_dep = dict()

    def get_id(self, name):

        if not self.match_id.get(name, False):
            self.match_id[name] = self.iterr
            self.iterr += 1
        return self.match_id[name]



def parse_edges(idgive):

    edges_list = list()

    cursor = db.js6.find()
    for entry in cursor:
        p = entry['project']
        d = entry['depenendent']
        idgive.repo_pro[p]= idgive.repo_pro.get(p, 0) + 1
        idgive.repo_dep[d]= idgive.repo_dep.get(d, 0) + 1

    # come up with edge list:
    cursor = db.js6.find()
    for entry in cursor:
        if idgive.repo_pro.get(entry['project'],0)>=20 and \
                        idgive.repo_dep.get(entry['depenendent'],0)>=20:

            p = idgive.get_id(entry['project'])
            d = idgive.get_id(entry['depenendent'])

            #gather statistics
            idgive.final_repo_pro[p] = idgive.final_repo_pro.get(p, 0) + 1
            idgive.final_repo_dep[d] = idgive.final_repo_dep.get(d, 0) + 1

            edges_list.append('{0} {1}'.format(p,d))

    print('# of nodes {0}. \n # of edges: {1}  \n project average {2}. \n Dependent average: {3}. '.format(
                                        len(idgive.match_id), len(edges_list),
                                        np.average(list(idgive.final_repo_pro.values())),
                                        np.average(list(idgive.final_repo_dep.values()))))

    #write edges
    with  open('/home/ubuntu/projects/obj/edges.list', 'w') as f:
        f.write('\n'.join(edges_list))

    with  open('/home/ubuntu/projects/obj/nodes.tsv', 'w') as f:
        match = idgive.match_id
        list_of_names = [k for k in sorted(match, key=match.get, reverse=False)]
        f.write('\n'.join(list_of_names))




if __name__ == '__main__':
    print('convert edges to mtx')
    idgive = IdGiver()
    parse_edges(idgive)
