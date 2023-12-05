import time
start_time = time.time()
import mpi4py
from mpi4py import MPI
import glob
import os
import random
from collections import defaultdict ,Counter
import json
import numpy as np
random.seed(69)
import argparse

def parseOptions(comm):
    parser = argparse.ArgumentParser(
        description='Print some messages.')

    parser.add_argument('nodes', help='Number of nodes', type=int)

    args = None
    try:
        if comm.Get_rank() == 0:
            args = parser.parse_args()
    finally:
        args = comm.bcast(args, root=0)

    if args is None:
        exit(0)
    return args
def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parseOptions(comm)

    if rank==0:
        def generate_adjacency_matrix(n):
            # Define the large number for no edge
            NO_EDGE = 10000000000000

            # Initialize an nxn matrix with NO_EDGE value
            adj_matrix = np.full((n, n), NO_EDGE)

            # Iterate over the matrix
            for i in range(n):
                for j in range(n):
                    if i == j:
                        # No self-loops, distance to self is 0
                        adj_matrix[i, j] = 0
                    else:
                        # Randomly decide to add an edge with a weight 1, 2, or 3
                        if random.choice([True, False]):
                            adj_matrix[i, j] = random.randint(1, 10)

            return np.matrix(adj_matrix)

        # Example usage
        n = args.nodes  # Size of the matrix
        adj_matrix = generate_adjacency_matrix(n)
        num_columns=adj_matrix.shape[1]
        scatter_send=[[] for _ in range(size)]
        for i in range(num_columns):
            scatter_send[i%size].append({i:adj_matrix[:,i]})
    else:
        scatter_send=None
    curr_columns=comm.scatter(scatter_send,root=0)
    start_time = time.time()
    source=0
    marked_nodes=set([source])
    if rank==0:
        all_final_distances=[(source,0)]
    for iteration in range(args.nodes):
        #GET CURRENT MINIMUM ESTIMATE, distance from source.
        id_to_vals=[]
        for column in curr_columns:
            assert len(column)==1
            k,v=list(column.items())[0]
            if k in marked_nodes:
                continue
            id_to_vals.append((k,v[source].item()))
        # estimate will be min of id_to_vals
        if len(id_to_vals)>0:
            local_min=sorted(id_to_vals,key=lambda x:x[1])[0]
        else:
            local_min=(-1,10000000000000)
        local_min_all=comm.gather(local_min,root=0)
        to_break=False
        if rank==0:
            global_minimum_id,global_minimum_distance=sorted(local_min_all,key=lambda x:x[1])[0]
            if global_minimum_distance==10000000000000:
                to_break=True
            else:
                all_final_distances.append((global_minimum_id,global_minimum_distance))
        else:
            global_minimum_id,global_minimum_distance=None,None
        # get global min id and distance
        global_minimum_id,global_minimum_distance,to_break=comm.bcast([global_minimum_id,global_minimum_distance,to_break],root=0)
        if to_break:
            break
        marked_nodes.add(global_minimum_id)
        # update all paths from global min id if new value less than previous
        for i,column in enumerate(curr_columns):
            assert len(column)==1
            k,v=list(column.items())[0]
            if k in marked_nodes:
                continue
            curr_dist=v[source].item()
            dist_from_global=v[global_minimum_id].item()
            if curr_dist>(dist_from_global+global_minimum_distance):
                v[source]=dist_from_global+global_minimum_distance
                curr_columns[i]={k:v}
    if rank==0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print "Time elapsed: {}".format(elapsed_time)
    MPI.Finalize()

if __name__=='__main__':
    main()
