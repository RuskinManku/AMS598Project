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

def find_bucket_index(num_buckets, k):
    return k%num_buckets

def gather_and_broadcast_maps(comm,rank,local_map):
    """
    Gathers maps from all MPI processes, merges them at the root, and then broadcasts the merged map.
    Assumes that the keys in the map are unique across all processes.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Gather maps from all processes to the root process
    all_maps = comm.gather(local_map, root=0)

    # If this is the root process, merge the maps
    if rank == 0:
        merged_map = {}
        for m in all_maps:
            merged_map.update(m)
    else:
        merged_map = None

    return merged_map

def mapper_initiation(comm,rank,size,files_dir):
    # Split data files to be processed by each process randomly
    if rank==0:
        all_files=glob.glob(files_dir+'*')
        print all_files
        #Split all_files into an array of size "size"=number of processes for scatter to work
        chunked_files=[[] for _ in range(size)]
        for i,file in enumerate(all_files):
            chunked_files[i%size].append(file)
    else:
        chunked_files=None

    current_files_to_process=comm.scatter(chunked_files,root=0)

    # Calculate frequency within each process
    node_counts=Counter()
    data=[]
    for filename in current_files_to_process:
        with open(filename, 'r') as file:
            lines = file.readlines()
        for line in lines:
            try:
                from_node,to_node = line.split('\t')
                from_node=int(from_node)
                to_node=int(to_node)
                node_counts[from_node]+=1
                node_counts[to_node]+=1
		data.append((from_node,to_node))
            except ValueError:
                print "Warning: Invalid integer found in file {}".format(filename)
    return node_counts,data

def reducer_initiation(comm,rank,size,received_elements):
    node_counts_total=Counter()
    for (node_id,cnt) in received_elements:
        node_counts_total[node_id]+=cnt
    return node_counts_total 

def mapper_induction(comm,rank,size,current_data,current_community):
    output_list=[]
    for line in current_data:
        try:
            from_node,to_node = line
            from_node=int(from_node)
            to_node=int(to_node)
            if from_node in current_community and to_node not in current_community:
                output_list.append((to_node,1))
            elif from_node not in current_community and to_node in current_community:
                output_list.append((from_node,1))
            elif from_node not in current_community and to_node not in current_community:
                output_list.append((from_node,-1))
                output_list.append((to_node,-1))
        except ValueError:
            print ("Warning: Invalid integer found in file {}".format(filename))
    return output_list

def reducer_induction(comm,rank,size,received_elements,in_degree,out_degree):
    local_degree_increment=defaultdict(lambda: [0,0])
    for ele in received_elements:
        k,v=ele
        if v==-1:
            local_degree_increment[k][1]+=1
        elif v==1:
            local_degree_increment[k][0]+=1
            local_degree_increment[k][1]-=1
    m_star_map={}
    for k,v in local_degree_increment.items():
        local_in_degree=in_degree+v[0]
        local_out_degree=out_degree+v[1]
        if local_in_degree==0:
            continue
        m_star_map[k]=(local_in_degree,local_out_degree)
    return m_star_map

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    files_dir="Data_Twitch/"

    #Run Initiation stage mapper and get generate key value pairs from edges to get out degree of each node
    node_counts,current_data=mapper_initiation(comm,rank,size,files_dir)

    #Coalate data by keys
    comm.Barrier()
    to_send = [[] for _ in range(size)]
    for k,v in node_counts.items():
        k=int(k)
        dest = find_bucket_index(size,k)
        to_send[dest].append((k,v))

    comm.Barrier()
    received_elements = [[] for _ in range(size)]
    for i in range(size):
        received_data = comm.gather(to_send[i], root=i)
        if rank == i:
            received_elements = [item for sublist in received_data for item in sublist]
    comm.Barrier()
    
    # Initiation Reducer
    node_counts_total=reducer_initiation(comm,rank,size,received_elements)
    

    # Get global count of all out-degress and broadcast to each rank
    all_node_counts=gather_and_broadcast_maps(comm,rank,node_counts_total)

    # Sort it and select the key with highest out degree for first stage of community creation.
    # START INDUCTION STAGE
    if rank==0:
        nodes_possible=set(all_node_counts.keys())
        all_selected_till_now=set()
    communities=[]
    while 1:
        if rank==0:
            for node in all_selected_till_now:
                if node in nodes_possible:
                    nodes_possible.remove(node)
            if len(nodes_possible)==0:
                selected_node_id=None
                in_degree=None
                out_degree=None
            else:
                selected_node_id=random.choice(list(nodes_possible))
                print "Selecting :{}".format(selected_node_id)
		in_degree=0
                out_degree=all_node_counts[selected_node_id]
        else:
            selected_node_id=None
            in_degree=None
            out_degree=None
        [selected_node_id,in_degree,out_degree]=comm.bcast([selected_node_id,in_degree,out_degree], root=0)
        if selected_node_id==None:
            break
        current_community=set([selected_node_id])
        m_value=0
        while 1:
            output_list=mapper_induction(comm,rank,size,current_data,current_community)
            comm.Barrier()
            # Coalate keys at relevant reducers
            to_send = [[] for _ in range(size)]
            for ele in output_list:
                k=ele[0]
                dest = find_bucket_index(size,k)
                to_send[dest].append(ele)
            comm.Barrier()
            received_elements = [[] for _ in range(size)]
            for i in range(size):
                received_data = comm.gather(to_send[i], root=i)
                if rank == i:
                    received_elements = [item for sublist in received_data for item in sublist]
            m_star_map=reducer_induction(comm,rank,size,received_elements,in_degree,out_degree)
            m_star_map_list=list(m_star_map.items())
            m_star_map_list.sort(key=lambda x:float(x[1][0])/float(x[1][1]),reverse=True)
            if len(m_star_map_list)!=0:
                top_value={m_star_map_list[0][0]:m_star_map_list[0][1]}
            else:
                top_value={}
            global_m_star_map=gather_and_broadcast_maps(comm,rank,top_value)
            comm.Barrier()
            if rank==0:
                # Sort by highest M-value
                list_m_star=list(global_m_star_map.items())
                list_m_star.sort(key=lambda x:float(x[1][0])/float(x[1][1]),reverse=True)
                m_star=float(list_m_star[0][1][0])/float(list_m_star[0][1][1])
                if m_star>m_value:
		    print "Length Increased"
                    m_value=m_star
                    in_degree=list_m_star[0][1][0]
                    out_degree=list_m_star[0][1][1]
                    current_community.add(list_m_star[0][0])
                else:
                    print "breaking, old={}, highest_new={}, length:{}".format(m_value,m_star,len(current_community))
                    in_degree=None
                    out_degree=None
            else:
                in_degree=None
                out_degree=None
                current_community=None
            in_degree=comm.bcast(in_degree,root=0)
            out_degree=comm.bcast(out_degree,root=0)
            current_community=comm.bcast(current_community,root=0)
            if in_degree==None:
                break
        comm.Barrier()
        if rank==0:
            print "COMMUNITY FOUND" 
            for node in current_community:
                all_selected_till_now.add(node)
            communities.append(list(current_community))
    # Finalize MPI
    if rank==0:
        print len(communities)
    MPI.Finalize()

    end_time = time.time()

    elapsed_time = end_time - start_time
    if rank==0:
	print "elapsed_time:{}".format(elapsed_time)

if __name__=='__main__':
    main()
