from mpi4py import MPI
import csv
import os

def read_nodes(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  
        return [int(row[0]) for row in reader]

def read_edges(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        edges = []
        for row in reader:
            edge_str = row[0]
            u, v = edge_str.strip("()").split(", ")
            edges.append((int(u), int(v)))
        return edges

def find_triangles(nodes, edges):
    triangles = []
    for node in nodes:
        for edge in edges:
            u, v = edge
            if node != u and node != v:
                if (u, node) in edges or (node, u) in edges:
                    if (v, node) in edges or (node, v) in edges:
                        triangle = tuple(sorted([u, v, node]))
                        if triangle not in triangles:
                            triangles.append(triangle)
    return triangles

def write_triangles(rank, triangles, output_dir):
    output_path = "{}/rank_{}.csv".format(output_dir, rank)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path, 'w') as file:
        writer = csv.writer(file)
        for triangle in triangles:
            writer.writerow(triangle)

def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    nodes_file_path = "/gpfs/projects/AMS598/class2023/final_projects/group3/tri_count/nodes.csv"
    edges_file_path = "/gpfs/projects/AMS598/class2023/final_projects/group3/tri_count/edge_index.csv"
    temp_output_dir = "/gpfs/projects/AMS598/class2023/final_projects/group3/tri_count/temp_data"
    final_output_path = "/gpfs/projects/AMS598/class2023/final_projects/group3/tri_count/triangle"

    nodes = read_nodes(nodes_file_path)
    edges = read_edges(edges_file_path)

    chunk_size = len(nodes) // size
    start_index = rank * chunk_size
    end_index = start_index + chunk_size if rank != size - 1 else len(nodes)
    nodes_for_rank = nodes[start_index:end_index]

    triangles = find_triangles(nodes_for_rank, edges)
    write_triangles(rank, triangles, temp_output_dir)

    if rank == 0:
        final_triangles = []
        for i in range(size):
            temp_path = "{}/rank_{}.csv".format(temp_output_dir, i)
            with open(temp_path, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    final_triangles.append(tuple(map(int, row)))

        if not os.path.exists(os.path.dirname(final_output_path)):
            os.makedirs(os.path.dirname(final_output_path))
        with open(final_output_path, 'w') as file:
            writer = csv.writer(file)
            for triangle in final_triangles:
                writer.writerow(triangle)

        print("Total number of triangles found:", len(final_triangles))

if __name__ == "__main__":
    main()

