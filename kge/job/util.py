import torch
from torch import Tensor
import numpy as np
from typing import List, Union
from random import sample


def get_sp_po_coords_from_spo_batch(
    batch: Union[Tensor, List[Tensor]], num_entities: int, sp_index: dict, po_index: dict
) -> torch.Tensor:
    """Given a set of triples , lookup matches for (s,p,?) and (?,p,o).

    Each row in batch holds an (s,p,o) triple. Returns the non-zero coordinates
    of a 2-way binary tensor with one row per triple and 2*num_entites columns.
    The first half of the columns correspond to hits for (s,p,?); the second
    half for (?,p,o).

    """
    if type(batch) is list:
        batch = torch.cat(batch).reshape((-1, 3)).int()
    sp_coords = sp_index.get_all(batch[:, [0, 1]])
    po_coords = po_index.get_all(batch[:, [1, 2]])
    po_coords[:, 1] += num_entities
    coords = torch.cat(
        (
            sp_coords,
            po_coords
        )
    )

    return coords


def coord_to_sparse_tensor(
    nrows: int, ncols: int, coords: Tensor, device: str, value=1.0, row_slice=None
):
    if row_slice is not None:
        if row_slice.step is not None:
            # just to be sure
            raise ValueError()

        coords = coords[
            (coords[:, 0] >= row_slice.start) & (coords[:, 0] < row_slice.stop), :
        ]
        coords[:, 0] -= row_slice.start
        nrows = row_slice.stop - row_slice.start

    if device == "cpu":
        labels = torch.sparse.FloatTensor(
            coords.long().t(),
            torch.ones([len(coords)], dtype=torch.float, device=device) * value,
            torch.Size([nrows, ncols]),
        )
    else:
        labels = torch.cuda.sparse.FloatTensor(
            coords.long().t(),
            torch.ones([len(coords)], dtype=torch.float, device=device) * value,
            torch.Size([nrows, ncols]),
            device=device,
        )

    return labels

def sample_uniform(triples, sample_size, num_entities=None):
    return sample(triples, sample_size)

def sample_edge_neighbourhood(triples, sample_size, num_entities):
    # create list of all adjacent edges-neighbour pairs for all entities
    adjacencies_per_entity = [[] for _ in range(num_entities)]
    for edge_number, triple in enumerate(triples):
        adjacencies_per_entity[triple[0]].append([edge_number, triple[2]])
        adjacencies_per_entity[triple[2]].append([edge_number, triple[0]])

    # calculate degree per entity
    degrees = np.array([len(adjs) for adjs in adjacencies_per_entity])
    adjacencies_per_entity = [np.array(adjs) for adjs in adjacencies_per_entity]

    # create empty edge list where chosen edges are collected in sampling
    edges = np.zeros((sample_size), dtype=np.int64)

    # degree list, when an edge is chosen the edge is subtracted from the two nodes
    sample_counts = np.array([d for d in degrees])
    # keeps track of picked edges
    picked = np.array([False for _ in triples])
    # keeps track of seen entities, only if seen it can be picked
    seen = np.array([False for _ in degrees])

    # iteratively choose and reweight edges
    for i in range(0, sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts==0)] = 0
        
        # sample entity
        probabilities = weights / weights.sum()
        chosen_entity = np.random.choice(num_entities, p=probabilities)
        chosen_adjacencies = adjacencies_per_entity[chosen_entity]
        seen[chosen_entity] = True

        # sample one adjacent edge from entity
        chosen_edge = np.random.choice((degrees[chosen_entity]))
        chosen_edge = chosen_adjacencies[chosen_edge]
        edge_number = chosen_edge[0]

        # continue picking edges until one unseen is found
        while picked[edge_number]:
            chosen_edge = np.random.choice((degrees[chosen_entity]))
            chosen_edge = chosen_adjacencies[chosen_edge]
            edge_number = chosen_edge[0]

        # add edge to picked edges
        edges[i] = edge_number
        # identify other entity in the edge
        neighbour_entity = chosen_edge[1]
        # mark edge as picked
        picked[edge_number] = True
        # reduce sample counts of picked entities
        sample_counts[chosen_entity] -= 1
        sample_counts[neighbour_entity] -= 1
        # mark neighbour as seen, it can now be picked as well if previously unseen
        seen[neighbour_entity] = True
    
    # select the sampled edges from the triples
    edge_index = Tensor(edges).to(int)
    edges = torch.index_select(triples, 0, edge_index)

    return edges

def calculate_edge_index(triples, device, inverse=True):
    edge_index = []
    for sub, _, obj in triples:
        edge_index.append((sub.item(), obj.item()))
    # Add inverse edges
    if inverse:
        inverse_edges = [tuple(reversed(edge)) for edge in edge_index]
        edge_index = edge_index + inverse_edges
    edge_index = torch.LongTensor(edge_index).t().to(device) 
    return edge_index

def calculate_edge_type(triples, device, num_relations, inverse=True):
    edge_type = []
    for _, rel, _ in triples:
        edge_type.append(rel.item())
    # Add inverse relation types to edges
    if inverse:
        inverse_types = np.array(edge_type) + num_relations
        edge_type = edge_type + list(inverse_types)
    edge_type = torch.LongTensor(edge_type).t().to(device) 
    return edge_type

            


