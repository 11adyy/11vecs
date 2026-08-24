"""Node insertion for HNSW."""

import math
import random

from .node import Node
from .search import greedy_range, beam_range


def random_level(hnsw):
    if hnsw.m <= 1:
        raise ValueError(
            "m must be greater than 1"
        )

    probability = random.random()

    while probability == 0:
        probability = random.random()

    return int(
        -math.log(probability) / math.log(hnsw.m)
    )


def add(
    hnsw,
    content,
    metadata,
    vector,
):
    node_level = random_level(hnsw)

    new_node = Node(
        content=content,
        metadata=metadata,
        vector=vector,
        level=node_level,
    )


    if hnsw._entry_point is None:
        node_id = 0

        hnsw._nodes[node_id] = new_node

        hnsw._entry_point = node_id
        hnsw._max_level = node_level

        return node_id

    if node_level < hnsw._max_level:
        entry_point_id = greedy_range(
            hnsw=hnsw,
            query=vector,
            entry_point_id=hnsw._entry_point,
            start_level=hnsw._max_level,
            end_level=node_level + 1,
        )
    else:
        entry_point_id = hnsw._entry_point

    search_start_level = min(
        node_level,
        hnsw._max_level,
    )

    connections = beam_range(
        hnsw=hnsw,
        query=vector,
        entry_point_id=entry_point_id,
        start_level=search_start_level,
        end_level=0,
        beam_width=hnsw.ef_construction,
        top_k=hnsw.m,
    )

    for level, neighbor_ids in enumerate(
        connections
    ):
        new_node.neighbors[level] = neighbor_ids

    node_id = len(hnsw._nodes)

    hnsw._nodes[node_id] = new_node

    for level, neighbor_ids in enumerate(
        connections
    ):
        for neighbor_id in neighbor_ids:
            hnsw._nodes[
                neighbor_id
            ].neighbors[level].append(node_id)

    if node_level > hnsw._max_level:
        hnsw._entry_point = node_id
        hnsw._max_level = node_level

    return node_id