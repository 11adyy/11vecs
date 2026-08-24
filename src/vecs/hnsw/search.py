"""Greedy and beam search algorithms for HNSW."""

import heapq
from itertools import count

from .node import distance


def greedy(
    hnsw,
    query,
    entry_point_id,
    level,
):
    """
    Greedy search on a single HNSW level.

    Starting from entry_point_id, repeatedly moves to the
    closest neighbor until no neighbor improves the distance.
    """

    current_id = entry_point_id

    while True:
        current_distance = distance(
            query,
            hnsw._nodes[current_id].vector,
        )

        best_id = current_id

        for neighbor_id in hnsw._nodes[
            current_id
        ].neighbors[level]:

            neighbor_distance = distance(
                hnsw._nodes[neighbor_id].vector,
                query,
            )

            if neighbor_distance < current_distance:
                current_distance = neighbor_distance
                best_id = neighbor_id

        if best_id == current_id:
            return current_id

        current_id = best_id


def greedy_range(
    hnsw,
    query,
    entry_point_id,
    start_level,
    end_level,
):
    """
    Greedy search from start_level down to end_level.
    """

    if start_level < 0 or start_level > hnsw._max_level:
        raise ValueError(
            f"level must be between 0 and {hnsw._max_level}, "
            f"got {start_level}"
        )

    for level in range(
        start_level,
        end_level - 1,
        -1,
    ):
        entry_point_id = greedy(
            hnsw=hnsw,
            query=query,
            entry_point_id=entry_point_id,
            level=level,
        )

    return entry_point_id


def beam(
    hnsw,
    query,
    entry_point_id,
    level,
    beam_width,
    top_k,
):
    """
    Beam search on a single HNSW level.

    beam_width:
        Maximum number of candidates kept during exploration.

    top_k:
        Number of the best nodes returned.
    """

    candidates = []
    results = []
    counter = count()

    visited = {entry_point_id}

    entry_distance = distance(
        hnsw._nodes[entry_point_id].vector,
        query,
    )

    heapq.heappush(
        candidates,
        (
            entry_distance,
            next(counter),
            entry_point_id,
        ),
    )

    heapq.heappush(
        results,
        (
            -entry_distance,
            next(counter),
            entry_point_id,
        ),
    )

    while candidates:
        current_distance, _, current_id = (
            heapq.heappop(candidates)
        )

        worst_distance = -results[0][0]

        if (
            len(results) >= beam_width
            and current_distance > worst_distance
        ):
            break

        for neighbor_id in hnsw._nodes[
            current_id
        ].neighbors[level]:

            if neighbor_id in visited:
                continue

            visited.add(neighbor_id)

            neighbor_distance = distance(
                hnsw._nodes[neighbor_id].vector,
                query,
            )

            if (
                len(results) < beam_width
                or neighbor_distance < -results[0][0]
            ):
                heapq.heappush(
                    candidates,
                    (
                        neighbor_distance,
                        next(counter),
                        neighbor_id,
                    ),
                )

                heapq.heappush(
                    results,
                    (
                        -neighbor_distance,
                        next(counter),
                        neighbor_id,
                    ),
                )

                if len(results) > beam_width:
                    heapq.heappop(results)

    ordered_results = sorted(
        results,
        key=lambda item: -item[0],
    )

    return [
        node_id
        for _, _, node_id in ordered_results[:top_k]
    ]


def beam_range(
    hnsw,
    query,
    entry_point_id,
    start_level,
    end_level,
    beam_width,
    top_k,
):
    """
    Beam search from start_level down to end_level.

    Returns the selected nodes for every level.
    """

    connections = []

    if start_level < 0:
        raise ValueError(
            f"start_level cannot be negative, "
            f"got {start_level}"
        )

    for level in range(
        start_level,
        end_level - 1,
        -1,
    ):
        level_connections = beam(
            hnsw=hnsw,
            query=query,
            entry_point_id=entry_point_id,
            level=level,
            beam_width=beam_width,
            top_k=top_k,
        )

        connections.append(level_connections)

        if level_connections:
            entry_point_id = level_connections[0]

    connections.reverse()

    return connections


def search(hnsw, query):
    """
    Search the HNSW index for the nearest nodes.
    """

    if hnsw._entry_point is None:
        return []

    entry_point_id = greedy_range(
        hnsw=hnsw,
        query=query,
        entry_point_id=hnsw._entry_point,
        start_level=hnsw._max_level,
        end_level=0,
    )

    result_ids = beam(
        hnsw=hnsw,
        query=query,
        entry_point_id=entry_point_id,
        level=0,
        beam_width=hnsw.ef_search,
        top_k=hnsw.top_k,
    )

    return [
        hnsw._nodes[node_id]
        for node_id in result_ids
    ]