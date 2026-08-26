from vecs.hnsw import HnswState, HNSW, Node

from pathlib import Path
import json

def _node_to_dict(node: Node) -> dict:
    return {
        'vector': node.vector,
        'metadata': node.metadata,
        'content': node.content,
        'neighbors': node.neighbors
    }

def _dict_to_node(data: dict) -> Node:
    node = Node(
        vector=data['vector'],
        metadata=data['metadata'],
        content=data['content'],
        level=len(data['neighbors']) - 1 if data['neighbors'] else None
    )
    node.neighbors = data['neighbors']
    return node

def save(hnsw: HNSW, file: Path) -> None:
    serializable_nodes = {
        node_id: _node_to_dict(node)
        for node_id, node in hnsw._nodes.items()
    }

    state = HnswState(
        entry_point=hnsw._entry_point,
        nodes=serializable_nodes,
        max_level=hnsw._max_level,
        ef_construction=hnsw.ef_construction,
        m=hnsw.m,
        ef_search=hnsw.ef_search,
        top_k=hnsw.top_k,
    )

    with open(file, "w", encoding="utf-8") as f:
        f.write(json.dumps(state.model_dump()))

def load(path: Path) -> HnswState:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    deserialized_nodes = {
        int(node_id): _dict_to_node(node_data)
        for node_id, node_data in data['nodes'].items()
    }
    data['nodes'] = deserialized_nodes

    return HnswState.model_validate(data)