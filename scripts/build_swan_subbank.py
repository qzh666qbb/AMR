"""Build a deterministic shallow verbal subset from a SWAN AMR bank."""

import argparse
import json

import penman


def graph_depth(graph: penman.Graph) -> int:
    instances = {source: target for source, role, target in graph.triples if role == ":instance"}
    variables = set(instances)
    edges = [(source, target) for source, role, target in graph.triples if role != ":instance" and target in variables]
    maximum = 0
    stack = [(graph.top, 0)]
    visited = set()
    while stack:
        node, depth = stack.pop()
        maximum = max(maximum, depth)
        visited.add(node)
        stack.extend((target, depth + 1) for source, target in edges if source == node and target not in visited)
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-nodes", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=2)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        entries = json.load(handle)
    selected = []
    for entry in entries:
        graph = penman.decode(entry["normalized_template_amr"])
        instances = {source: target for source, role, target in graph.triples if role == ":instance"}
        if instances.get(graph.top) != "V":
            continue
        if len(instances) > args.max_nodes or graph_depth(graph) > args.max_depth:
            continue
        selected.append(entry)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(selected, handle, ensure_ascii=False, indent=2)
    print(f"selected={len(selected)} total={len(entries)}")


if __name__ == "__main__":
    main()
