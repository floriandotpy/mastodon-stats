import json


def convert_for_web(min_connections: int = 10):

    with open("neighborhood.json") as fp:
        neighborhood = json.load(fp)

    node_to_id = {instance: idx for idx, instance in enumerate(neighborhood.keys())}

    def get_id(instance):
        if instance not in node_to_id:
            new_id = max(node_to_id.values()) + 1
            node_to_id[instance] = new_id
        return node_to_id[instance]

    data = {
        # "nodes": [{"id": get_id(instance), "name": instance} for instance in neighborhood.keys()],
        "nodes": [],
        "links": [],
    }

    connected_nodes = set()
    for instance_from, direct_neighbors in neighborhood.items():
        for instance_to, count in direct_neighbors.items():
            if count < min_connections:
                continue
            data["links"].append(
                {"source_id": get_id(instance_from), "target_id": get_id(instance_to)}
            )
            connected_nodes.add(instance_from)
            connected_nodes.add(instance_to)

    for instance, idx in node_to_id.items():
        if instance not in connected_nodes:
            continue
        data["nodes"].append({"id": idx, "name": instance})

    print(data)
    with open("web/data.json", "w") as fp:
        json.dump(data, fp)


if __name__ == "__main__":
    convert_for_web()
