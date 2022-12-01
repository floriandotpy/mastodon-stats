#

from typing import Optional, Dict, Set
import requests
import json
from pathlib import Path
import time
from dataclasses import dataclass
from requests.exceptions import ConnectTimeout, ConnectionError
from json.decoder import JSONDecodeError
from urllib.parse import urlsplit

from requests.exceptions import ReadTimeout


@dataclass(eq=True, frozen=True)
class Instance:
    uri: str
    protocol: Optional[str] = "https"

    @property
    def url(self):
        return f"{self.protocol}://{self.uri}"

    def __str__(self) -> str:
        return self.uri


def instance_info(instance: Instance):
    url = f"{instance.url}/api/v1/instance"  # DEPRECATED end point. not sure what the replacement is
    response = requests.get(url, timeout=5)
    data = json.loads(response.text)
    return data


def get_peers(instance: Instance, force_refresh: bool = False):
    peers_file = Path("peers.json")
    if peers_file.exists() and not force_refresh:
        print(f"Loading peers from {peers_file}")
        with peers_file.open() as fp:
            return json.load(fp)

    url = f"{instance.url}/api/v1/instance/peers"
    print("Fetching {url}")
    response = requests.get(url)
    data = json.loads(response.text)

    with peers_file.open("w") as fp:
        json.dump(data, fp)
    return data


def fetch_peers_with_stats(instance: Instance, cache_dir: str = "cache", limit=100):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    peers_dir = cache_dir / "peers"
    peers_dir.mkdir(exist_ok=True)

    peers = get_peers(instance)

    print(f"{len(peers)} peers found")

    peers = [Instance(instance) for instance in peers]

    for peer in peers[:limit]:
        print(f"Fetching info for {peer.uri}")
        try:
            peer_info = instance_info(peer)
        except ConnectTimeout:
            print("  Connection timed out, skipping")
            continue
        except ConnectionError:
            print("  Can't establish connection, skipping")
        except JSONDecodeError:
            print("  Malformed JSON, skipping")
            continue
        with (peers_dir / f"{peer.uri}.json").open("w") as fp:
            json.dump(peer_info, fp)
        time.sleep(0.1)


def fetch_timeline(
    instance: Instance,
    remote: bool = False,
    local: bool = False,
    limit=200,
    limit_step: int = 40,
    cache_dir: str = "cache",
    force_refresh: bool = False,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    timelines_dir = cache_dir / "timeline"
    timelines_dir.mkdir(exist_ok=True)
    cache_file = timelines_dir / f"{instance.uri}.json"

    if cache_file.exists() and not force_refresh:
        with cache_file.open() as fp:
            return json.load(fp)

    toots = []
    last_id = None
    for i in range(limit // limit_step):
        url = f"{instance.url}/api/v1/timelines/public"
        params = dict(limit=limit_step, max_id=last_id)
        print(f"Requesting {url} with params {params} (loop step {i})")
        try:
            response = requests.get(url, params, timeout=2)
        except ReadTimeout:
            print("  Read timeout, skipping")
            return toots

        if not response.ok:
            print("  error fetching timeline (auth required?). skipping")
            return toots

        toots_ = json.loads(response.text)
        toots.extend(toots_)
        try:
            last_id = toots[-1]["id"]
        except TypeError as e:
            print(toots)
            raise e
        time.sleep(0.5)

    with cache_file.open("w") as fp:
        json.dump(toots, fp)

    return toots


def parse_instance_from_uri(toot_uri: str):
    parts = urlsplit(toot_uri)
    return Instance(uri=parts.netloc, protocol=parts.scheme)


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def fetch_neighborhood(instance: Instance):
    """
    Explore the neighborhood of the instance.
    The neighborhood is defined as instances that were recently referenced in a toot
    """

    def determine_neighbors_from_toots(toots: list, except_instance: Instance):
        direct_neighbors = set(parse_instance_from_uri(toot["uri"]) for toot in toots)
        if except_instance in direct_neighbors:
            direct_neighbors.remove(except_instance)
        return direct_neighbors

    # map from instance -> list of neighbor instances
    neighbors: Dict[Instance, Set[Instance]] = {}

    toots = fetch_timeline(instance)

    neighbors[instance] = determine_neighbors_from_toots(
        toots, except_instance=instance
    )
    for direct_neighbor in neighbors[instance]:
        toots = fetch_timeline(direct_neighbor)
        neighbors[direct_neighbor] = determine_neighbors_from_toots(
            toots, except_instance=direct_neighbor
        )

    # Turn instances into strings
    neighbors = {
        str(instance): [str(n) for n in instance_neighbors]
        for instance, instance_neighbors in neighbors.items()
    }

    with open("neighborhood.json", "w") as fp:
        json.dump(neighbors, fp)


def main():
    instance = Instance("sigmoid.social")
    # info = instance_info(instance)
    # print(info)

    # stats = fetch_peers_with_stats(instance)
    # print(stats)

    fetch_neighborhood(instance)


if __name__ == "__main__":
    main()
