import requests
import json
from pathlib import Path
import time


def fetch_users(
    instance_url: str,
    limit: int = 100,
    cache_dir: str = "cache",
    sleep_seconds: float = 1.0,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    url = f"{instance_url}/api/v1/directory"
    print(url)
    offset = 0
    done = False
    profiles = []
    while not done:

        print(f"Fetching users starting from {offset} to {offset+limit} ...")
        response = requests.get(url, {"local": True, "offset": offset, "limit": limit})
        data = json.loads(response.text)
        cache_file = cache_dir / f"directory_{offset}.json"
        with cache_file.open("w") as fp:
            json.dump(data, fp)

        if not len(data):
            # last profile reached
            done = True

        profiles.extend(data)
        time.sleep(sleep_seconds)
        offset += limit

    return profiles


def main():
    instance_url = "https://sigmoid.social"
    profiles = fetch_users(instance_url)
    with open("profiles.json", "w") as fp:
        json.dump(profiles, fp)


if __name__ == "__main__":
    main()
