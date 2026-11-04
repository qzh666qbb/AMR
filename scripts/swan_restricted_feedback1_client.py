"""Fixed-pool attacker: stdin is its complete detector-information interface."""
import hashlib
import json
import math
import sys


def select(candidates, mode, budget, query):
    if mode not in ("none", "label", "score") or type(budget) is not int or budget < 1:
        raise ValueError("Invalid feedback contract")
    pool = candidates[:budget]
    if not pool or len({r["candidate_id"] for r in candidates}) != len(candidates):
        raise ValueError("Empty or duplicate candidate pool")
    first = pool[0]["candidate_id"]
    if mode == "none":
        return {"selected": first, "queries": 0, "first_label_escape_query": None}
    observations = []
    for item in pool:
        response = query(item)
        if mode == "label":
            if set(response) != {"label"} or type(response["label"]) is not bool:
                raise ValueError("Binary feedback must be exactly one boolean")
        else:
            if (set(response) != {"score"} or type(response["score"]) not in (int, float)
                    or not math.isfinite(response["score"])):
                raise ValueError("Continuous feedback must be one finite number")
        observations.append((item["candidate_id"], response))
    if mode == "label":
        escaped = [(i + 1, key) for i, (key, response) in enumerate(observations)
                   if response["label"] is False]
        chosen, first_escape = (escaped[0][1], escaped[0][0]) if escaped else (first, None)
    else:
        # Python min keeps the first observed candidate on a score tie.
        chosen = min(observations, key=lambda r: r[1]["score"])[0]
        first_escape = None
    return {"selected": chosen, "queries": len(observations),
            "first_label_escape_query": first_escape}


def main():
    contract = json.loads(sys.stdin.readline())
    if set(contract) != {"candidates", "mode", "budget"}:
        raise ValueError("Unexpected attacker input field")
    def query(item):
        request = {"op": "query", "candidate_id": item["candidate_id"],
                   "text_sha256": hashlib.sha256(item["text"].encode()).hexdigest()}
        print(json.dumps(request), flush=True)
        return json.loads(sys.stdin.readline())
    result = select(contract["candidates"], contract["mode"], contract["budget"], query)
    print(json.dumps({"op": "final", **result}), flush=True)


if __name__ == "__main__":
    main()
