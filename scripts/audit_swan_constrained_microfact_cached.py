"""Run the frozen constrained-microfact audit with cached tokenizer metadata."""

import argparse
import json
from pathlib import Path

import audit_swan_constrained_microfact as frozen
from run_swan_constrained_microfact import OUTPUT, safe_token_ids


_SAFE_CACHE = {}


def cached_validate_token_path(tokenizer, token_ids):
    identity = id(tokenizer)
    if identity not in _SAFE_CACHE:
        _SAFE_CACHE[identity] = set(safe_token_ids(tokenizer))
    safe = _SAFE_CACHE[identity]
    period = tokenizer.encode(".", add_special_tokens=False)[0]
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    prefix = []
    ended = False
    for position, token_id in enumerate(token_ids):
        if ended:
            if token_id not in {eos, pad}:
                raise ValueError(
                    f"Non-padding token after constrained EOS at {position}"
                )
            continue
        text = tokenizer.decode(prefix, skip_special_tokens=True).strip()
        words = len(text.split())
        if text.endswith("."):
            allowed = token_id == eos
        elif words >= 10:
            allowed = token_id == period
        else:
            allowed = token_id in safe or (
                words >= 7 and token_id == period
            )
        if not allowed:
            raise ValueError(
                f"Illegal constrained token at {position}: {token_id}"
            )
        prefix.append(token_id)
        ended = token_id == eos
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "recovery"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    frozen.validate_token_path = cached_validate_token_path
    result = (
        frozen.audit(args.output)
        if args.stage == "audit"
        else frozen.recovery(args.output)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
