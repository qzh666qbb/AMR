"""SHA-preserving bridge for the migrated, frozen supervision experiment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src/core")]
import prepare_semantic_supervision2 as original

P2 = ROOT / "第二篇_语义消息写入与读回"
OUT = P2 / "02_实验/semantic_multiscale_supervision2"
PRIVATE = P2 / "08_本地原始资产/cache_amr/semantic_multiscale_supervision2"
COMPLETION = OUT / "feature_completion1"


def generation_path(old):
    return P2 / "08_本地原始资产/cache_amr" / old.split("/.cache/amr/", 1)[1]


def protected_path(old):
    if not old.startswith("experiments/"):
        raise ValueError("Unexpected historical path")
    return P2 / "02_实验" / old.split("experiments/", 1)[1]


def verify():
    """Check the original manifest without editing old SHA/path strings."""
    f = original.load(OUT / "FREEZE.json")
    checks = {
        OUT / "PROTOCOL.md": f["protocol_sha256"],
        OUT / "SPLITS.json": f["split_sha256"],
        OUT / "MEMBERSHIP.json": f["membership_sha256"],
        **{ROOT / k: v for k, v in f["source_sha256"].items()},
        **{PRIVATE / k: v for k, v in f["private_sha256"].items()},
        **{generation_path(k): v for k, v in f["input_sha256"].items()},
        **{protected_path(k): v for k, v in f["protected_sha256"].items()},
        **{OUT / "private_review" / k: v for k, v in f["blank_packet_sha256"].items()
           if k.endswith(".zip")},
    }
    for path, expected in checks.items():
        if not path.is_file() or original.sha(path) != expected:
            raise RuntimeError("Historical asset missing or changed: " + str(path))
    return f


def bind():
    verify()
    original.OUT, original.PRIVATE, original.verify = OUT, PRIVATE, verify
    import collect_semantic_supervision2 as collector
    import run_semantic_supervision2 as runner
    for module in (collector, runner):
        module.OUT, module.PRIVATE, module.verify = OUT, PRIVATE, verify
    return collector, runner
