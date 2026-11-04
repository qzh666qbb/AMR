import types
import unittest

import semstamp_bounded_generation as bounded


class FakeIds:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return 1

    def __getitem__(self, _index):
        return range(self.length)

    def size(self, _dimension):
        return self.length

    def to(self, _device):
        return self


class FakeTokenizer:
    def encode(self, _prompt, return_tensors):
        if return_tensors != "pt":
            raise AssertionError("unexpected tensor format")
        return FakeIds(2)


class FakeSentenceEndCriteria:
    def __init__(self, _tokenizer):
        self.updates = []

    def update(self, text):
        self.updates.append(text)


class FakeLshModel:
    def __init__(self, candidates):
        self.candidates = iter(candidates)

    def get_hash(self, values):
        if values == ["prompt"]:
            return [0]
        return [next(self.candidates)]


def fake_official(candidates, close_results, max_trials=3):
    close_iter = iter(close_results)

    def gen_sent(**kwargs):
        token_length = kwargs["text_ids"].length + 1
        return f" segment-{token_length}.", FakeIds(token_length)

    def reject_close_generation(*_args, **_kwargs):
        return (["accepted"] if next(close_iter) else []), [0]

    def get_mask_from_seed(_dimension, _rate, _seed):
        return {1}

    return types.SimpleNamespace(
        MAX_TRIALS=max_trials,
        SentenceEndCriteria=FakeSentenceEndCriteria,
        StoppingCriteriaList=list,
        gen_sent=gen_sent,
        reject_close_generation=reject_close_generation,
        get_mask_from_seed=get_mask_from_seed,
    )


def run(candidates, close_results, max_trials=3, max_new_tokens=2):
    official = fake_official(candidates, close_results, max_trials)
    return bounded.bounded_lsh_reject_completion(
        prompt="prompt",
        model=types.SimpleNamespace(
            config=types.SimpleNamespace(_name_or_path="opt-fixture")
        ),
        tokenizer=FakeTokenizer(),
        gen_config=types.SimpleNamespace(max_new_tokens=max_new_tokens),
        lsh_model=FakeLshModel(candidates),
        lsh_dim=3,
        lmbd=0.25,
        device="cpu",
        margin=0.02,
        official_module=official,
    )


class BoundedSamplerTests(unittest.TestCase):
    def test_accepts_first_valid_candidate(self):
        text, metrics = run([1], [True])
        self.assertTrue(text.endswith("segment-3."))
        self.assertEqual(metrics["total_trials"], 1)
        self.assertEqual(metrics["fallback_acceptances"], 0)
        self.assertEqual(metrics["max_current_trials_observed"], 1)

    def test_signature_rejection_is_bounded(self):
        _, metrics = run([2, 2, 2], [True, True, True])
        self.assertEqual(metrics["total_trials"], 3)
        self.assertEqual(metrics["accepted_segments"], 1)
        self.assertEqual(metrics["max_trials_reached_segments"], 1)
        self.assertEqual(metrics["fallback_acceptances"], 1)
        self.assertEqual(metrics["max_current_trials_observed"], 3)

    def test_close_filter_rejection_is_bounded(self):
        _, metrics = run([2], [False, False, False])
        self.assertEqual(metrics["total_trials"], 3)
        self.assertEqual(metrics["fallback_acceptances"], 1)

    def test_rejection_counter_resets_per_segment(self):
        _, metrics = run(
            [2, 2, 2, 2, 2, 2],
            [True, True, True, True, True, True],
            max_new_tokens=3,
        )
        self.assertEqual(metrics["accepted_segments"], 2)
        self.assertEqual(metrics["total_trials"], 6)
        self.assertEqual(metrics["fallback_acceptances"], 2)
        self.assertEqual(metrics["max_current_trials_observed"], 3)

    def test_invalid_trial_caps_rejected(self):
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                bounded.validate_trial_cap(invalid)


if __name__ == "__main__":
    unittest.main()
