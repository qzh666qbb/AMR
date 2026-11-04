"""Engineering tensors only; these tests provide no detection evidence."""
import unittest

import torch

from src.core.multiscale_family_training import (
    family_groups, family_negative_weights, guarded_epoch, max_format_loss,
    select_coefficient,
)


class FamilyTrainingTests(unittest.TestCase):
    def test_budget_representatives_and_mass(self):
        keys = [("d", attack, slot) for slot in range(4) for attack in ("a", "b", "c")]
        labels = torch.tensor([1.] * 12 + [0.] * 12)
        keys = keys + keys
        base = torch.full((24,), 1 / 24)
        scores = torch.tensor([5., 5., 4., 3., 9., 8., 0., 1., 2., 1., 1., 1.])
        weights, selected, audit = family_negative_weights(
            base, labels, keys, scores, fraction=0.25)
        chosen = [keys[i] for i in selected.nonzero().flatten().tolist()]
        self.assertEqual(chosen, [("d", "a", 0), ("d", "b", 1), ("d", "c", 2)])
        self.assertEqual(len({(k[0], k[2]) for k in chosen}), 3)
        self.assertEqual(audit["selected_rows"], 3)
        self.assertTrue(torch.equal(weights[:12], base[:12]))
        self.assertLess(abs(float(weights[12:].sum() - base[12:].sum())), 1e-7)
        self.assertTrue((weights > 0).all())
        # Selected multiplier remains exactly three relative to unselected rows.
        self.assertAlmostEqual(float(weights[12] / weights[13]), 3., places=6)

    def test_family_ties_and_invalid_budget(self):
        keys = [("d", "z", 0), ("d", "a", 0), ("d", "a", 1), ("d", "z", 1)]
        base, labels = torch.ones(4) / 4, torch.zeros(4)
        _, selected, _ = family_negative_weights(base, labels, keys, torch.ones(4), 0.25)
        self.assertEqual(selected.tolist(), [False, True, False, False])
        with self.assertRaises(ValueError):
            family_negative_weights(base, labels, keys, torch.ones(4), 1.)
        with self.assertRaises(ValueError):
            family_groups(keys + keys[:1])

    def test_max_loss_gradients_and_missing(self):
        logits = torch.tensor([1., 3., 3., -torch.inf], requires_grad=True)
        loss = max_format_loss(logits, [[0, 1, 2], [3]], torch.tensor([1., 1.]))
        self.assertAlmostEqual(float(loss), float(torch.nn.functional.softplus(torch.tensor(3.)) / 2))
        loss.backward()
        self.assertEqual(float(logits.grad[0]), 0.)
        self.assertAlmostEqual(float(logits.grad[1]), float(torch.sigmoid(torch.tensor(3.)) / 4))
        self.assertEqual(float(logits.grad[1]), float(logits.grad[2]))
        self.assertEqual(float(logits.grad[3]), 0.)
        self.assertTrue(torch.isfinite(logits.grad).all())
        with self.assertRaises(ValueError):
            max_format_loss(torch.tensor([float("nan")]), [[0]], torch.ones(1))

    def test_positive_guard_and_worst_condition(self):
        def row(epoch, bce, tpr, worst, positive):
            return {"epoch": epoch, "validation_bce": bce, "validation_tpr": tpr,
                    "validation_worst_fpr": worst, "validation_positive_bce": positive}
        history = [row(1, .3, .8, .2, .3), row(2, .31, .79, .04, .31),
                   row(3, .32, .5, 0., .8)]
        selection = guarded_epoch(history)
        self.assertEqual(selection["bce_epoch"], 1)
        self.assertEqual(selection["guarded_epoch"], 2)
        coefficient = select_coefficient(
            {"L000": [history[0]], "L005": [history[1]], "L025": [history[2]]})
        self.assertEqual(coefficient["selected"], "L005")
        self.assertNotIn("L025", coefficient["eligible"])


if __name__ == "__main__":
    unittest.main()
