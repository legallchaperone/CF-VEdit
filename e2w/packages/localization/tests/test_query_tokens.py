"""Unit tests for the ADR-0006 query-token attention/position design.

No GPU, no checkpoint, no transformers import needed — `_build_query_attention_mask`
and `_continuation_offsets` are pure tensor/list construction. This is the only
executable verification for ADR-0006 in this change; it does not prove the real
Sa2VA forward pass still runs end-to-end under the new 4D mask (see the ADR's
"Not validated on GPU" consequence).
"""
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import torch

from e2w_localization.query_tokens import _build_query_attention_mask, _continuation_offsets


class ContinuationOffsetsTest(unittest.TestCase):
    def test_offsets_are_seg_then_tied_edit(self):
        self.assertEqual(_continuation_offsets(4), [1, 2, 3, 3, 3, 3])
        self.assertEqual(_continuation_offsets(1), [1, 2, 3])
        self.assertEqual(_continuation_offsets(0), [1, 2])

    def test_offsets_tie_edit_slots_to_one_shared_position(self):
        last = 10
        offsets = torch.tensor(_continuation_offsets(4), dtype=torch.long)
        positions = (last + offsets).tolist()
        seg_dir, seg_ind, *edit = positions
        self.assertNotEqual(seg_dir, seg_ind)
        self.assertTrue(all(p == edit[0] for p in edit), edit)
        self.assertNotIn(edit[0], (seg_dir, seg_ind))


class BuildQueryAttentionMaskTest(unittest.TestCase):
    """ADR-0006 connectivity contract: prompt/video causal unchanged; [EDIT]
    bidirectional internally; [EDIT] <-> [SEG_DIR]/[SEG_IND] blocked both ways;
    [SEG_DIR]/[SEG_IND] mutual visibility unchanged (open question, not this ADR's
    concern)."""

    N_PROMPT = 4
    NUM_EDIT_SLOTS = 4

    def _mask(self, prompt_padding_mask=None):
        return _build_query_attention_mask(
            n_prompt=self.N_PROMPT, num_edit_slots=self.NUM_EDIT_SLOTS,
            prompt_padding_mask=prompt_padding_mask,
            dtype=torch.float32, device=torch.device("cpu"),
        )

    def test_shape(self):
        total_len = self.N_PROMPT + 2 + self.NUM_EDIT_SLOTS
        mask = self._mask()
        self.assertEqual(tuple(mask.shape), (1, 1, total_len, total_len))

    def test_prompt_block_still_causal(self):
        mask = self._mask()[0, 0]
        min_val = torch.finfo(torch.float32).min
        for i in range(self.N_PROMPT):
            for j in range(self.N_PROMPT):
                expected = 0.0 if j <= i else min_val
                self.assertEqual(mask[i, j].item(), expected, f"prompt[{i},{j}]")

    def test_seg_dir_seg_ind_mutual_visibility_unchanged(self):
        mask = self._mask()[0, 0]
        seg_dir, seg_ind = self.N_PROMPT, self.N_PROMPT + 1
        min_val = torch.finfo(torch.float32).min
        self.assertEqual(mask[seg_ind, seg_dir].item(), 0.0)      # seg_ind sees seg_dir
        self.assertEqual(mask[seg_dir, seg_ind].item(), min_val)  # seg_dir does not see seg_ind

    def test_edit_slots_fully_bidirectional(self):
        mask = self._mask()[0, 0]
        edit_start = self.N_PROMPT + 2
        edit_idx = range(edit_start, edit_start + self.NUM_EDIT_SLOTS)
        for i in edit_idx:
            for j in edit_idx:
                self.assertEqual(mask[i, j].item(), 0.0, f"edit[{i},{j}]")

    def test_edit_and_seg_never_see_each_other(self):
        mask = self._mask()[0, 0]
        seg_idx = [self.N_PROMPT, self.N_PROMPT + 1]
        edit_start = self.N_PROMPT + 2
        edit_idx = list(range(edit_start, edit_start + self.NUM_EDIT_SLOTS))
        min_val = torch.finfo(torch.float32).min
        for e in edit_idx:
            for s in seg_idx:
                self.assertEqual(mask[e, s].item(), min_val, f"edit[{e}]->seg[{s}] must be blocked")
        for s in seg_idx:
            for e in edit_idx:
                self.assertEqual(mask[s, e].item(), min_val, f"seg[{s}]->edit[{e}] must be blocked")

    def test_padded_prompt_column_blocked_for_every_row(self):
        padding_mask = torch.tensor([[1, 1, 1, 0]])  # last prompt token is padding
        mask = self._mask(prompt_padding_mask=padding_mask)[0, 0]
        min_val = torch.finfo(torch.float32).min
        padded_col = self.N_PROMPT - 1
        for i in range(mask.shape[0]):
            self.assertEqual(mask[i, padded_col].item(), min_val, f"row {i} should not see padded col")

    def test_batch_size_greater_than_one_raises(self):
        bad_padding_mask = torch.ones(2, self.N_PROMPT)
        with self.assertRaises(AssertionError):
            self._mask(prompt_padding_mask=bad_padding_mask)


if __name__ == "__main__":
    unittest.main()
