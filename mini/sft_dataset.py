"""Prefix rendering adapter for Qwen3.5 templates requiring a user query.

The upstream trainer/collator and loss calculation are unchanged. Only tokenization
of individual turns changes; exact prefix equality is enforced before slicing.
"""
import torch
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


from mini.template import turn_tokens


class Qwen35MiniSFTDataset(MultiTurnSFTDataset):
    def _read_files_and_process(self):
        super()._read_files_and_process()

    def _process_single_message(self, index, message, full_message, tools=None, enable_thinking=None):
        ids, mask = turn_tokens(self.tokenizer, full_message, index)
        values = torch.tensor(ids, dtype=torch.long)
        return values, torch.tensor(mask, dtype=torch.long), torch.ones_like(values), {}
