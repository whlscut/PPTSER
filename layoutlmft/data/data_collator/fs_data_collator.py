import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers.data.data_collator import (
    DataCollatorMixin,
)

flatten = lambda l:[item for sublist in l for item in sublist]
POSITION_ID_UPPER_BOUND = 514
MAX_SEQUENCE_LENGTH = 512

def create_position_ids_from_input_ids(input_ids, padding_idx, len_of_tokenized_label):

    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask + padding_idx 
    total_label_len = sum(len_of_tokenized_label)
    start = -total_label_len

    return incremental_indices.long()

def pre_calc_rel_mat(segment_ids):
    valid_span = torch.zeros((segment_ids.shape[0], segment_ids.shape[1], segment_ids.shape[1]),
                             device=segment_ids.device, dtype=torch.bool)
    for i in range(segment_ids.shape[0]):
        for j in range(segment_ids.shape[1]):
            valid_span[i, j, :] = segment_ids[i, :] == segment_ids[i, j]

    return valid_span

@dataclass
class FewShotDataCollator(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """
    def __init__(self, **kargs):
        super().__init__()
        self.label_pad_token_id: int = -100
        for k, v in kargs.items():
            self.__dict__.update({k: v})

        if kargs.get('tokenized_label', None):
            self.cat_label = flatten(self.tokenized_label)
            self.len_of_tokenized_label = [len(i) for i in self.tokenized_label]
            self.total_len_of_all_tokenized_label = sum(self.len_of_tokenized_label)

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None 

        images = None
        if "images" in features[0]:
            images = torch.stack([torch.tensor(d.pop("images")) for d in features])
            IMAGE_LEN = int(images.shape[-1] / 16) * int(images.shape[-1] / 16) + 1

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if images is not None:
            batch["images"] = images
            batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) and k == 'attention_mask' else v
                     for k, v in batch.items()}
            visual_attention_mask = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long)
            batch["attention_mask"] = torch.cat([batch['attention_mask'], visual_attention_mask], dim=1)
        if labels is None:
            return batch

        has_bbox_input = "bbox" in features[0]
        has_position_input = "position_ids" in features[0]
        padding_idx=self.tokenizer.pad_token_id
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            if isinstance(batch["labels"][0][0], list):
                batch["labels"] = [[sub_label + [self.label_pad_token_id] * (sequence_length - len(sub_label)) for sub_label in label] 
                    for label in labels]
            else:
               batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
            if has_bbox_input:
                batch["bbox"] = [bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
            if has_position_input:
                batch["position_ids"] = [position_id + [padding_idx] * (sequence_length - len(position_id))
                                          for position_id in batch["position_ids"]]

        else:
            if isinstance(batch["labels"][0][0], list):
                batch["labels"] = [[[self.label_pad_token_id] * (sequence_length - len(sub_label)) + sub_label for sub_label in label]
                    for label in labels]
            else:
                batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
            if has_bbox_input:
                batch["bbox"] = [[[0, 0, 0, 0]] * (sequence_length - len(bbox)) + bbox for bbox in batch["bbox"]]
            if has_position_input:
                batch["position_ids"] = [[padding_idx] * (sequence_length - len(position_id))
                                          + position_id for position_id in batch["position_ids"]]

        if 'tokenized_label' in self.__dict__:
            for i in range(len(batch["labels"])):
                batch['input_ids'][i][-self.total_len_of_all_tokenized_label:] = self.cat_label
                batch['bbox'][i][-self.total_len_of_all_tokenized_label:] = [[0] * 4 for i in range(self.total_len_of_all_tokenized_label)]

                batch['attention_mask'][i][MAX_SEQUENCE_LENGTH - \
                    self.total_len_of_all_tokenized_label: MAX_SEQUENCE_LENGTH] = 1

                if 'segment_ids' in batch:
                    assert 'position_ids' in batch
                    batch['segment_ids'][i] = batch['segment_ids'][i] + [batch['segment_ids'][i][-1] + 1] * (sequence_length - len(batch['segment_ids'][i])) + [
                        batch['segment_ids'][i][-1] + 2] * IMAGE_LEN
                    last_segment_id = batch['segment_ids'][i][-1]

                if 'position_ids' in batch:
                    start_index = MAX_SEQUENCE_LENGTH - self.total_len_of_all_tokenized_label
                    for j in range(len(self.len_of_tokenized_label)):
                        end_index = start_index + self.len_of_tokenized_label[j]
                        batch['position_ids'][i][start_index + 1: end_index - 1] = list(range(2, self.len_of_tokenized_label[j]))
                        batch['position_ids'][i][start_index], batch['position_ids'][i][end_index - 1] = 2, 2

                        batch['bbox'][i][end_index - 1] = [1000] * 4

                        if 'segment_ids' in batch:
                            batch['segment_ids'][i][start_index] = last_segment_id + 1
                            batch['segment_ids'][i][start_index + 1: end_index - 1] = [last_segment_id + 2] * (self.len_of_tokenized_label[j] - 2)
                            batch['segment_ids'][i][end_index - 1] = last_segment_id + 3
                            last_segment_id += 3
                        
                        start_index = end_index

        batch = {k: torch.tensor(v, dtype=torch.int64) if (isinstance(v[0], list)) else v
                    for k, v in batch.items()for k, v in batch.items() }

        if 'position_ids' not in batch:
            batch['position_ids'] = create_position_ids_from_input_ids(batch['input_ids'],
                                                                       self.tokenizer.pad_token_id,
                                                                       len_of_tokenized_label=self.len_of_tokenized_label)

        if 'segment_ids' in batch:
            valid_span = pre_calc_rel_mat(
                segment_ids=batch['segment_ids']
            )
            batch['valid_span'] = valid_span
            del batch['segment_ids']

        if images is not None:
            if len(batch["labels"].shape) == 3:
                bs, n_label, _ = batch["labels"].shape
                visual_labels = torch.ones((bs, n_label, IMAGE_LEN), dtype=torch.long) * -100
                batch["labels"] = torch.cat([batch["labels"], visual_labels], dim=-1)
            else:
                visual_labels = torch.ones((batch["labels"].shape[0], IMAGE_LEN), dtype=torch.long) * -100
                batch["labels"] = torch.cat([batch["labels"], visual_labels], dim=-1)

        return batch