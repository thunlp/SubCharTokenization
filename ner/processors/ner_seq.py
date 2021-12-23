""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """
import torch
import logging
import os
import copy
import json
from .utils_ner import DataProcessor, get_token_pos, get_labels_of_tokens, get_subchar_pos
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_len, segment_ids, label_ids, 
                 char_label_ids=None, token_ids=None, pos_left=None, pos_right=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len

        # For mapping token-level predictions to char-level predictions.
        self.char_label_ids = char_label_ids
        # For using two-level embeddings to make predictions for eaach subchar.
        self.token_ids = token_ids
        # Shared among two embedding methods
        self.pos_left = pos_left
        self.pos_right = pos_right

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    (all_input_ids, 
     all_attention_mask, 
     all_token_type_ids, 
     all_lens, 
     all_labels, 
    #  all_char_labels, 
     all_token_ids,
    #  all_left_index,
    #  all_right_index,
    #  all_subchar_pos,
     all_pos_left,
     all_pos_right) = map(torch.stack, zip(*batch))

    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:, :max_len]
    # all_char_labels = all_char_labels[:, :max_len]
    all_token_ids = all_token_ids[:, :max_len]
    # all_inv_idx = all_inv_idx[:, :max_len]
    # all_left_index = all_left_index[:, :max_len]
    # all_right_index = all_right_index[:, :max_len]
    # all_subchar_pos = all_subchar_pos[:, :max_len]
    all_pos_left = all_pos_left[:, :max_len]
    all_pos_right = all_pos_right[:, :max_len]

    return (all_input_ids, 
            all_attention_mask, 
            all_token_type_ids, 
            all_labels, 
            all_lens, 
            # all_char_labels,
            all_token_ids,
            # all_left_index,
            # all_right_index,
            # all_subchar_pos,
            all_pos_left,
            all_pos_right)


def collate_fn_with_char_labels(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    (all_input_ids, 
     all_attention_mask, 
     all_token_type_ids, 
     all_lens, 
     all_labels, 
     all_char_labels, 
     all_left_index,
     all_right_index) = map(torch.stack, zip(*batch))

    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:, :max_len]
    all_char_labels = all_char_labels[:, :max_len]
    all_left_index = all_left_index[:, :max_len]
    all_right_index = all_right_index[:, :max_len]

    return (all_input_ids, 
            all_attention_mask, 
            all_token_type_ids, 
            all_labels, 
            all_lens, 
            all_char_labels,
            all_left_index,
            all_right_index)


def collate_fn_wtih_token_ids(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    (all_input_ids, 
     all_attention_mask, 
     all_token_type_ids, 
     all_lens, 
     all_labels, 
     all_token_ids,
     all_pos_left,
     all_pos_right) = map(torch.stack, zip(*batch))

    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:, :max_len]
    all_token_ids = all_token_ids[:, :max_len]
    all_pos_left = all_pos_left[:, :max_len]
    all_pos_right = all_pos_right[:, :max_len]

    return (all_input_ids, 
            all_attention_mask, 
            all_token_type_ids, 
            all_labels, 
            all_lens, 
            all_token_ids,
            all_pos_left,
            all_pos_right)


def get_collate_fn(two_level_embeddings: bool):
    if two_level_embeddings:
        return collate_fn_wtih_token_ids
    else:
        return collate_fn_with_char_labels


def convert_examples_to_features(*args, **kwargs):
    if kwargs['two_level_embeddings']:
        return convert_examples_to_features_token_ids(*args, **kwargs)
    else:
        return convert_examples_to_features_char_labels(*args, **kwargs)


def convert_examples_to_features_token_ids(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    two_level_embeddings=True,
    ):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        # tokens = [tokenizer.tokenize(w) for w in example.text_a]
        # 拆字，并求每个 subchar 的 label
        subchars = []
        labels = []
        for c, label in zip(example.text_a, example.labels):
            this_subchars = tokenizer.tokenize(c)
            label_b = 'B' + label[1:]
            label_i = 'I' + label[1:]
            label_s = 'S' + label[1:]
            if label[0] == 'I':
                labels += [label] * len(this_subchars)
            elif label[0] == 'B':
                labels += [label_b] + [label_i] * (len(this_subchars) - 1)
            elif label[0] == 'S':
                if len(this_subchars) == 1:
                    labels += [label_s]
                else:
                    labels += [label_b] + [label_i] * (len(this_subchars) - 1)
            elif label[0] == 'O':
                labels += ['O'] * len(this_subchars)
            else:
                raise ValueError(f'Invalid label found: {label}')
            subchars += this_subchars
        

        # 获得 tokens
        tokens = tokenizer.tokenize(example.text_a)
        label_ids = [label_map[x] for x in labels]  # 各 subchar 的 label
        subchar_pos = get_subchar_pos(tokens, subchars)
        subchar_pos.append(len(tokens))

        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(subchars) > max_seq_length - special_tokens_count:
            subchars = subchars[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            subchar_pos = subchar_pos[: (max_seq_length - special_tokens_count)]
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        subchars += [sep_token]
        tokens += [sep_token]
        label_ids += [label_map['O']]
        subchar_pos += [len(tokens)]
        segment_ids = [sequence_a_segment_id] * len(subchars)

        if cls_token_at_end:
            subchars += [cls_token]
            tokens += [cls_token]
            label_ids += [label_map['O']]
            subchar_pos += [len(tokens)]
            # char_label_ids += [label_map['O']]
            segment_ids += [cls_token_segment_id]
        else:
            subchars = [cls_token] + subchars
            tokens = [cls_token] + tokens
            label_ids = [label_map['O']] + label_ids
            subchar_pos = [0] + [x + 1 for x in subchar_pos]  # all index in tokens is shifted by 1
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(subchars)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(label_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        token_padding_length = max_seq_length - len(token_ids)

        pos_left = [subchar_pos[i] for i in range(len(subchar_pos) - 1)]
        pos_right = [subchar_pos[i+1] for i in range(len(subchar_pos) - 1)]
        for i in range(len(pos_left) - 1):
            if pos_left[i] == pos_right[i]:
                pos_right[i] += 1

        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids

            label_ids = ([label_map['O']] * padding_length) + label_ids
            pos_left = ([-1] * padding_length) + pos_left
            pos_right = ([-1] * padding_length) + pos_right
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            
            label_ids += [label_map['O']] * padding_length
            pos_left += [-1] * padding_length
            pos_right += [-1] * padding_length
            token_ids += [pad_token] * token_padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(token_ids) == max_seq_length
        assert len(pos_left) == len(pos_right) == max_seq_length

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info(f"token_ids: {' '.join([str(x) for x in token_ids])}")
            logger.info(f'pos_left: {pos_left}')
            logger.info(f'pos_right: {pos_right}')

        features.append(
            InputFeatures(
                input_ids=input_ids, 
                input_mask=input_mask, 
                input_len=input_len,
                segment_ids=segment_ids, 
                label_ids=label_ids, 
                token_ids=token_ids, 
                pos_left=pos_left, 
                pos_right=pos_right,
            )
        )
    return features


def convert_examples_to_features_char_labels(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    two_level_embeddings=False,  # Will be ignored
    ):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        # Prediction on tokens, then map to chars predictions.
        tokens = tokenizer.tokenize(example.text_a)
        token_pos = get_token_pos(tokens, tokenizer, split_tokens)
        token_pos.append(len(tokens))
        token_labels = get_labels_of_tokens(example.labels, tokens, token_pos)
        label_ids = [label_map[x] for x in token_labels]
        char_label_ids = [label_map[x] for x in example.labels]


        # print(example.text_a)
        # print(tokens)
        # print(token_pos)
        # print(token_labels)
        # print(example.labels)
        # exit()

        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            token_pos = token_pos[: (max_seq_length - special_tokens_count)]
        if len(char_label_ids) > max_seq_length - special_tokens_count:
            char_label_ids = char_label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        tokens += [sep_token]        
        label_ids += [label_map['O']]
        char_label_ids += [label_map['O']]
        token_pos += [token_pos[-1] + 1]
        segment_ids = [sequence_a_segment_id] * len(subchars)

        if cls_token_at_end:
            subchars += [cls_token]
            tokens += [cls_token]
            subchar_label_ids += [label_map['O']]
            subchar_pos += [len(tokens)]
            # char_label_ids += [label_map['O']]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [label_map['O']] + label_ids
            token_pos = [0] + [x + 1 for x in token_pos]
            char_label_ids = [label_map['O']] + char_label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(label_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        char_padding_length = max_seq_length - len(char_label_ids)        
        pos_left = [token_pos[i] for i in range(len(token_pos) - 1)]
        pos_right = [token_pos[i+1] for i in range(len(token_pos) - 1)]

        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([label_map['O']] * padding_length) + label_ids

            pos_left = ([-1] * padding_length) + pos_left
            pos_right = ([-1] * padding_length) + pos_right
            # label_ids = ([pad_token] * padding_length) + label_ids
            char_label_ids = ([pad_token] * char_padding_length) + char_label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [label_map['O']] * padding_length

            char_label_ids += [0] * char_padding_length
            pos_left += [-1] * padding_length
            pos_right += [-1] * padding_length

        # print(example.text_a)
        # print(example.labels)
        # print(subchars)
        # print(input_ids)
        # print(labels)
        # print(label_ids)
        # exit()

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(char_label_ids) == max_seq_length
        assert len(pos_left) == len(pos_right) == max_seq_length

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info("char_label_ids: %s", " ".join([str(x) for x in char_label_ids]))
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info(f'pos_left: {pos_left}')
            logger.info(f'pos_right: {pos_right}')

        features.append(
            InputFeatures(
                input_ids=input_ids, 
                input_mask=input_mask, 
                input_len=input_len,
                segment_ids=segment_ids, 
                label_ids=label_ids, 
                char_label_ids=char_label_ids, 
                pos_left=pos_left, 
                pos_right=pos_right,
            )
        )
    return features


class CnerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_labels(self):
        """See base class."""
        return ["X",'B-CONT','B-EDU','B-LOC','B-NAME','B-ORG','B-PRO','B-RACE','B-TITLE',
                'I-CONT','I-EDU','I-LOC','I-NAME','I-ORG','I-PRO','I-RACE','I-TITLE',
                'O','S-NAME','S-ORG','S-RACE',"[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-','I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class CluenerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["X", "B-address", "B-book", "B-company", 'B-game', 'B-government', 'B-movie', 'B-name',
                'B-organization', 'B-position','B-scene',"I-address",
                "I-book", "I-company", 'I-game', 'I-government', 'I-movie', 'I-name',
                'I-organization', 'I-position','I-scene',
                "S-address", "S-book", "S-company", 'S-game', 'S-government', 'S-movie',
                'S-name', 'S-organization', 'S-position',
                'S-scene','O',"[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

ner_processors = {
    "cner": CnerProcessor,
    'cluener':CluenerProcessor
}
