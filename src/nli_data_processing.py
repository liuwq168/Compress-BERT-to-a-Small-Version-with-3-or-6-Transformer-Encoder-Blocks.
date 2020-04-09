import csv
import sys
import os
import logging
import glob
import torch
import numpy as np

import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score


from src.modeling import BertForSequenceClassificationEncoder, FCClassifierForSequenceClassification


logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    def get_test_examples(self, data_dir):
        """ Get a collection of 'inputexample's for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a comma separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter=",", quotechar=quotechar) # ","分割
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                if len(line) == 0:continue
                lines.append(line)
            return lines

class AGProcessor(DataProcessor):
    """Processor for the AGnews dataset (four classes. world,sports, business, sci/tech)"""
    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {}",format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    
    
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["1", "2", "3", "4"] #four classes

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = ",".join(line[1:])
            # text_b = None
            if set_type == 'test':
                label = self.get_labels()[0] #不用测试集的真实标签，用假标签
            else:
                label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    acc = (preds == labels).mean()
    return {
        "acc":acc,
        "error rate": 1.0 - acc}


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return simple_accuracy(preds, labels)
    else:
        raise KeyError(task_name)


processors = {
    "AG": AGProcessor,
}

output_modes = {
    "AG": "classification",
}


def get_glue_task_dataloader(task_name, set_name, tokenizer, args, sampler, batch_size=None, knowledge=None, extra_knowledge=None):
    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    if set_name.lower() == 'train':
        examples = processor.get_train_examples(args.raw_data_dir)
    elif set_name.lower() == 'dev':
        examples = processor.get_dev_examples(args.raw_data_dir)
    elif set_name.lower() == 'test':
        examples = processor.get_test_examples(args.raw_data_dir)
    else:
        raise ValueError('{} as set name not available for now, use \'train\' or \'dev\' instead'.format(set_name))

    if batch_size is None:
        batch_size = args.train_batch_size if set_name.lower() == 'train' else args.eval_batch_size

    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode)
    # logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(examples))
    # logger.info("  Batch size = %d", batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    if knowledge is not None:
        all_knowledge = torch.tensor(knowledge, dtype=torch.float)
        print(all_knowledge.shape) #torch.Size([67349, 2])√
        if extra_knowledge is None:
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_knowledge)
        else:
            tok_index = [int(i) for i in range(128)]
            print('extraknledgeshape*****************88')
            print(np.array(extra_knowledge).shape)#(128, 21, 768)
            extra_knowledge_tensor = torch.stack([torch.FloatTensor(extra_knowledge[int(i)])for i in tok_index]).transpose(0,1)
            print('extraknledge--tensor----shape*****************88')
            print(extra_knowledge_tensor.shape)#torch.Size([128, 21, 768])
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_knowledge, extra_knowledge_tensor) #第0维形状不一样
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    dataloader = DataLoader(dataset, sampler=sampler(dataset), batch_size=batch_size)
    return examples, dataloader, all_label_ids #train_examples, train_dataloader,


def init_glue_model(task_name, output_all_layers, num_hidden_layers, config):
    logger.info('predicting for %s' % task_name.upper())
    processor = processors[task_name.lower()]()
    output_mode = output_modes[task_name.lower()]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    encoder_bert = BertForSequenceClassificationEncoder(config, output_all_encoded_layers=output_all_layers,
                                                        num_hidden_layers=num_hidden_layers)
    classifier = FCClassifierForSequenceClassification(config, num_labels, config.hidden_size, 0)
    return encoder_bert, classifier # student_encoder, student_classifier


def parse_filename(file_name):
    info = file_name.split('_')
    model = info[0]
    task = info[1]
    nlayer = int(info[2].split('.')[1])
    lr = float(info[3].split('.')[1])
    T = float(info[4].split('.')[1])
    alpha = float('.'.join(info[5].split('.')[1:]))
    beta = float('.'.join(info[6].split('.')[1:]))
    bs = info[7].split('.')[1]
    return model, task, nlayer, lr, T, alpha, beta, int(bs.split('-')[0]), int(bs.split('-')[-1])


def find_eval_res_task_subdir(task, kd_folder, sub_dir):
    res_dir = os.path.join(kd_folder, task, sub_dir)
    all_files = glob.glob(res_dir + '/*')
    all_res, mean_res = [], []
    for f in all_files:
        fbase = os.path.basename(f)
        try:
            res = pd.read_csv(os.path.join(f, 'eval_log.txt'), sep=',|\s+')
        except:
            print(f'opening {f} failed!')
            continue

        if len(res) == 0:
            print(f'results in {f} are missing!')
            continue

        best_acc_idx = res.acc.idxmax()
        res_best = res.values[best_acc_idx]

        all_res.append([fbase] + list(res_best))
        mean_res.append(['-'.join(fbase.split('-')[:-2])] + list(res_best))

    all_res = pd.DataFrame(all_res, columns=['uname', 'epoch', 'acc', 'loss'])
    mean_res = pd.DataFrame(mean_res, columns=['uname', 'epoch', 'acc', 'loss'])
    mean_res = mean_res.groupby('uname', as_index=False).mean()

    best_res_idx = all_res.acc.idxmax()
    best_res_all = [list(all_res.values[best_res_idx])]

    best_res_idx = mean_res.acc.idxmax()
    best_res_mean = [list(mean_res.values[best_res_idx])]
    return all_res, mean_res, best_res_all, best_res_mean

