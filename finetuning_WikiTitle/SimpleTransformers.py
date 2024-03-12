from __future__ import absolute_import, division, print_function
#import sys
import os
#script_dir = os.path.abspath(os.path.dirname(sys.argv[0]) or '.')
import math
import json
import random
import pickle
import copy
import warnings
import logging
import argparse
from multiprocessing import cpu_count

from DataProcessor import MultipleChoiceExample 
from typing import Optional, List, Any, Union
from transformers import PreTrainedTokenizer,AutoModelForMultipleChoice
from dataclasses import dataclass

import torch
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from tensorboardX import SummaryWriter
from tqdm.auto import trange, tqdm

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertForMultipleChoice,
    BertTokenizer,
)
from transformers import (
    DistilBertConfig,
    DistilBertForMultipleChoice,
    DistilBertTokenizer,
)
from transformers import RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer
from transformers import (
    XLMRobertaConfig,
    XLMRobertaForMultipleChoice,
    XLMRobertaTokenizer,
)
from transformers import (
    CamembertConfig,
    CamembertForMultipleChoice,
    CamembertTokenizer,
)

from simpletransformers.config.global_args import global_args

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)

from argparse import ArgumentParser

path="/s/red/a/nobackup/cwc-ro/shadim/languages/"

@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: Any
    input_mask: Any
    segment_ids: Any = None
    label_ids: Any = None
    candidates: Any = None
    example_id: str = None

def convert_multiple_choice_examples_to_features(
    examples: List[MultipleChoiceExample],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    label_list: List[str],
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            inputs = tokenizer(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                truncation='longest_first',
                pad_to_max_length=True,
            )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            choices_inputs.append(inputs)
 
        label = label_map[example.label]

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        features.append(
            InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                input_mask=attention_mask,
                segment_ids=token_type_ids,
                label_ids=label,
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features



# Train and evaluate the model
def main(trainingdataset, testdataset, outputdir):
    

    torch.cuda.empty_cache()
    model = titleModel(model_type='bert', 
                    model_name='bert-base-multilingual-cased',
                    labels=['A', 'B', 'C', 'D'],
                    #use_cuda=False,
                    args={'save_model_every_epoch':False, 'save_steps': 10000,'output_dir':outputdir, 'evaluate_during_training':True,'overwrite_output_dir':True, 'classification_report': True, 'save_eval_checkpoints':False})
    
    model.train_model(trainingdataset,eval_df=testdataset)# Make sure eval_df is passed to the training method if enabled.
    

class titleModel:
    def __init__(
        self, model_type, model_name, labels=None, args=None, use_cuda=True, cuda_device=-1, **kwargs,
    ):
        """
        Initializes a NERModel
        Args:
            model_type: The type of model (bert, roberta)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            labels (optional): A list of all Named Entity labels.  If not given, ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"] will be used.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
        """  # noqa: ignore flake8"

        if args and 'manual_seed' in args:
            random.seed(args['manual_seed'])
            np.random.seed(args['manual_seed'])
            torch.manual_seed(args['manual_seed'])
            if 'n_gpu' in args and args['n_gpu'] > 0:
                torch.cuda.manual_seed_all(args['manual_seed'])

        if labels:
            self.labels = labels
        else:
            self.labels = ['A', 'B', 'C', 'D']
        self.num_labels = len(self.labels)
        self.output_mode = 'classification'
        MODEL_CLASSES = {
            "bert": (BertConfig, BertForMultipleChoice, BertTokenizer),
            "roberta": (RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer),
            "distilbert": (DistilBertConfig, DistilBertForMultipleChoice, DistilBertTokenizer),
            "camembert": (CamembertConfig, CamembertForMultipleChoice, CamembertTokenizer),
            "xlmroberta": (XLMRobertaConfig, XLMRobertaForMultipleChoice, XLMRobertaTokenizer),
        }

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.model_type=model_type
        self.model = model_class.from_pretrained(model_name, num_labels=self.num_labels, **kwargs)

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        self.results = {}

        self.args = {}
        self.args = {"classification_report": False}
        self.args.update(global_args)
        
        self.args['learning_rate']=2e-5
        self.args['num_train_epochs']=3
        self.args['train_batch_size']=32
        self.args['eval_batch_size']=32
        
        args["reprocess_input_data"]=False
        args["use_cached_eval_features"]=True

#         if not use_cuda:
        if True:
            self.args["fp16"] = False

        if args:
            self.args.update(args)

        self.tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=True, **kwargs)

        self.args["model_name"] = model_name
        self.args["model_type"] = model_type

        self.pad_token_label_id = CrossEntropyLoss().ignore_index

        if model_type == "camembert":
            warnings.warn(
                "use_multiprocessing automatically disabled as CamemBERT"
                " fails when using multiprocessing for feature conversion."
            )
            self.args["use_multiprocessing"] = False

        if self.args["wandb_project"] and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args["wandb_project"] = None



    def train_model(self, train_data,language_source,language_target,run=1, output_dir=None, show_running_loss=True, args=None, eval_df=None, verbose=True):
        """
        Trains the model using 'train_data'
        Args:
            train_data: train_data should be the path to a .txt file containing the training data OR a pandas DataFrame with 3 columns.
                        If a text file is given the data should be in the CoNLL format. i.e. One word per line, with sentences seperated by an empty line.
                        The first word of the line should be a word, and the last should be a Name Entity Tag.
                        If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.
            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
        Returns:
            None
        """  # noqa: ignore flake8"

        if args:
            self.args.update(args)

        if self.args["silent"]:
            show_running_loss = False

        if self.args["evaluate_during_training"] and eval_df is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_df is not specified."
                " Pass eval_df to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args["output_dir"]

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args["overwrite_output_dir"]:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Use --overwrite_output_dir to overcome.".format(output_dir)
            )

        self._move_model_to_device()

        train_dataset = self.load_and_cache_examples(train_data,language_source,run)

        os.makedirs(output_dir, exist_ok=True)

        results = self.train(
            train_dataset,language_source,language_target,run, output_dir, show_running_loss=show_running_loss, eval_df=eval_df
        )

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        logger.info(" Training of {} model complete. Saved to {}.".format(self.args["model_type"], output_dir))
        
        return results

    def train(self, train_dataset,language_source,language_target,run, output_dir, show_running_loss=True, eval_df=None, verbose=True):
        """
        Trains the model on train_dataset.
        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        device = self.device
        model = self.model
        args = self.args

        tb_writer = SummaryWriter()
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args["train_batch_size"])

        t_total = len(train_dataloader) // args["gradient_accumulation_steps"] * args["num_train_epochs"]

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args["weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        warmup_steps = math.ceil(t_total * args["warmup_ratio"])
        args["warmup_steps"] = warmup_steps if args["warmup_steps"] == 0 else args["warmup_steps"]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args["learning_rate"], eps=args["adam_epsilon"],)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args["warmup_steps"], num_training_steps=t_total
        )

        if args["fp16"]:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

#             model, optimizer = amp.initialize(model, optimizer, opt_level=args["fp16_opt_level"])
            model, optimizer = amp.initialize(model, optimizer)

        if args["n_gpu"] > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args["num_train_epochs"]), desc="Epoch", disable=args["silent"])
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0

        if args["evaluate_during_training"]:
            training_progress_scores = self._create_training_progress_scores()
        if args["wandb_project"]:
            wandb.init(project=args["wandb_project"], config={**args}, **args["wandb_kwargs"])
            wandb.watch(self.model)

        model.train()
        for _ in train_iterator:
            # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(tqdm(train_dataloader, desc="Current iteration", disable=args["silent"])):
                batch = tuple(t.to(device) for t in batch)

                inputs = self._get_inputs_dict(batch)
                outputs = model(**inputs)
                # model outputs are always tuple in pytorch-transformers (see doc)
                loss = outputs[0]

                if args["n_gpu"] > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()

                if show_running_loss:
                    print("\rRunning loss: %f" % loss, end="")

                if args["gradient_accumulation_steps"] > 1:
                    loss = loss / args["gradient_accumulation_steps"]

                if args["fp16"]:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(
                    #     amp.master_params(optimizer), args["max_grad_norm"]
                    # )
                else:
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(
                    #     model.parameters(), args["max_grad_norm"]
                    # )

                tr_loss += loss.item()
                if (step + 1) % args["gradient_accumulation_steps"] == 0:
                    if args["fp16"]:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args["max_grad_norm"])
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args["logging_steps"] > 0 and global_step % args["logging_steps"] == 0:
                        # Log metrics
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar(
                            "loss", (tr_loss - logging_loss) / args["logging_steps"], global_step,
                        )
                        logging_loss = tr_loss
#                         if args["wandb_project"]:
#                             wandb.log(
#                                 {
#                                     "Training loss": current_loss,
#                                     "lr": scheduler.get_lr()[0],
#                                     "global_step": global_step,
#                                 }
#                             )

                    if args["save_steps"] > 0 and global_step % args["save_steps"] == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        self._save_model(output_dir_current, model=model)

                    if args["evaluate_during_training"] and (
                        args["evaluate_during_training_steps"] > 0
                        and global_step % args["evaluate_during_training_steps"] == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _ ,accuracy_result= self.eval_model(eval_df,language_target,run, verbose=True)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        os.makedirs(output_dir_current, exist_ok=True)

                        if args["save_eval_checkpoints"]:
                            self._save_model(output_dir_current, model=model, results=results)

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        for key in results:
                            training_progress_scores[key].append(results[key])
                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(args["output_dir"], "training_progress_scores.csv"), index=False,
                        )

                        if args["wandb_project"]:
                            wandb.log(self._get_last_metrics(training_progress_scores))

                        if not best_eval_metric:
                            best_eval_metric = results[args["early_stopping_metric"]]
                            self._save_model(args["best_model_dir"], model=model, results=results)
                        if best_eval_metric and args["early_stopping_metric_minimize"]:
                            if results[args["early_stopping_metric"]] - best_eval_metric < args["early_stopping_delta"]:
                                best_eval_metric = results[args["early_stopping_metric"]]
                                self._save_model(args["best_model_dir"], model=model, results=results)
                                early_stopping_counter = 0
                            else:
                                if args["use_early_stopping"]:
                                    if early_stopping_counter < args["early_stopping_patience"]:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args['early_stopping_metric']}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args['early_stopping_patience']}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args['early_stopping_patience']} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return global_step, tr_loss / global_step
                        else:
                            if results[args["early_stopping_metric"]] - best_eval_metric > args["early_stopping_delta"]:
                                best_eval_metric = results[args["early_stopping_metric"]]
                                self._save_model(args["best_model_dir"], model=model, results=results)
                                early_stopping_counter = 0
                            else:
                                if args["use_early_stopping"]:
                                    if early_stopping_counter < args["early_stopping_patience"]:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args['early_stopping_metric']}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args['early_stopping_patience']}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args['early_stopping_patience']} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return global_step, tr_loss / global_step

            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))

            if args["save_model_every_epoch"] or args["evaluate_during_training"]:
                os.makedirs(output_dir_current, exist_ok=True)

            if args["save_model_every_epoch"]:
                self._save_model(output_dir_current, model=model)

            if args["evaluate_during_training"]:
                results, _, _,accuracy_result = self.eval_model(eval_df, language_target,run,verbose=True)

                self._save_model(output_dir_current, results=results)

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(os.path.join(args["output_dir"], "training_progress_scores.csv"), index=False)

                if not best_eval_metric:
                    best_eval_metric = results[args["early_stopping_metric"]]
                    self._save_model(args["best_model_dir"], model=model, results=results)
                if best_eval_metric and args["early_stopping_metric_minimize"]:
                    if results[args["early_stopping_metric"]] - best_eval_metric < args["early_stopping_delta"]:
                        best_eval_metric = results[args["early_stopping_metric"]]
                        self._save_model(args["best_model_dir"], model=model, results=results)
                        early_stopping_counter = 0
                else:
                    if results[args["early_stopping_metric"]] - best_eval_metric > args["early_stopping_delta"]:
                        best_eval_metric = results[args["early_stopping_metric"]]
                        self._save_model(args["best_model_dir"], model=model, results=results)
                        early_stopping_counter = 0

        return accuracy_result

    def eval_model(self, eval_data,lang,source_lang=None,run=1, output_dir=None, verbose=True,perturb=False,perturb_type=None,out_overlap=None):
        """
        Evaluates the model on eval_data. Saves results to output_dir.
        Args:
            eval_file: eval_data should be the path to a .txt file containing the evaluation data or a pandas DataFrame.
                        If a text file is used the data should be in the CoNLL format. I.e. One word per line, with sentences seperated by an empty line.
                        The first word of the line should be a word, and the last should be a Name Entity Tag.
                        If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.
            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
        Returns:
            result: Dictionary containing evaluation results. (eval_loss, precision, recall, f1_score)
            model_outputs: List of raw model outputs
            preds_list: List of predicted tags
        """  # noqa: ignore flake8"
        if not output_dir:
            output_dir = self.args["output_dir"]

        self._move_model_to_device()

        eval_dataset = self.load_and_cache_examples(eval_data,lang,source_lang,run,evaluate=True,perturb=perturb,perturb_type=perturb_type,out_overlap=out_overlap)

        result, model_outputs, preds_list,accuracy_result = self.evaluate(eval_dataset, output_dir)
        self.results.update(result)

        if verbose:
            logger.info(self.results)

        return result, model_outputs, preds_list,accuracy_result

    def evaluate(self, eval_dataset, output_dir):
        """
        Evaluates the model on eval_dataset.
        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        device = self.device
        model = self.model
        args = self.args
        pad_token_label_id = self.pad_token_label_id
        eval_output_dir = output_dir

        results = {}

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        model.eval()

        for batch in tqdm(eval_dataloader, disable=args["silent"]):
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }
                # XLM and RoBERTa don"t use segment_ids
                if args["model_type"] in ["bert", "xlnet"]:
                    inputs["token_type_ids"] = batch[2]
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        model_outputs = preds
        preds = np.argmax(preds, axis=1)

        label_map = {i: label for i, label in enumerate(self.labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]
        for i in range(out_label_ids.shape[0]):
            if out_label_ids[i] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i]])
                preds_list[i].append(label_map[preds[i]])

        result = {
            "eval_loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1_score": f1_score(out_label_list, preds_list),
        }

        results.update(result)
        
        
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            if args["classification_report"]:
                cls_report = classification_report(out_label_list, preds_list)
                writer.write("{}\n".format(cls_report))
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        #MAKE A TABEL FROM OUT LABEL LIST AND PREDS LIST

        return results, model_outputs, preds_list, result

    def predict(self, to_predict):
        """
        Performs predictions on a list of text.
        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction.
        Returns:
            preds: A Python list of lists with dicts containg each word mapped to its NER tag.
            model_outputs: A python list of the raw model outputs for each text.
        """

        device = self.device
        model = self.model
        args = self.args
        pad_token_label_id = self.pad_token_label_id

        self._move_model_to_device()

        predict_examples = [
            InputExample(i, sentence.split(), ["O" for word in sentence.split()])
            for i, sentence in enumerate(to_predict)
        ]

        eval_dataset = self.load_and_cache_examples(None,lang,run, to_predict=predict_examples)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        model.eval()

        for batch in tqdm(eval_dataloader, disable=args["silent"]):
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }
                # XLM and RoBERTa don"t use segment_ids
                if args["model_type"] in ["bert", "xlnet"]:
                    inputs["token_type_ids"] = batch[2]
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        model_outputs = preds
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(self.labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        preds = [
            [{word: preds_list[i][j]} for j, word in enumerate(sentence.split()[: len(preds_list[i])])]
            for i, sentence in enumerate(to_predict)
        ]

        return preds, model_outputs


    
    def load_and_cache_examples(self, examples,lang,source_lang=None,run=1, evaluate=False, no_cache=False, to_predict=None,perturb=False,perturb_type=None,out_overlap=None):
        """
        Reads data_file and generates a TensorDataset containing InputFeatures. Caches the InputFeatures.
        Utility function for train() and eval() methods. Not intended to be used directly.
        Args:
            data: Path to a .txt file containing training or evaluation data OR a pandas DataFrame containing 3 columns - sentence_id, words, labels.
                    If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.
            evaluate (optional): Indicates whether the examples are for evaluation or for training.
            no_cache (optional): Force feature conversion and prevent caching. I.e. Ignore cached features even if present.
        """  # noqa: ignore flake8"

        
        
        
        process_count = self.args["process_count"]

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args["no_cache"]

        mode = "dev" if evaluate else "train"

#         if not to_predict:
#             if isinstance(data, str):
#                 examples = read_examples_from_file(data, mode)
#             else:
#                 examples = get_examples_from_df(data)
#         else:
#             examples = to_predict
#             no_cache = True

        if perturb:
            pre_examples=copy.deepcopy(examples)
            
            if perturb_type=='cosine_text':
                out_overlap["all_O"]=0
                out_overlap["same_O"]=0
                same_file=open(path+lang+'/cosine_textword_with_'+source_lang+"_"+self.model_type+'1.txt','r').readlines()
                
                same_word=dict()
                for line in same_file:
                    line=line.strip().split("\t")
                    if len(line)==2:
                        same_word[line[0]]=line[1]

                for ex_id,example in enumerate(examples):
                    words=example.contexts[0].split(" ")
                    for w_i,word in enumerate(words):
                        out_overlap["all_O"]+=1
                        if word.lower() in same_word.keys():
                            out_overlap["same_O"]+=1
                            if word.lower()==word:
                                words[w_i]=same_word[word.lower()]
                            else:
                                words[w_i]=same_word[word.lower()].capitalize()

                    new_text=" ".join(words)
                    examples[ex_id]=MultipleChoiceExample(example_id=example.example_id,
                                        question=example.question,
                                        contexts=[new_text,new_text,new_text,new_text],
                                        endings=example.endings,
                                        label=example.label)
            
            if perturb_type=='random_text':
                out_overlap["all_O"]=0
                out_overlap["same_O"]=0
                
                same_file=open(path+lang+'/cosine_textword_with_'+source_lang+"_"+self.model_type+'1.txt','r').readlines()
                same_word=dict()
                for line in same_file:
                    line=line.strip().split("\t")
                    if len(line)==2:
                        same_word[line[0]]=line[1]

                unsame_file=open(path+lang+'/unsame_textword_with_'+source_lang+"_"+self.model_type++'1.txt','r').readlines()
              
                unsame_word_list=[]
                for line in unsame_file:
                    unsame_word_list.append(line.strip())
                    
                    
                for ex_id,example in enumerate(examples):
                    words=example.contexts[0].split(" ")
                    for w_i,word in enumerate(words):
                        out_overlap["all_O"]+=1
                        if word.lower() in same_word.keys():
                            
                            out_overlap["same_O"]+=1
                            if word.lower()==word:
                                words[w_i]=random.choice(unsame_word_list)
                            else:
                                words[w_i]=random.choice(unsame_word_list).capitalize()
                            
                    new_text=" ".join(words)
                    examples[ex_id]=MultipleChoiceExample(example_id=example.example_id,
                                    question=example.question,
                                    contexts=[new_text,new_text,new_text,new_text],
                                    endings=example.endings,
                                    label=example.label)

                    
            with open(path+lang+'/dataset_test_'+source_lang+'_'+perturb_type+'_'+self.model_type+str(run)+'.txt', 'w') as file:
                    for p_example,example in zip(pre_examples,examples):
                        file.write("previous example \n")
                        file.write(p_example.contexts[0]+"\n")                                                                 
                        file.write("new example \n")
                        file.write(example.contexts[0]+"\n")
        
#         cached_features_file = os.path.join(
#             args["cache_dir"],
#             "cached_title_{}_{}_{}_{}_{}".format(
#                 mode, args["model_type"],  lang,args["max_seq_length"],run
#             ),
#         )

        if not perturb:
            cached_features_file = os.path.join(
                args["cache_dir"],
                "cached_{}_{}_{}_{}_{}".format(
                    mode, args["model_type"],  lang,args["max_seq_length"],run
                ),
            )
        
        if perturb:
            cached_features_file = os.path.join(
                args["cache_dir"],
                "cached_{}_{}_{}_{}_{}_{}".format(
                    mode, args["model_type"],  lang,args["max_seq_length"],run,perturb_type
                ),
            )

#         print(cached_features_file)
        os.makedirs(self.args["cache_dir"], exist_ok=True)
        
        if os.path.exists(cached_features_file) and (
            (not args["reprocess_input_data"] and not no_cache) or (
                mode == "dev" and args["use_cached_eval_features"] and not no_cache)
        ):
            features = torch.load(cached_features_file)
            logger.info(f" Features loaded from cache at {cached_features_file}")
        else:
            logger.info(f" Converting to features started.")
            features = convert_multiple_choice_examples_to_features(
                    examples,
                    self.tokenizer,
                    max_length=self.args['max_seq_length'],
                    label_list=self.labels
            )

            if not no_cache:
                torch.save(features, cached_features_file)
                print("saved")
#         print(features)   
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids or 0 for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        return dataset

    

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def _get_inputs_dict(self, batch):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[3],
        }
        # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
        if self.args["model_type"] in ["bert", "xlnet", "albert"]:
            inputs["token_type_ids"] = batch[2]

        return inputs

    def _create_training_progress_scores(self):
        training_progress_scores = {
            "global_step": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "train_loss": [],
            "eval_loss": [],
        }

        return training_progress_scores

    def _save_model(self, output_dir, model=None, results=None):
        os.makedirs(output_dir, exist_ok=True)

        if model:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))


if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    parser.add_argument('trainingset', metavar='trainingset', type=str,
                    help='input path trainingset')
    parser.add_argument('testset', metavar='testset', type=str,
                    help='input path testset')
    parser.add_argument('outputdir', metavar='outputdir', type=str,
                    help='input path outputdir')
    
    args = parser.parse_args()
    
    main(args.trainingset, args.testset, args.outputdir) # can I make output directory optional?
   
