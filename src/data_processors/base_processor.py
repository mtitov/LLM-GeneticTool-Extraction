import os
import sys
import re
import random
import torch, gc
import copy
import math
import logging

from abc import abstractmethod
from itertools import chain
from torch import distributed as dist
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm.auto import tqdm
from trl.trainer import ConstantLengthDataset
import warnings
        
from peft import (
    PeftModel, 
    PeftConfig, 
    LoraConfig, 
    TaskType,
    get_peft_config, 
    get_peft_model, 
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
    set_seed,
    default_data_collator,
    BitsAndBytesConfig,
    SchedulerType,
    get_scheduler,
)

from trl import SFTTrainer

from accelerate import Accelerator, DistributedType
from accelerate.utils import gather_object
from accelerate.logging import get_logger
        
# setting path
sys.path.append('../prompters')
from prompters import *

from evaluators import compute_metrics

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)


class BaseProcessor:
    def __init__(self, *argv):
        self.data_name = argv[0]
        self.data_repo_path = argv[1]
        self.task = argv[2]
        self.test_sample_size = argv[3]
        self.model_name = argv[4]
        self.tokenizer = argv[5]
        self.model_max_len = argv[6]
        self.generation_config = argv[7]
        
        self.model_prompt = self.get_model_prompt()		
        self.task_prompt = {}
        self.shot_samples = []
        
        self.relation_query_answers = ['yes', 'no']
        
        self.train_dataset = self.val_dataset = self.test_dataset = None
        
        self.results = {self.task: {'preprocessed': [], 'original': []}}


    @abstractmethod
    def generate_datasets(
        self,
        n_shots: int,
        is_training: False,
    ):
        raise NotImplementedError
    
    @abstractmethod
    def format_dataset(
        self, 
        dataset, 
        data_type
    ):
        raise NotImplementedError
        
    # deprecated.
    @abstractmethod
    def infer(
        self,
        model, 
        generation_config, 
        batch_size: int,
    ):
        raise NotImplementedError
    
    
    @abstractmethod
    def update_results(
        self,
        decoded_entity, 
        decoded_pred, 
        decoded_gold
    ):
        raise NotImplementedError
        
        
    def infer_by_accelerator(
        self,
        model, 
        data_type: str, # validation or test
        batch_size: int,
    ):
        task = self.task
        
        if data_type == "validation":
            eval_data = self.val_dataset
        elif data_type == "test":
            eval_data = self.test_dataset

        accelerator = Accelerator(cpu=False, mixed_precision=None)		
                
        # Apply the method we just defined to all the examples in all the splits of the dataset
        # starting with the main process first:
        with accelerator.main_process_first():
            formatted_datasets = self.format_dataset(eval_data, data_type)
        
        def collate_fn(examples):
            inputs = [x["text"] for x in examples]
            labels = [x["answer"] for x in examples]
            entities = [x["entity"] for x in examples]
            
            model_inputs = self.tokenizer(inputs, padding=True, return_tensors="pt")
            labels = self.tokenizer(labels, padding=True, return_tensors="pt")
            entities = self.tokenizer(entities, padding=True, return_tensors="pt")
            
            # Check if the input length doesn't exceed the model max input length.
            # assert self.model_max_len >= len(model_inputs["input_ids"][0]) + self.generation_config.max_new_tokens
        
            model_inputs["labels"] = labels["input_ids"]
            model_inputs["entities"] = entities["input_ids"]
            
            return model_inputs
            
        eval_dataloader = DataLoader(formatted_datasets, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)	

        # Prepare everything
        # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the prepare method.
        model, _, _, eval_dataloader, _ = accelerator.prepare(
            model, None, None, eval_dataloader, None
        )
        
        if accelerator.is_local_main_process:
            num_of_processed_samples = 0	
            
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                gen_kwargs = {}
                gen_kwargs["input_ids"] = batch["input_ids"]
                gen_kwargs["attention_mask"] = batch["attention_mask"]
                generated_tokens = accelerator.unwrap_model(model).generate(**gen_kwargs, generation_config=self.generation_config)
            
            if self.model_name == 'RST':
                pred_tokens = generated_tokens
            else:
                max_source_length = batch["input_ids"].shape[1]
                pred_tokens = generated_tokens[:, max_source_length :]
                
            pred_tokens = accelerator.pad_across_processes(pred_tokens, dim=1, pad_index=self.tokenizer.pad_token_id)
            gold_tokens = batch["labels"]
            entity_tokens = batch["entities"]

            gold_tokens = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=self.tokenizer.pad_token_id)
            entity_tokens = accelerator.pad_across_processes(batch["entities"], dim=1, pad_index=self.tokenizer.pad_token_id)

            entity_tokens, pred_tokens, gold_tokens = accelerator.gather_for_metrics((entity_tokens, pred_tokens, gold_tokens))
            entity_tokens, pred_tokens, gold_tokens = entity_tokens.cpu().numpy(), pred_tokens.cpu().numpy(), gold_tokens.cpu().numpy()

            decoded_entity = self.tokenizer.batch_decode(entity_tokens, skip_special_tokens=True)
            decoded_pred = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
            decoded_gold = self.tokenizer.batch_decode(gold_tokens, skip_special_tokens=True)
            
            self.update_results(decoded_entity, decoded_pred, decoded_gold)
                
            if accelerator.is_local_main_process:
                num_of_processed_samples += len(pred_tokens)
                accelerator.print(f">> the number of processed samples: {num_of_processed_samples} / total samples: {len(eval_data)}")


    def finetune(
        self,
        model,
        model_type: str,
        train_batch_size: int,
        validation_batch_size: int,
        output_dir: str,
    ):
        set_seed(42)
                
        # wandb params
        wandb_project = "LLM-GeneticTool-Extraction"
        wandb_run_name = model_type #run_name="bert-base-high-lr",  # name of the W&B run (optional)
        # wandb_run_name = model_type + "_4" #run_name="bert-base-high-lr",  # name of the W&B run (optional)
        wandb_watch = "all"  # options: false | gradients | all
        wandb_log_model = ""  # options: false | true
        
        # Check if parameter passed or if set within environ
        # use_wandb = len(wandb_project) > 0 or (
            # "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
        # )
        # Only overwrite environ if wandb param passed
        if len(wandb_project) > 0:
            os.environ["WANDB_PROJECT"] = wandb_project
        if len(wandb_watch) > 0:
            os.environ["WANDB_WATCH"] = wandb_watch
        if len(wandb_log_model) > 0:
            os.environ["WANDB_LOG_MODEL"] = wandb_log_model

        if self.model_name in ['LLaMA-2', 'LLaMA-3', 'Solar'] or model_type == 'mistralai/Mistral-7B-Instruct-v0.2':
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "lm_head"] # If targeting all linear layers
            
        elif model_type == 'mistralai/Mixtral-8x7B-Instruct-v0.1':
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3", "lm_head"]
            
        elif self.model_name == 'Falcon':
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            
        elif self.model_name == 'MPT':
            target_modules = ["Wqkv", "out_proj", "up_proj", "down_proj"]
        
        lora_config = LoraConfig(
            r=8, # 8, 16
            lora_alpha=16, # 8, 16, 32
            lora_dropout=0.05, # 0, 0.05
            fan_in_fan_out=False, # False (default)
            inference_mode=False, # False (default)
            bias="none", # "none" (default) Note that it's a string not None.
            target_modules=target_modules,
            task_type=TaskType.SEQ_2_SEQ_LM if self.model_name == 'RST' else TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        
        training_args = TrainingArguments(
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=train_batch_size, # options: batch_size, micro_batch_size
            per_device_eval_batch_size=validation_batch_size, # options: batch_size, micro_batch_size
            gradient_accumulation_steps=1, # options: 1 (default), 4, 8, gradient_accumulation_steps
            num_train_epochs=5,
            learning_rate=1e-4, # 5e-05 (default), 1e-4, 2e-4, 2e-5, 3e-4
            warmup_steps=100, # 5, 10, 50, 100, 400
            # max_steps=500,
            optim="paged_adamw_8bit", # "adamw_torch" (default), "adamw_8bit", "paged_adamw_8bit", "paged_adamw_32bit"
            # weight_decay=0.01, # 0 (default)
            # fp16=True, # False (default) -> True causes an error for Llama model (loss is zero).
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10, # 1, 10
            save_strategy="steps", # "epoch", "steps"
            eval_steps=10, # 1, 10, 200
            save_steps=10, # 1, 10, 200
            save_total_limit=3,
            load_best_model_at_end=True, # False (default)
            metric_for_best_model="eval_loss",
            evaluation_strategy="steps", # "epoch", "steps"
            output_dir=output_dir,
            overwrite_output_dir=True,
            report_to="wandb",
            run_name=wandb_run_name,
        )
        
        train_dataset = self.format_dataset(self.train_dataset, "train")
        val_dataset = self.format_dataset(self.val_dataset, "validation")

        def preprocess_function(examples):
            inputs = examples["text"]
            answer = examples["answer"]

            model_inputs = self.tokenizer(inputs, padding=True)

            labels = copy.deepcopy(model_inputs)
            
            ## TODO: label length is not correct. Fix it. Get ignore_pad_token_for_loss as an argument.
            ignore_pad_token_for_loss = True
            '''
            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore padding in the loss.
            if ignore_pad_token_for_loss:
                # get the length of the target tokens. -1 to kick out the <BOS> token
                label_tokens = self.tokenizer(answer, padding=False)
                label_len = [len(label) - 1 for label in label_tokens["input_ids"]]
                # label_len = [len(label) for label in label_tokens["input_ids"]]

                # don't calculate the loss from source and padding (left padding)
                for i in range(len(labels["input_ids"])):
                    
                    # debug
                    print(labels["input_ids"][i])
                    decoded_string = self.tokenizer.decode(labels["input_ids"][i])
                    print('>> labels:', decoded_string)

                    # labels["input_ids"][i, : -label_len[i]] = -100 # Error!!
                    for j in range(len(labels["input_ids"][i]) - label_len[i]):
                        labels["input_ids"][i][j] = -100
                    
                    # debug
                    print(labels["input_ids"][i])
                    decoded_string = self.tokenizer.decode(label_tokens["input_ids"][i])
                    print('>> label_tokens:', decoded_string)
                    print('>> decoded_token:', self.tokenizer.decode(28747))
                    print('>> decoded_token:', self.tokenizer.decode(2652))
                    print('>> decoded_token:', self.tokenizer.decode(525))
                    print('>> decoded_token:', self.tokenizer.decode(13320))
                    print('>> decoded_token:', self.tokenizer.decode(13))
                    print('>> decoded_token:', self.tokenizer.decode(13))
                    input('enter..')
            '''
            # this is a temporary code until the code above is fixed. 04-10-2024
            if ignore_pad_token_for_loss:
                # don't calculate the loss from padding (left padding)
                for i in range(len(labels["input_ids"])):
                    for j in range(len(labels["input_ids"][i])):
                        if labels["input_ids"][i][j] == 0:
                            labels["input_ids"][i][j] = -100

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # dataset = dataset.map(
            # preprocess_function,
            # batched=True,
        # )
        
        # dataset = dataset.train_test_split(test_size=test_sample_size//2, shuffle=True)
        
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
        )
        
        val_dataset = val_dataset.map(
            preprocess_function,
            batched=True,
        )
        
        class CustomizedConstantLengthDataset(ConstantLengthDataset):
            def __init__(
                self,
                tokenizer,
                dataset,
                dataset_text_field=None,
                formatting_func=None,
                infinite=False,
                seq_length=1024,
                num_of_sequences=1024,
                chars_per_token=3.6,
                eos_token_id=0,
                shuffle=True,
                append_concat_token=True,
                add_special_tokens=True,
            ):
                self.tokenizer = tokenizer

                if tokenizer.eos_token_id is None:
                    warnings.warn(
                        "The passed tokenizer does not have an EOS token. We will use the passed eos_token_id instead which corresponds"
                        f" to {eos_token_id}. If this is not the correct EOS token, make sure to pass the correct eos_token_id."
                    )

                self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else eos_token_id
                self.dataset = dataset
                self.seq_length = seq_length
                self.infinite = infinite
                self.current_size = 0
                self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
                self.shuffle = shuffle
                self.append_concat_token = append_concat_token
                self.add_special_tokens = add_special_tokens

                if formatting_func is not None:
                    if formatting_func.__code__.co_argcount > 1:
                        warnings.warn(
                            "The passed formatting_func has more than one argument. Usually that function should have a single argument `example`"
                            " which corresponds to the dictionary returned by each element of the dataset. Make sure you know what you are doing."
                        )
            
            def __iter__(self):
                iterator = iter(self.dataset)

                more_examples = True
                while more_examples:
                    buffer, buffer_len = [], 0
                    while True:
                        if buffer_len >= self.max_buffer_size:
                            break
                        try:
                            buffer.append(next(iterator))
                            buffer_len += len(buffer[-1])
                        except StopIteration:
                            if self.infinite:
                                iterator = iter(self.dataset)
                                warnings.warn("The dataset reached end and the iterator is reset to the start.")
                            else:
                                more_examples = False
                                break

                    tokenized_inputs = [x["input_ids"] for x in buffer]
                    tokenized_labels = [x["labels"] for x in buffer]

                    all_input_token_ids = []
                    all_label_token_ids = []
                    for tokenized_input, tokenized_label in zip(tokenized_inputs, tokenized_labels):
                        if self.append_concat_token:
                            tokenized_input = tokenized_input + [self.concat_token_id]
                            tokenized_label = tokenized_label + [self.concat_token_id]
                        all_input_token_ids.extend(tokenized_input)
                        all_label_token_ids.extend(tokenized_label)
                    examples = []
                    for i in range(0, len(all_input_token_ids), self.seq_length):
                        input_ids = all_input_token_ids[i : i + self.seq_length]
                        label_ids = all_label_token_ids[i : i + self.seq_length]
                        if len(input_ids) == self.seq_length:
                            examples.append((input_ids, label_ids))
                    if self.shuffle:
                        random.shuffle(examples)
                    for example in examples:
                        self.current_size += 1
                        yield {
                            "input_ids": torch.LongTensor(example[0]),
                            "labels": torch.LongTensor(example[1]),
                        }
        
        train_seq_length = len(train_dataset["input_ids"][0]) + self.generation_config.max_new_tokens
        val_seq_length = len(val_dataset["input_ids"][0]) + self.generation_config.max_new_tokens
        
        train_dataset = CustomizedConstantLengthDataset(
            self.tokenizer,
            train_dataset,
            infinite=False,
            shuffle=False,
            seq_length=train_seq_length, # 1024 (default)
        )

        val_dataset = CustomizedConstantLengthDataset(
            self.tokenizer,
            val_dataset,
            infinite=False,
            shuffle=False,
            seq_length=val_seq_length, # 1024 (default)
        )
        
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            packing=True,
            tokenizer=self.tokenizer,
            peft_config=lora_config,
            dataset_text_field="text",
            max_seq_length=self.model_max_len,
        )
        
        model.config.use_cache = False # silence the warnings. Please re-enable for inference!

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        
        trainer.train()
        trainer.model.save_pretrained(os.path.join(output_dir, "final_checkpoint"))

        del model
        del trainer
        torch.cuda.empty_cache()


    def finetune_by_accelerator(
        self,
        model,
        model_type: str,
        train_batch_size: int,
        validation_batch_size: int,
        output_dir: str,
    ):
        model.config.use_cache = False # silence the warnings. Please re-enable for inference!
        
        logger = get_logger(__name__)

        task = self.task

        if self.model_name in ['LLaMA-2', 'LLaMA-3', 'Solar'] or model_type == 'mistralai/Mistral-7B-Instruct-v0.2':
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "lm_head"] # If targeting all linear layers

        elif model_type == 'mistralai/Mixtral-8x7B-Instruct-v0.1':
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3", "lm_head"]
            
        elif self.model_name == 'Falcon':
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            
        elif self.model_name == 'MPT':
            target_modules = ["Wqkv", "out_proj", "up_proj", "down_proj"]
        
        lora_config = LoraConfig(
            r=8, # 8, 16
            lora_alpha=16, # 8, 16, 32
            lora_dropout=0.05, # 0, 0.05
            fan_in_fan_out=False, # False (default)
            inference_mode=False, # False (default)
            bias="none", # "none" (default) Note that it's a string not None.
            target_modules=target_modules,
            task_type=TaskType.SEQ_2_SEQ_LM if self.model_name == 'RST' else TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)

        accelerator = Accelerator(cpu=False, mixed_precision="bf16", log_with="wandb") # mixed_precision=None
        
        # Set the training seed for reproducible training.
        set_seed(42)
        
        accelerator.wait_for_everyone()
        
        def preprocess_function_train(examples):
            inputs = examples["text"]
            answer = examples["answer"]

            model_inputs = self.tokenizer(inputs, padding=True)

            labels = copy.deepcopy(model_inputs)
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        
        def preprocess_function_test(examples):
            inputs = examples["text"]
            labels = examples["answer"]
            
            model_inputs = self.tokenizer(inputs, padding=True)
            labels = self.tokenizer(labels, padding=True)

            model_inputs["labels"] = labels["input_ids"]

            return model_inputs
        
        with accelerator.main_process_first():
            train_dataset = self.format_dataset(self.train_dataset, "train")
            # val_dataset = self.format_dataset(self.val_dataset, "validation")
        
        def collate_fn(examples):
            inputs = [x["text"] for x in examples]
            labels = [x["answer"] for x in examples]
            
            model_inputs = self.tokenizer(inputs, padding=True, return_tensors="pt")
            labels = self.tokenizer(labels, padding=True, return_tensors="pt")
                        
            # Check if the input length doesn't exceed the model max input length.
            assert self.model_max_len >= len(model_inputs["input_ids"][0]) + self.generation_config.max_new_tokens
        
            model_inputs["labels"] = labels["input_ids"]
            
            return model_inputs
        
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size)	
        # val_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_fn, batch_size=validation_batch_size)	

        ## TODO: set training arguments.
        # weight_decay, learning_rate, gradient_accumulation_steps, max_train_steps, num_train_epochs, num_warmup_steps 
        weight_decay = 0 # 0, 0.01 (default)
        learning_rate = 1e-4 # 5e-05 (default), 1e-4, 2e-4, 2e-5, 3e-4
        gradient_accumulation_steps = 1 # options: 1 (default), 4, 8, gradient_accumulation_steps
        max_train_steps = None
        num_train_epochs = 5 # 5, 10
        num_warmup_steps = 100 # 5, 10, 50, 100, 400
        lr_scheduler_type = "linear" 
        per_device_train_batch_size = train_batch_size
        checkpointing_steps = "epoch"
        
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "lora" in n],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        if max_train_steps is None:
            max_train_steps = num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps * gradient_accumulation_steps,
            num_training_steps=max_train_steps * gradient_accumulation_steps,
        )
        
        # model.to(accelerator.device)
        
        # Prepare everything with our `accelerator`.
        # model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            # model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        # )
        model, optimizer, train_dataloader, _, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, None, lr_scheduler
        )
        
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        if overrode_max_train_steps:
            max_train_steps = num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
        
        # Figure out how many steps we should save the Accelerator states
        # checkpointing_steps = args.checkpointing_steps
        if checkpointing_steps is not None and checkpointing_steps.isdigit():
            checkpointing_steps = int(checkpointing_steps)
        
        # Initialise the wandb run, passing wandb parameters and any config information
        accelerator.init_trackers(
            project_name="LLM-GeneticTool-Extraction",
            config={
                "learning_rate": learning_rate, 
                "weight_decay": weight_decay, 
                "num_warmup_steps": num_warmup_steps, 
                "lr_scheduler_type": lr_scheduler_type, 
                "per_device_train_batch_size": per_device_train_batch_size,
            },
            init_kwargs={"wandb": {"name": model_type}}
        )
        
        # Train!
        total_batch_size = per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0
        
        ## TODO: complete this later.
        '''
        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                checkpoint_path = args.resume_from_checkpoint
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
                checkpoint_path = path
                path = os.path.basename(checkpoint_path)

            accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
            accelerator.load_state(path)
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
                completed_steps = starting_epoch * num_update_steps_per_epoch
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)
                completed_steps = resume_step // args.gradient_accumulation_steps
        '''

        # update the progress_bar if load from checkpoint
        progress_bar.update(completed_steps)

        for epoch in range(starting_epoch, num_train_epochs):
            model.train()
            
            train_total_loss = 0
            
            ## TODO: complete this later.
            '''
            if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            else:
            '''
            active_dataloader = train_dataloader
            
            for step, batch in enumerate(active_dataloader):
                if self.model_name == 'Galactica':
                    del batch["token_type_ids"]
                
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss

                    train_total_loss += loss.detach().float()
                    
                    accelerator.backward(loss)
                    if completed_steps % 50:
                        accelerator.print(f"Epoch: {epoch} | Step: {completed_steps} | Loss: {loss}")
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        checkpointing_output_dir = f"step_{completed_steps}"
                        if output_dir is not None:
                            checkpointing_output_dir = os.path.join(output_dir, checkpointing_output_dir)
                        accelerator.save_state(checkpointing_output_dir)
                
                if completed_steps >= max_train_steps:
                    break
            
            model.eval()
            
            if self.model_name == 'MPT':
                self.tokenizer.padding_side = "left" # MPT is trained with right padding, so change it to left padding for inference.
            
            self.infer_by_accelerator(model, "validation", validation_batch_size)
            
            if len(self.results[task]['preprocessed']) == 3:
                # store the source item in the query to be used in entity_relation task for STRING, KEGG. 04/12/2023
                src, pred, true = self.results[task]['preprocessed']
            else:
                src = None
                pred, true = self.results[task]['preprocessed']

            scores = compute_metrics(pred, true)

            # reset the result dict.
            self.results[task]['preprocessed'] = []
            
            if self.model_name == 'MPT':
                self.tokenizer.padding_side = "right" # MPT is trained with right padding.
            
            if checkpointing_steps == "epoch":
                checkpointing_output_dir = f"epoch_{epoch}"
                if output_dir is not None:
                    checkpointing_output_dir = os.path.join(output_dir, checkpointing_output_dir)
                accelerator.save_state(checkpointing_output_dir)

            # Log to wandb by calling `accelerator.log`, `step` is optional
            accelerator.log(
                {
                    "train_loss": train_total_loss.item() / len(train_dataloader),
                    # "accuracy": accuracy,
                    "valid_micro f1": scores["micro_f"],
                    "valid_macro f1": scores["macro_f"],
                    # "validation_loss": val_total_loss.item() / len(val_dataloader),
                    "epoch": epoch + 1,
                },
                step=completed_steps,
            )

        accelerator.end_training()
        
        if output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                self.tokenizer.save_pretrained(output_dir)


    def get_response(self, model, generation_config, batch_input_texts, return_full_text=False):
        inputs = self.tokenizer(batch_input_texts, padding=True, return_tensors="pt").to(device)
        
        input_ids = inputs["input_ids"]
        
        generated_sequence = model.generate(input_ids=input_ids, generation_config=generation_config)
        
        generated_text = self.tokenizer.batch_decode(generated_sequence, skip_special_tokens=True)
        
        if not return_full_text: 
            new_text = []
            for input_id, text in zip(input_ids, generated_text):
                prompt_length = len(self.tokenizer.decode(input_id, skip_special_tokens=True))
                new_text.append(text[prompt_length:].strip())
            generated_text = new_text

        gc.collect()
        torch.cuda.empty_cache()
        
        return generated_text
    
    
    def clean_response(
        self, 
        response, 
        true=None, 
        entity=None
    ):
        """
        Remove a prompt and unnecessary texts in model's generated texts.
        
        """
        
        choices = self.data_reader.ent_types

        cleaned_response = 'None' # models (e.g., LLaMA) sometimes generate nothing. In this case, put in 'None'.
        
        entity = entity.lower()
        response = response.lower()
        response = response.replace(entity, '')
        
        c_list = [] # debug - to check if a response has multiple choices.
        for c in choices:
            if c.lower() in response:
                c_list.append(c.lower())
        
        if len(c_list) == 1:
            cleaned_response = c_list[0]
        elif len(c_list) > 1: # debug - to check if a response has multiple choices.
            for i in c_list:
                if i == true:
                    cleaned_response = i
                    break
            
        return cleaned_response

    
    def get_model_prompt(self):
        if self.model_name == 'LLaMA-2':
            if self.data_name == 'kbase':
                return Llama2Prompter.get_kbase_prompt(self)
        
        elif self.model_name == 'LLaMA-3':
            if self.data_name == 'kbase':
                return Llama3Prompter.get_kbase_prompt(self)
                
        elif self.model_name == 'Falcon':
            if self.data_name == 'kbase':
                return FalconPrompter.get_kbase_prompt(self)
        
        elif self.model_name == 'MPT':
            if self.data_name == 'kbase':
                return MptPrompter.get_kbase_prompt(self)
        
        elif self.model_name == 'Mistral':
            if self.data_name == 'kbase':
                return MistralPrompter.get_kbase_prompt(self)
            
        elif self.model_name == 'Solar':
            if self.data_name == 'kbase':
                return SolarPrompter.get_kbase_prompt(self)
        else:
            raise ValueError("Invalid model name: " + self.model_name)
    
    
    def sort_and_pad(self, pred, true, max_entity_list_len=10):
        common_values = list(set(pred) & set(true))
        new_pred = common_values + list(set(pred) - set(common_values))
        new_true = common_values + list(set(true) - set(common_values))

        if len(new_pred) > max_entity_list_len:
            new_pred = new_pred[:max_entity_list_len]
            
        if len(new_true) > max_entity_list_len:
            new_true = new_true[:max_entity_list_len]
        
        new_pred_len = len(new_pred)
        new_true_len = len(new_true)
        
        diff = abs(new_pred_len - new_true_len)

        # padding number of elements to the end of the list
        if new_pred_len < new_true_len:
            new_pred += ['NONE'] * diff

        return new_pred, new_true
    
    
    def get_rank(self):
        """
        Get the rank of this process in distributed processes.

        Return 0 for single process case.
        """
        if dist.is_initialized():
            return dist.get_rank()
        if "RANK" in os.environ:
            return int(os.environ["RANK"])
        return 0
