import os
import sys
import time
import json
import string
import argparse
import warnings
import torch
from torch import distributed as dist
from datetime import timedelta, datetime

# Change the HF cache directory to a scratch directory. 
# Make sure to set the variable before importing transformers module (including indirect import through galai).
# ref: https://github.com/paperswithcode/galai/blob/main/notebooks/Introduction%20to%20Galactica%20Models.ipynb
# os.environ["TRANSFORMERS_CACHE"] = "/scratch/ac.gpark/.cache/huggingface"
# os.environ["HF_HOME"] = "/scratch/ac.gpark/.cache/huggingface"

# ref: https://huggingface.co/docs/transformers/v4.21.1/en/troubleshooting#troubleshoot
#os.environ["CUDA_VISIBLE_DEVICES"] = "" # to run on CPU
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # to get a better traceback from the GPU error

from peft import PeftModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    BioGptForCausalLM,
    AutoModelForSeq2SeqLM,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    BitsAndBytesConfig,
)

from accelerate import Accelerator
from accelerate.utils import DistributedType

from data_processors import *
from evaluators import *



def get_rank():
    """
    Get the rank of this process in distributed processes.

    Return 0 for single process case.
    """
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0
    

def get_data_processor(*argv, **kwargs):
    data_name = argv[0]
    
    if data_name == "kbase":
        return KBaseProcessor(*argv, **kwargs)
    else:
        raise ValueError("Invalid data name: " + data_name)


def load_model(
    model_name,
    model_type, 
    max_new_tokens, 
    data_type,
    load_in_4bit_flag,
    load_in_8bit_flag,
    max_memory,
    low_cpu_mem_usage,
    quantization_config, 
    device_map,
    lora_finetune,
):
    if model_name == 'LLaMA-2':
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        tokenizer.padding_side = "left" # "right", left padding performs better than right padding and allow batched inference.
        tokenizer.truncation_side = "left" 

        tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
        
        generation_config = GenerationConfig.from_pretrained(model_type)
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.max_new_tokens = max_new_tokens
        generation_config.temperature = 1.0 # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        
        model = AutoModelForCausalLM.from_pretrained(
            model_type, 
            torch_dtype=data_type,
            load_in_4bit=load_in_4bit_flag,
            load_in_8bit=load_in_8bit_flag,
            max_memory=max_memory,
            low_cpu_mem_usage=low_cpu_mem_usage,
            quantization_config=quantization_config, 
            device_map=device_map
        )
        
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.config.pad_token_id = tokenizer.pad_token_id
            model.resize_token_embeddings(model.config.vocab_size + 1)
    
    elif model_name == 'LLaMA-3':
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        tokenizer.padding_side = "left" # "right", left padding performs better than right padding and allow batched inference.
        tokenizer.truncation_side = "left" 

        tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
        
        generation_config = GenerationConfig.from_pretrained(model_type)
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.max_new_tokens = max_new_tokens
        generation_config.temperature = 1.0 # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        
        model = AutoModelForCausalLM.from_pretrained(
            model_type, 
            torch_dtype=data_type,
            load_in_4bit=load_in_4bit_flag,
            load_in_8bit=load_in_8bit_flag,
            max_memory=max_memory,
            low_cpu_mem_usage=low_cpu_mem_usage,
            quantization_config=quantization_config, 
            device_map=device_map
        )
        
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.config.pad_token_id = tokenizer.pad_token_id
            model.resize_token_embeddings(model.config.vocab_size + 1)
    
    elif model_name == 'Mistral':
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.unk_token_id
        
        generation_config = GenerationConfig.from_pretrained(model_type)
        generation_config.pad_token_id = tokenizer.unk_token_id
        generation_config.max_new_tokens = max_new_tokens
        
        model = AutoModelForCausalLM.from_pretrained(
            model_type, 
            torch_dtype=data_type,
            load_in_4bit=load_in_4bit_flag,
            load_in_8bit=load_in_8bit_flag,
            max_memory=max_memory,
            low_cpu_mem_usage=low_cpu_mem_usage,
            quantization_config=quantization_config, 
            device_map=device_map
        )
    
    elif model_name == 'Solar':
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
        generation_config = GenerationConfig.from_pretrained(model_type)
        generation_config.pad_token_id = tokenizer.eos_token_id
        generation_config.max_new_tokens = max_new_tokens

        model = AutoModelForCausalLM.from_pretrained(
            model_type, 
            torch_dtype=data_type,
            load_in_4bit=load_in_4bit_flag,
            load_in_8bit=load_in_8bit_flag,
            max_memory=max_memory,
            low_cpu_mem_usage=low_cpu_mem_usage,
            quantization_config=quantization_config, 
            device_map=device_map
        )

    elif model_name == 'Falcon':
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
        generation_config = GenerationConfig.from_pretrained(model_type)
        generation_config.pad_token_id = tokenizer.eos_token_id
        generation_config.max_new_tokens = max_new_tokens
        
        model = AutoModelForCausalLM.from_pretrained(
            model_type, 
            torch_dtype=data_type,
            load_in_4bit=load_in_4bit_flag,
            load_in_8bit=load_in_8bit_flag,
            max_memory=max_memory,
            low_cpu_mem_usage=low_cpu_mem_usage,
            quantization_config=quantization_config, 
            device_map=device_map
        )
        
        model.config.pad_token_id = tokenizer.pad_token_id

    elif model_name == 'MPT':
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        if tokenizer.pad_token_id is None:
            warnings.warn(
                "pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id."
            )
        tokenizer.padding_side = "right" if lora_finetune else "left" # NotImplementedError: MPT does not support training with left padding.
        tokenizer.pad_token_id = tokenizer.eos_token_id

        generation_config = GenerationConfig.from_pretrained(model_type)
        generation_config.pad_token_id = tokenizer.eos_token_id
        generation_config.max_new_tokens = max_new_tokens
        generation_config.use_cache = True
        
        model = AutoModelForCausalLM.from_pretrained(
            model_type, 
            torch_dtype=data_type,
            load_in_4bit=load_in_4bit_flag,
            load_in_8bit=load_in_8bit_flag,
            max_memory=max_memory,
            low_cpu_mem_usage=low_cpu_mem_usage,
            quantization_config=quantization_config, 
            device_map=device_map,
            trust_remote_code=True, 
        )
        
        model.config.pad_token_id = tokenizer.eos_token_id
    
    else:
        raise ValueError("Invalid model name: " + model_name)	

    return model, tokenizer, generation_config


def get_args():
    """
    Get arguments.
    :return: Arguments namespace.
    :rtype: _AttributeHolder
    """
    parser = argparse.ArgumentParser()

    # general args
    parser.add_argument('--model_name', type=str)      # , required=True)
    parser.add_argument('--model_type', type=str)      # , required=True)
    parser.add_argument('--data_repo_path', type=str)  # , required=True)
    parser.add_argument('--output_dir', type=str)      # , required=True)
    parser.add_argument('--data_name', type=str)       # , required=True)
    parser.add_argument('--task', type=str)            # , required=True)
    # attributes which are set "required" will be provided within the config

    parser.add_argument('--test_sample_size', type=int, default=-1,
                        help="-1 means all data")
    parser.add_argument('--max_new_tokens', type=int, default=0,
                        help="the number of tokens to be generated by a model")
    parser.add_argument('--test_batch_size', type=int, default=32,
                        help="the number of samples to infer")
    parser.add_argument('--n_shots', type=int, default=1)

    # model/data specific args
    parser.add_argument('--num_of_kbase_classes', type=int, default=14)

    # PEFT (Parameter-Efficient Fine-tuning) args
    # 1. LoRA args
    # ref: https://github.com/huggingface/peft/blob/main/examples/lora_dreambooth/train_dreambooth.py
    parser.add_argument("--lora_weights", help="LoRA weights")
    parser.add_argument("--lora_finetune", action="store_true",
                        help="Whether to finetune a model using LoRA")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank, only used if lora_finetune is True")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha, only used if lora_finetune is True")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout, only used if lora_finetune is True")
    parser.add_argument("--lora_bias", type=str, default="none",
                        help="Bias type for LoRA. Can be 'none', 'all' or "
                             "'lora_only', only used if lora_finetune is True")
    parser.add_argument('--lora_output_dir', type=str)
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help="the number of samples to train")
    parser.add_argument('--validation_batch_size', type=int, default=32,
                        help="the number of samples to validate")

    parser.add_argument("--use_quantization", action="store_true",
                        help="Whether to use quantization or not")

    ## TODO: currently not used. parse this args.
    # Accelerator config (PATH TO: .cache/huggingface/accelerate/default_config.yaml)
    # will be used by a command "accelerate launch PYTHON script"
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
             "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
             "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true",
                        help="If passed, will train on the CPU.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every "
             "n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from "
             "the environment and use them for logging.",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default="logs",
        help="Location on where to store experiment tracking logs` and "
             "relevant project information",
    )

    return parser.parse_args(sys.argv[1:])


def main():

    args = get_args()

    model_name = args.model_name
    model_type = args.model_type
    data_repo_path = os.path.expanduser(args.data_repo_path)
    output_dir = os.path.expanduser(args.output_dir)
    data_name = args.data_name
    task = args.task
    test_sample_size = args.test_sample_size
    max_new_tokens = args.max_new_tokens
    test_batch_size = args.test_batch_size
    n_shots = args.n_shots
    num_of_kbase_classes = args.num_of_kbase_classes
    lora_weights = args.lora_weights
    lora_finetune = args.lora_finetune
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    lora_bias = args.lora_bias
    train_batch_size = args.train_batch_size
    validation_batch_size = args.validation_batch_size
    use_quantization = args.use_quantization

    output_dir = os.path.join(output_dir, model_name)
    output_dir = os.path.join(output_dir,
                              model_type.rsplit('/', 1)[1] \
                              if '/' in model_type else model_type)
    output_dir = os.path.join(output_dir, data_name)

    if get_rank() == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if lora_finetune:
        lora_output_dir = os.path.expanduser(args.lora_output_dir)
        lora_output_dir = os.path.join(lora_output_dir, model_type)
        lora_output_dir = os.path.join(lora_output_dir, data_name)
        lora_output_dir = os.path.join(lora_output_dir, task)
        
        if get_rank() == 0:
            if not os.path.exists(lora_output_dir):
                os.makedirs(lora_output_dir)
    
    # Sanity checks
    # Ensure that a task for a dataset is correct.
    if data_name == 'kbase':
        assert task in ['entity_type', 'entity_and_entity_type']
    
    # Set the max token length for a task.
    # relation and relation_type tasks only return a selection of multiple choices (e.g., True or False), so a longer length is not necessary. 
    if max_new_tokens == 0:
        if task == 'entity_type': # data: kbase
            max_new_tokens = 40
        elif task == 'entity_and_entity_type': # data: kbase
            max_new_tokens = 2000 # 500, 2000
        else:
            raise ValueError("Invalid task: " + task)
    
    # Set the model max length (context length -> (input + output) tokens) according to the model description.
    # - Mistral 7b - (ref: https://mistral.ai/news/announcing-mistral-7b/ 
    # - Mixtral 8x7b - Supports a context length of 32k tokens. (ref: https://huggingface.co/blog/mixtral, https://mistral.ai/product/)
    # - MPT-7B-Chat model can have up to 4096 tokens with the following configuration. (ref: https://huggingface.co/mosaicml/mpt-7b-chat)
    # 	>> config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
    # 	>> config.max_seq_len = 4096 # (input + output) tokens can now be up to 4096
    if model_name in ['Falcon'] or model_type == 'mosaicml/mpt-7b-chat':
        model_max_len = 2048
    elif model_name in ['LLaMA-2', 'Solar']:
        model_max_len = 4096
    elif model_type in ['mosaicml/mpt-30b-chat', 'mistralai/Mistral-7B-Instruct-v0.2'] or model_name in ['LLaMA-3']:
        model_max_len = 8192
    elif model_type == 'mistralai/Mixtral-8x7B-Instruct-v0.1':
        model_max_len = 32768 # 8192 * 4

    ## TODO: list not supported models. Galactica 30B, Falcon 40B, Llama2 70B, Mixtral-8x7B-Instruct-v0.1, MPT-30B, Solar (30B), RST, etc. 
    # check if a distributed environment is set up by the cmd "accelerate launch".
    accelerator = Accelerator()
    state = accelerator.state
    
    if state.distributed_type == DistributedType.NO:
        device_map = "auto" 
    else:
        device_map = None
    
    data_type = torch.bfloat16 if use_quantization else "auto" # torch.bfloat16, torch.float16,

    max_memory = {i: '80GiB' for i in range(torch.cuda.device_count())}
    
    load_in_4bit_flag = False
    load_in_8bit_flag = False
    low_cpu_mem_usage = True

    # Set quantization parameters.
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # load_in_8bit=True,
            bnb_4bit_compute_dtype=data_type,
            # bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", # defaults to "fp4"
        )
        
        load_in_4bit_flag = True

    quantization_config = bnb_config if use_quantization else None
    
    # Load a model, a tokenizer, and configure generation settings.
    model, tokenizer, generation_config = load_model(
                                            model_name,
                                            model_type, 
                                            max_new_tokens, 
                                            data_type,
                                            load_in_4bit_flag,
                                            load_in_8bit_flag,
                                            max_memory,
                                            low_cpu_mem_usage,
                                            quantization_config, 
                                            device_map,
                                            lora_finetune,
                                        )

    # Measure the processing time.
    if get_rank() == 0:
        st = time.time()

    data_processor = get_data_processor(
                        data_name,
                        data_repo_path,
                        task,
                        test_sample_size,
                        model_name,
                        tokenizer,
                        model_max_len,
                        generation_config,
                        num_of_kbase_classes=num_of_kbase_classes,
                    )
    
    is_training = lora_finetune
    
    data_processor.generate_datasets(n_shots, is_training)
    
    if lora_finetune:
        if model_type in ['meta-llama/Llama-2-70b-chat-hf', 'meta-llama/Meta-Llama-3-70B', 'meta-llama/Meta-Llama-3-70B-Instruct'] or model_name in ['Falcon', 'Solar']:
            data_processor.finetune(
                model,
                model_type,
                train_batch_size,
                validation_batch_size,
                lora_output_dir,
            )
            lora_output_dir = os.path.join(lora_output_dir, "final_checkpoint")
        
        else:
            data_processor.finetune_by_accelerator(
                model,
                model_type,
                train_batch_size,
                validation_batch_size,
                lora_output_dir,
            )
        
        base_model, _, _ = load_model(
                            model_name,
                            model_type, 
                            max_new_tokens, 
                            data_type,
                            load_in_4bit_flag,
                            load_in_8bit_flag,
                            max_memory,
                            low_cpu_mem_usage,
                            quantization_config, 
                            device_map,
                            lora_finetune,
                        )
        
        if model_name == 'MPT':
            tokenizer.padding_side = "left" # MPT is trained with right padding, so change it to left padding for inference.
            
        # Merge Model with Adapter
        model = PeftModel.from_pretrained(
            base_model, # The base model with full precision
            lora_output_dir, # Path to the finetuned adapter
        )

    elif lora_weights:
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
        )
    
    model.config.use_cache = True # enable for inference!
    
    model.eval()
    
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    if task == 'entity_and_entity_type':
        data_processor.infer(model, generation_config, test_batch_size)
    else:
        data_processor.infer_by_accelerator(model, "test", test_batch_size)
    
    results = data_processor.results
    
    if get_rank() == 0:
        et = time.time()
        elapsed_time = et - st
        exec_time = timedelta(seconds=elapsed_time)
        exec_time = str(exec_time)
        print('>> Execution time in hh:mm:ss:', exec_time)
        
        if task == "entity_and_entity_type":
            task_output_dir = os.path.join(output_dir, task)
            
            if not os.path.exists(task_output_dir):
                os.makedirs(task_output_dir)
            
            # get current date and time
            current_datetime = str(datetime.now())

            with open(os.path.join(task_output_dir, task + "_result_" + current_datetime + ".txt"), "w+") as fout:
                json.dump(results[task]['preprocessed'], fout)

        else:
            if len(results[task]['preprocessed']) == 3:
                src, pred, true = results[task]['preprocessed']
            else:
                src = None
                pred, true = results[task]['preprocessed']
            
            if hasattr(data_processor.data_reader, "rel_types"):
                labels = data_processor.data_reader.rel_types
            elif hasattr(data_processor.data_reader, "ent_types"):
                labels = data_processor.data_reader.ent_types
            elif task in ["relation", "entity_relation"]:
                labels = data_processor.relation_query_answers
            else:
                labels = None
            
            scores = compute_metrics(pred, true)
            save_results(
                scores=scores, 
                src=src,
                orig=results[task]['original'],
                pred=pred, 
                true=true,
                task=task, 
                labels=labels, 
                output_dir=output_dir,
                num_processes=state.num_processes,
                batch_size=test_batch_size,
                n_shots=n_shots,
                test_sample_size=len(data_processor.test_dataset),
                model_config=model.config,
                generation_config=generation_config,
                task_prompt=data_processor.task_prompt[task],
                data_name=data_name,
                num_of_kbase_classes=num_of_kbase_classes,
                exec_time=exec_time,
            )
            
        current_datetime = datetime.now()
        str_current_datetime = str(current_datetime)
        print('>> Current date and time:', str_current_datetime)

    
if __name__ == "__main__":
    main()
