#!/bin/bash

BASE_DIR=${GT_BASE_DIR:-/home/matitov/gllm}
SCRATCH_DIR=${GT_SCRATCH_DIR:-/eagle/RECUP/matitov}

PS3='Please enter your choice: '
options=("Run LLaMA-2"
         "Run LLaMA-3"
         "Run Mistral"
         "Run Solar"
         "Run Falcon"
         "Run MPT"
         "Quit")
select opt in "${options[@]}"
do
    export DATA_REPO_PATH="${BASE_DIR}/BioIE-LLM-WIP/data"
    export OUTPUT_DIR="${BASE_DIR}/BioIE-LLM-WIP/result"
    export LORA_OUTPUT_DIR="${SCRATCH_DIR}/LoRA_finetuned_models"
    
    case $opt in
        "Run LLaMA-2")
            echo "you chose Run LLaMA-2."
            
            export MODEL_NAME=LLaMA-2
            export MODEL_TYPE=meta-llama/Llama-2-7b-chat-hf
            # export MODEL_TYPE=meta-llama/Llama-2-70b-chat-hf
            export DATA_NAME=kbase
            export TASK=entity_type # entity_type (kbase), entity_and_entity_type (kbase)
            export TRAIN_BATCH_SIZE=2 # used in finetuning
            export VALIDATION_BATCH_SIZE=8 # used in finetuning
            export TEST_BATCH_SIZE=8
            export N_SHOTS=0
            export LORA_WEIGHTS="${LORA_OUTPUT_DIR}/meta-llama/Llama-2-7b-chat-hf/kbase/entity_type/final_checkpoint"
            
            # python ~/BioIE-LLM-WIP/src/run_model.py \
            # torchrun --nproc_per_node=4 ~/BioIE-LLM-WIP/src/run_model.py \
            # accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
            
            accelerate launch "${BASE_DIR}/BioIE-LLM-WIP/src/run_model.py" \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --train_batch_size $TRAIN_BATCH_SIZE \
                --validation_batch_size $VALIDATION_BATCH_SIZE \
                --test_batch_size $TEST_BATCH_SIZE \
                --n_shots $N_SHOTS
                
                
            : '
                --lora_finetune \
                --use_quantization \
                --lora_output_dir $LORA_OUTPUT_DIR
                --lora_weights $LORA_WEIGHTS
            '
            
            break
            ;;
        "Run LLaMA-3")
            echo "you chose Run LLaMA-3."
            
            export MODEL_NAME=LLaMA-3
            # export MODEL_TYPE=meta-llama/Meta-Llama-3-8B
            export MODEL_TYPE=meta-llama/Meta-Llama-3-70B
            export DATA_NAME=kbase
            export TASK=entity_type # entity_type (kbase), entity_and_entity_type (kbase)
            export TRAIN_BATCH_SIZE=2 # used in finetuning
            export VALIDATION_BATCH_SIZE=8 # used in finetuning
            export TEST_BATCH_SIZE=8
            export N_SHOTS=0
            export LORA_WEIGHTS="${LORA_OUTPUT_DIR}/meta-llama/Llama-3-8B/kbase/entity_type/final_checkpoint"
            
            # python ~/BioIE-LLM-WIP/src/run_model.py \
            # torchrun --nproc_per_node=4 ~/BioIE-LLM-WIP/src/run_model.py \
            # accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
            
            python "${BASE_DIR}/BioIE-LLM-WIP/src/run_model.py" \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --train_batch_size $TRAIN_BATCH_SIZE \
                --validation_batch_size $VALIDATION_BATCH_SIZE \
                --test_batch_size $TEST_BATCH_SIZE \
                --n_shots $N_SHOTS \
                --lora_finetune \
                --use_quantization \
                --lora_output_dir $LORA_OUTPUT_DIR
                
            
            : '
                --lora_finetune \
                --use_quantization \
                --lora_output_dir $LORA_OUTPUT_DIR
                --lora_weights $LORA_WEIGHTS
            '
            
            break
            ;;
        "Run Mistral")
            echo "you chose Run Mistral."
            
            export MODEL_NAME=Mistral
            export MODEL_TYPE=mistralai/Mistral-7B-Instruct-v0.2
            # export MODEL_TYPE=mistralai/Mixtral-8x7B-Instruct-v0.1
            export DATA_NAME=kbase
            export TASK=entity_type # entity_type (kbase), entity_and_entity_type (kbase)
            export TRAIN_BATCH_SIZE=2 # used in finetuning
            export VALIDATION_BATCH_SIZE=8 # used in finetuning
            export TEST_BATCH_SIZE=8
            export N_SHOTS=0
            export LORA_WEIGHTS="${LORA_OUTPUT_DIR}/mistralai/Mistral-7B-Instruct-v0.2/kbase/entity_type"
            # export LORA_WEIGHTS=/scratch/ac.gpark/LoRA_finetuned_models/mistralai/Mixtral-8x7B-Instruct-v0.1/kbase/entity_type
            
            
            # python ~/BioIE-LLM-WIP/src/run_model.py \
            # torchrun --nproc_per_node=4 ~/BioIE-LLM-WIP/src/run_model.py \
            # accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
            
            accelerate launch "${BASE_DIR}/BioIE-LLM-WIP/src/run_model.py" \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --train_batch_size $TRAIN_BATCH_SIZE \
                --validation_batch_size $VALIDATION_BATCH_SIZE \
                --test_batch_size $TEST_BATCH_SIZE \
                --n_shots $N_SHOTS \
                --lora_finetune \
                --use_quantization \
                --lora_output_dir $LORA_OUTPUT_DIR
            
            : '
                --lora_finetune \
                --use_quantization \
                --lora_output_dir $LORA_OUTPUT_DIR
                --lora_weights $LORA_WEIGHTS
            '
            
            break
            ;;
        "Run Solar")
            echo "you chose Run Solar."
            
            export MODEL_NAME=Solar
            export MODEL_TYPE=upstage/SOLAR-10.7B-Instruct-v1.0
            export DATA_NAME=kbase
            export TASK=entity_type # entity_type (kbase), entity_and_entity_type (kbase)
            export TRAIN_BATCH_SIZE=2 # used in finetuning
            export VALIDATION_BATCH_SIZE=8 # used in finetuning
            export TEST_BATCH_SIZE=8
            export N_SHOTS=0
            export LORA_WEIGHTS="${LORA_OUTPUT_DIR}/tiiuae/falcon-7b/kbase/entity_type/final_checkpoint"

            # python ~/BioIE-LLM-WIP/src/run_model.py \
            # torchrun --nproc_per_node=4 ~/BioIE-LLM-WIP/src/run_model.py \
            # accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
            
            python "${BASE_DIR}/BioIE-LLM-WIP/src/run_model.py" \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --train_batch_size $TRAIN_BATCH_SIZE \
                --validation_batch_size $VALIDATION_BATCH_SIZE \
                --test_batch_size $TEST_BATCH_SIZE \
                --n_shots $N_SHOTS \
                --lora_finetune \
                --use_quantization \
                --lora_output_dir $LORA_OUTPUT_DIR
            
            : '
                --lora_finetune \
                --lora_output_dir $LORA_OUTPUT_DIR
                --use_quantization \
                --lora_weights $LORA_WEIGHTS
            '
            
            break
            ;;
        "Run Falcon")
            echo "you chose Run Falcon."
            
            export MODEL_NAME=Falcon
            # export MODEL_TYPE=tiiuae/falcon-7b
            export MODEL_TYPE=tiiuae/falcon-40b
            export DATA_NAME=kbase
            export TASK=entity_type # entity_type (kbase), entity_and_entity_type (kbase)
            export TRAIN_BATCH_SIZE=2 # used in finetuning
            export VALIDATION_BATCH_SIZE=8 # used in finetuning
            export TEST_BATCH_SIZE=8
            export N_SHOTS=0
            export LORA_WEIGHTS="${LORA_OUTPUT_DIR}/tiiuae/falcon-7b/kbase/entity_type/final_checkpoint"
            
            # python ~/BioIE-LLM-WIP/src/run_model.py \
            # torchrun --nproc_per_node=4 ~/BioIE-LLM-WIP/src/run_model.py \
            # accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
            
            python "${BASE_DIR}/BioIE-LLM-WIP/src/run_model.py" \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --train_batch_size $TRAIN_BATCH_SIZE \
                --validation_batch_size $VALIDATION_BATCH_SIZE \
                --test_batch_size $TEST_BATCH_SIZE \
                --n_shots $N_SHOTS \
                --lora_finetune \
                --use_quantization \
                --lora_output_dir $LORA_OUTPUT_DIR
                
            break
            ;;
        "Run MPT")
            echo "you chose Run MPT."
            
            export MODEL_NAME=MPT
            # export MODEL_TYPE=mosaicml/mpt-7b-chat
            export MODEL_TYPE=mosaicml/mpt-30b-chat
            export DATA_NAME=kbase
            export TASK=entity_type # entity_type (kbase), entity_and_entity_type (kbase)
            export TRAIN_BATCH_SIZE=2 # used in finetuning
            export VALIDATION_BATCH_SIZE=8 # used in finetuning
            export TEST_BATCH_SIZE=8
            export N_SHOTS=0
            export LORA_WEIGHTS="${LORA_OUTPUT_DIR}/mosaicml/mpt-7b-chat/kbase/entity_type/final_checkpoint"
            
            # python ~/BioIE-LLM-WIP/src/run_model.py \
            # torchrun --nproc_per_node=4 ~/BioIE-LLM-WIP/src/run_model.py \
            # accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
            
            accelerate launch "${BASE_DIR}/BioIE-LLM-WIP/src/run_model.py" \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --train_batch_size $TRAIN_BATCH_SIZE \
                --validation_batch_size $VALIDATION_BATCH_SIZE \
                --test_batch_size $TEST_BATCH_SIZE \
                --n_shots $N_SHOTS \
                --lora_finetune \
                --use_quantization \
                --lora_output_dir $LORA_OUTPUT_DIR
            
            : '
                --lora_finetune \
                --lora_output_dir $LORA_OUTPUT_DIR
                --lora_weights $LORA_WEIGHTS
            '
            
            break
            ;;
        "Quit")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done