# Setup and testing instructions on Polaris

## Environment setup

Environment activation
```shell
module use /soft/modulefiles; module load conda
conda activate llm
```

```shell
pip install huggingface_hub
huggingface-cli login
```

## Source code

```shell
git clone --branch rct --single-branch \
    https://github.com/mtitov/LLM-GeneticTool-Extraction.git BioIE-LLM-WIP
```

## Interactive run

```shell
qsub -I -l select=1 -l filesystems=home:eagle -l walltime=00:30:00 \
     -q debug -A <project_name>

cd BioIE-LLM-WIP
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:}${PYTHONPATH:-}"
export HF_HOME="/eagle/RECUP/matitov/.cache/huggingface"
./scripts/run.sh
```
