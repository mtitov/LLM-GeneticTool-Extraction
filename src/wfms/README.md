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

## Interactive run

```shell
qsub -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 \
     -q debug -A <project_name>
```
