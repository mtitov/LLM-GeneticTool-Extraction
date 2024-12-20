#!/usr/bin/env python3

# module use /soft/modulefiles; module load conda
# conda activate llm
# export PYTHONPATH="/home/matitov/gllm/BioIE-LLM-WIP/src:$PYTHONPATH"
# nohup python3 launcher.rp.py > OUTPUT 2>&1 </dev/null &

import os

import radical.pilot as rp

os.environ['RADICAL_LOG_LVL'] = 'DEBUG'
os.environ['RADICAL_REPORT']  = 'TRUE'

# Polaris (ALCF) specific
WORK_DIR = '/home/matitov/gllm/BioIE-LLM-WIP'
N_NODES = 1
CPUS_PER_NODE = 64
GPUS_PER_NODE = 4
TASK_PRE_EXEC = [
    'module load PrgEnv-nvhpc',
    'unset https_proxy',
    'unset http_proxy',
    'module use /soft/modulefiles; module load conda',
    'conda activate llm',
    f'export PYTHONPATH="{WORK_DIR}/src:$PYTHONPATH"',
    'export PYTHONDONTWRITEBYTECODE=1',
    'export HF_HOME="/eagle/RECUP/matitov/.cache/huggingface"'
]

PILOT_DESCRIPTION = {
    'resource': 'anl.polaris',
    'project' : 'NNNN',
    'nodes'   : N_NODES,
    'runtime' : 20,
    'sandbox' : WORK_DIR,
    # 'input_staging': ['service.py', 'service.json', 'client.py']
}


def main():
    session = rp.Session()
    pmgr = rp.PilotManager(session)
    tmgr = rp.TaskManager(session)

    service_td = rp.TaskDescription({
        'executable'    : 'python3',
        'arguments'     : ['-m', 'wfms.service'],
        'pre_exec'      : TASK_PRE_EXEC,
        'ranks'         : 1,
        'gpus_per_rank' : GPUS_PER_NODE,
        'gpu_type'      : rp.CUDA
    })

    pilot = pmgr.submit_pilots(rp.PilotDescription(
        dict(services=[service_td], **PILOT_DESCRIPTION)
    ))

    tmgr.add_pilots(pilot)
    pilot.wait(rp.PMGR_ACTIVE)

    task = rp.TaskDescription({'executable': 'sleep', 'arguments': [30]})
    tmgr.submit_tasks(task)
    tmgr.wait_tasks()

    task_template = rp.TaskDescription({
        'executable': 'python3',
        'arguments' : ['-m', 'wfms.client', '--action', None],
        'pre_exec'  : TASK_PRE_EXEC
    })

    task = rp.TaskDescription(task_template.as_dict())
    task.arguments[-1] = 'load_model'
    tmgr.submit_tasks(task)
    tmgr.wait_tasks()

    task = rp.TaskDescription(task_template.as_dict())
    task.arguments[-1] = 'generate_datasets'
    tmgr.submit_tasks(task)
    tmgr.wait_tasks()

    task = rp.TaskDescription(task_template.as_dict())
    task.arguments[-1] = 'finetune'
    tmgr.submit_tasks(task)
    tmgr.wait_tasks()

    task = rp.TaskDescription(task_template.as_dict())
    task.arguments[-1] = 'inference'
    tmgr.submit_tasks(task)
    tmgr.wait_tasks()

    session.close(download=True)


if __name__ == '__main__':
    main()
