#!/usr/bin/env python3

# source ve.rp/bin/activate
# nohup python3 launcher.rp.py > OUTPUT 2>&1 </dev/null &

import os

import radical.pilot as rp

os.environ['RADICAL_LOG_LVL'] = 'DEBUG'
os.environ['RADICAL_REPORT']  = 'TRUE'

# Polaris (ALCF) specific
WORK_DIR = '/home/matitov/LLM-GeneticTool-Extraction'
N_NODES = 1
CPUS_PER_NODE = 64
GPUS_PER_NODE = 4
TASK_PRE_EXEC = [
    'module load PrgEnv-nvhpc',
    'unset https_proxy',
    'unset http_proxy',
    f'source {WORK_DIR}/ve.rp/bin/activate'
]

PILOT_DESCRIPTION = {
    'resource': 'anl.polaris',
    'project': 'RECUP',
    'nodes': N_NODES,
    'runtime': 60,
    'sandbox': WORK_DIR,
    'input_staging': ['service.py', 'client.py']
}

IS_BASE = False  # 1 task with many prompts
WITH_PROBE_TASK = False  # have a single-core task before a batch of tasks


def main():
    session = rp.Session()
    pmgr = rp.PilotManager(session)
    tmgr = rp.TaskManager(session)

    service_td = rp.TaskDescription({
        'executable'    : 'python3',
        'arguments'     : ['$RP_PILOT_SANDBOX/service.py'],
        'pre_exec'      : TASK_PRE_EXEC,
        'gpus_per_rank' : GPUS_PER_NODE,
        'gpu_type'      : rp.CUDA,
        'sandbox'       : '$RP_PILOT_SANDBOX'
    })

    pilot = pmgr.submit_pilots(rp.PilotDescription(
        dict(services=[service_td], **PILOT_DESCRIPTION)
    ))

    tmgr.add_pilots(pilot)
    pilot.wait(rp.PMGR_ACTIVE)

    task_template = rp.TaskDescription({
        'executable': 'python3',
        'arguments' : ['$RP_PILOT_SANDBOX/client.py', '--action', None],
        'pre_exec'  : TASK_PRE_EXEC
    })

    task = rp.TaskDescription(task_template.as_dict())
    task.arguments[-1] = 'generate'
    tmgr.submit_tasks(task)
    tmgr.wait_tasks()

    task = rp.TaskDescription(task_template.as_dict())
    task.arguments[-1] = 'tune'
    tmgr.submit_tasks(task)
    tmgr.wait_tasks()

    task = rp.TaskDescription(task_template.as_dict())
    task.arguments[-1] = 'inference'
    tmgr.submit_tasks(task)
    tmgr.wait_tasks()

    session.close(download=True)


if __name__ == '__main__':
    main()
