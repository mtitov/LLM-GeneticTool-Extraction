import argparse
import os
import sys

import radical.utils as ru


def get_args():
    parser = argparse.ArgumentParser(
        description='Configure service client',
        usage='client.py [<options>]')
    parser.add_argument(
        '-a', '--action',
        dest='action',
        type=str,
        required=True)
    return parser.parse_args(sys.argv[1:])


def main(action):

    reg_addr = os.getenv('RP_REGISTRY_ADDRESS')
    reg = ru.zmq.RegistryClient(url=reg_addr)
    service_addr = reg['app.service_addr']
    reg.close()

    if not service_addr:
        service_cfg = ru.read_json(
            f'{os.getenv("RP_PILOT_SANDBOX")}/model_service_reg.json')
        service_addr = service_cfg['service_addr']

    uid = os.getenv('RP_TASK_ID')
    prof = ru.Profiler(name=uid, ns='radical.pilot', path=os.getcwd())

    client = ru.zmq.Client(url=service_addr)
    prof.prof('client_req_start', uid=uid)
    prompt_response = client.request('processor', {'action': action})
    prof.prof('client_req_stop', uid=uid)
    print(prompt_response)
    client.close()


if __name__ == '__main__':
    args = get_args()
    main(args.action)

