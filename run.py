import argparse
import yaml
from typing import TypedDict
import subprocess
from pathlib import Path
import datetime

# parse config.yaml
class Config:
    config_file: str
    core: str
    method: str
    dataset: str
    epochs: int
    bs: int
    gpus: int

    def __init__(self, config_path):
        self.config_path = config_path
        config = self.parse_config()

        self.config_file = config['config_file']['value']
        self.set_core(config['core']['value'])
        self.set_method(config['method']['value'])
        self.set_dataset(config['dataset']['value'])
        self.epochs = config['epochs']['value']
        self.bs = config['bs']['value']
        self.gpus = config['gpus']['value']

    def parse_config(self):
        with open(self.config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def set_method(self, method):
        if not method in ['denstereo', 'gdrn', 'gdrn_selfocc', 'gdrnpp']:
            raise ValueError('Method must be one of [denstereo, gdrn, gdrn_selfocc, gdrnpp]')
        self.method = method

    def set_core(self, core):
        if not core in ['denstereo', 'gdrn_selfocc', 'gdrnpp']:
            raise ValueError('Method must be one of [denstereo, gdrn_selfocc, gdrnpp]')
        self.core = core
    
    def set_dataset(self, dataset):
        if not dataset in ['denstereo']:
            raise ValueError('Dataset must be one of [denstereo]')
        self.dataset = dataset

    def __str__(self):
        return f'Config: {self.config_file}, {self.core}, {self.method}, {self.dataset}, {self.epochs}, {self.bs}, {self.gpus}'


def generate_timestamp():
    now = datetime.datetime.now()
    return now.strftime("%m%d_%H%M%S")

def get_tmux_pane():
    result = subprocess.run(['bash', str(Path(Path().absolute()) / 'get_tmux_pane.sh')], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Train a GDRN model')
    parser.add_argument('--config', type=str, required=None, help='path to config file')
    parser.add_argument('--docker_session', type=str, help='docker session name')
    parser.add_argument('--resume', type=str, default=None, help='runid to resume training from')
    parser.add_argument('--eval', type=str, default=None, help='evaluate run id')
    parser.add_argument('--eval_ampere', type=str, default=None, help='evaluate run id on ampere')
    parser.add_argument('--node', type=str, default=None, help='node to run on')
    parser.add_argument('--debug', action='store_true', help='run in debug mode')
    args = parser.parse_args()

    # load config
    if args.eval_ampere:
        args.eval = args.eval_ampere

        s = "runs/{}.yaml".format(args.eval_ampere[:-12])
        print(s)
        config_name = args.eval_ampere[:-12]
        config = Config("runs/{}.yaml".format(config_name))
    else:
        config = Config(args.config)
        config_name = Path(args.config).stem

    timestamp = generate_timestamp()
    run_id = config_name + '_' + timestamp
    gpus = config.gpus

    if args.eval:
        evaluate = "True"
        run_id = args.eval
        gpus = 1
    else:
        evaluate = "False"
    if args.resume:
        resume = args.resume
    else:
        resume = "False"

    s = (
        "srun {node} --gpus {gpus} --nodes=1 --cpus-per-gpu=10 --mem-per-cpu=8G --pty"
        + " bash run_gdrn_container.sh {gpus} {config} {run_id} {method} {dataset} {eval} {docker_session} {branch}"
    )
    s = s.format(
        node="-w {}".format(args.node) if args.node else "",
        gpus=gpus,
        config=config_name,
        run_id=run_id,
        method=config.method,
        dataset=config.dataset,
        eval=evaluate,
        docker_session=get_tmux_pane(),
        branch='debug' if args.debug else 'denstereo',
    )

    print(s + '\n')
    subprocess.call(s, shell=True)
