import argparse
import yaml
from typing import TypedDict
import subprocess
from pathlib import Path
import datetime

# parse config.yaml
class Config:
    config_file: str
    method: str
    dataset: str
    epochs: int
    bs: int
    gpus: int

    def __init__(self, config_path):
        self.config_path = config_path
        config = self.parse_config()

        self.config_file = config['config_file']['value']
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
    
    def set_dataset(self, dataset):
        if not dataset in ['denstereo']:
            raise ValueError('Dataset must be one of [denstereo]')
        self.dataset = dataset

    def __str__(self):
        return f'Config: {self.config_file}, {self.method}, {self.dataset}, {self.epochs}, {self.bs}, {self.gpus}'


def generate_timestamp():
    now = datetime.datetime.now()
    return now.strftime("%m%d_%H%M%S")

def get_tmux_pane():
    result = subprocess.run(['bash', str(Path(Path().absolute()) / 'get_tmux_pane.sh')], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Train a GDRN model')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--docker_session', type=str, help='docker session name')
    parser.add_argument('--resume', type=str, default=None, help='runid to resume training from')
    parser.add_argument('--eval', type=str, default=None, help='evaluate run id')
    args = parser.parse_args()

    # load config
    config = Config(args.config)
    timestamp = generate_timestamp()
    
    config_name = Path(args.config).stem
    run_id = config_name + '_' + timestamp

    if args.eval:
        evaluate = "True"
        run_id = args.eval
    else:
        evaluate = "False"
    if args.resume:
        resume = args.resume
    else:
        resume = "False"

    s = "srun --gpus {gpus} -w ampere4 --nodes=1 --cpus-per-gpu=10 --mem-per-cpu=8G --pty bash run_gdrn_container.sh {gpus} {config} {run_id} {method} {dataset} {eval} {docker_session}"
    s = s.format(
        gpus=config.gpus,
        config=config_name,
        run_id=run_id,
        method=config.method,
        dataset=config.dataset,
        eval=evaluate,
        docker_session=get_tmux_pane(),
    )

    print(s + '\n')
    subprocess.call(s, shell=True)
