import argparse
import yaml
from typing import TypedDict
import subprocess
from pathlib import Path

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

    
if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Train a GDRN model')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume training from')
    parser.add_argument('--eval', action="store_true", default=False, help='evaluate run')
    args = parser.parse_args()

    # load config
    config = Config(args.config)
    print(config)

    s = "srun --gpus {gpus} --nodes=1 --cpus-per-gpu=10 --mem-per-cpu=8G --pty bash run_gdrn_container.sh {gpus} {config} {method} {dataset} {eval}"
    s = s.format(
        gpus=config.gpus,
        config=Path(args.config).stem,
        method=config.method,
        dataset=config.dataset,
        eval=args.eval
    )

    print(s + '\n')
    subprocess.call(s, shell=True)
