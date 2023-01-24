import argparse
import yaml
import subprocess
from pathlib import Path
import datetime
from time import sleep

# parse config.yaml
class Config:
    config_file: str
    core: str
    method: str
    dataset: str
    epochs: int
    bs: int
    gpus: int
    seed = [0]

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
        if 'seed' in config:
            self.seed = config['seed']['value']

    def parse_config(self):
        with open(self.config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def set_method(self, method):
        if not method in ['denstereo', 'gdrn', 'gdrn_selfocc', 'gdrnpp', 'gdrn_stereo']:
            raise ValueError('Method must be one of [denstereo, gdrn, gdrn_selfocc, gdrnpp, gdrn_stereo]')
        self.method = method

    def set_core(self, core):
        if not core in ['denstereo', 'gdrn_selfocc', 'gdrnpp', 'gdrn_stereo']:
            raise ValueError('Method must be one of [denstereo, gdrn_selfocc, gdrnpp, gdrn_stereo]')
        self.core = core
    
    def set_dataset(self, dataset):
        if not dataset in ['denstereo', 'stereobj_1m']:
            raise ValueError('Dataset must be one of [denstereo, stereobj_1m]')
        self.dataset = dataset

    def __str__(self):
        return f'Config: {self.config_file}, {self.core}, {self.method}, {self.dataset}, {self.epochs}, {self.bs}, {self.gpus}'


def generate_timestamp():
    now = datetime.datetime.now()
    return now.strftime("%m%d_%H%M%S")

def get_tmux_pane():
    result = subprocess.run(['bash', str(Path(Path().absolute()) / 'get_tmux_pane.sh')], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')

def run(config, config_name, args, seed=0):
    timestamp = generate_timestamp()
    run_id = config_name + '_' + timestamp + '_' + str(int(seed)) + 's'
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
        "tmux new-session -d -s {run_id}"
        # + " start_session.sh {node} {gpus}"
        # + " {config} {run_id} {method} {dataset} {eval} {branch}"
        + " srun {node} --gpus {gpus} --nodes=1 --cpus-per-gpu=10 --mem-per-cpu=8G --pty"
        + " bash run_gdrn_container.sh {gpus} {config} {run_id} {method} {dataset} {eval} {branch}"
        # + " bash run_gdrn_container.sh {gpus} {config} {run_id} {method} {dataset} {eval} {docker_session} {branch}"
        # + " 2>&1 | tee ~/log/{run_id}.log"
    )
    s = s.format(
        node="-w {}".format(args.node) if args.node else "",
        gpus=gpus,
        config=config_name,
        run_id=run_id,
        method=config.method,
        dataset=config.dataset,
        eval=evaluate,
        # docker_session=get_tmux_pane(),
        branch='debug' if args.debug else 'denstereo',
    )

    print(s + '\n')
    process = subprocess.Popen(s, shell=True)
    if args.eval:
        process.wait()


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Train a GDRN model')
    # config may be a list of configs
    parser.add_argument('--config', action='append', type=str, required=None, help='path to config file')
    parser.add_argument('--docker_session', type=str, help='docker session name')
    parser.add_argument('--resume', type=str, default=None, help='runid to resume training from')
    parser.add_argument('--eval', action='append', type=str, default=None, help='evaluate run id')
    # parser.add_argument('--eval_ampere', type=str, default=None, help='evaluate run id on ampere')
    parser.add_argument('--node', type=str, default=None, help='node to run on')
    parser.add_argument('--debug', action='store_true', help='run in debug mode')
    args = parser.parse_args()

    # load config
    if args.eval:
        for run_id in args.eval:
            if run_id[-1] == 's':
                config_name = '_'.join(run_id.split('_')[:-1])
            config_name = config_name[:-12]
            config_path = "runs/{}.yaml".format(config_name)
            config = Config(config_path)

            args.eval = run_id

            run(config, config_name, args)
    else:
        for config in args.config:
            cfg = Config(config)
            config_name = Path(config).stem

            for seed in cfg.seed:
                run(cfg, config_name, args, seed=seed)