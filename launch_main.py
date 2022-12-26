import argparse
import yaml
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or evaluate a gdrn model")
    parser.add_argument("run", help="config file name", type=str)
    parser.add_argument("run_id", help="unique run id", type=str)
    parser.add_argument("--evaluate", action="store_true", default=False, help="evaluate or train")
    parser.add_argument("--checkpoint", help="checkpoint to load", type=str)

    args = parser.parse_args()
    run = args.run
    run_id = args.run_id

    with open('runs/{run}.yaml'.format(run=run), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config_file = config['config_file']['value']
    method = config['method']['value']
    dataset = config['dataset']['value']
    epochs = config['epochs']['value']
    bs = config['bs']['value']
    gpus = config['gpus']['value']

    config_path = 'configs/{method}/{dataset}/{config_file}'.format(
        method=method,
        dataset=dataset,
        config_file=config_file
    )

    if args.evaluate:
        weights = 'output/{method}/{dataset}/{run_id}/model_final.pth'.format(
            method=method,
            dataset=dataset,
            run=run
        )
        if args.checkpoint:
            weights = args.checkpoint

        print('Evaluating model: {weights}'.format(weights=weights))


    s = "python core/{method}_modeling/main_gdrn.py --config-file {config} --num-gpus {gpus} {eval} --opts SOLVER.IMS_PER_BATCH={bs} SOLVER.TOTAL_EPOCHS={epochs} OUTPUT_DIR=\"output/{method}/{dataset}/{run_id}\" SOLVER.MAX_TO_KEEP={max_to_keep} SOLVER.CHECKPOINT_PERIOD={checkpoint_period} {weights}"
    s = s.format(
        method=method,
        dataset=dataset,
        config=config_path,
        gpus=gpus if args.evaluate else 1,
        eval="--eval-only" if args.evaluate is True else "",
        bs=bs,
        epochs=epochs,
        run=run,
        max_to_keep=2,
        checkpoint_period=40,
        weights="MODEL.WEIGHTS=\"{}\"".format(weights) if args.evaluate else ""
    )
    print(s + '\n')

    subprocess.call(s, shell=True)