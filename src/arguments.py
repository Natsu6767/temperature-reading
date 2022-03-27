import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default='cnn', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--seed', default='0', type=str)
    parser.add_argument('--exp_name', default='default', type=str)
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--batch_size', default=32, type=int)

    parser.add_argument('--overfit', default=0, type=int)
    parser.add_argument('--data_root', default="thermometers", type=str)
    parser.add_argument('--epochs', default=10, type=int)

    args = parser.parse_args()

    assert args.seed is not None, 'must provide seed for experiment'
    assert args.exp_name is not None, 'must provide an experiment suffix for experiment'
    assert args.log_dir is not None, 'must provide a log directory for experiment'

    return args