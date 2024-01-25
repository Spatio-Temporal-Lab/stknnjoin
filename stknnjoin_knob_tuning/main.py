import os

from search import search_outline, search_online
from utils.RuntimeContext import RuntimeContext
from utils.logger import TrainingLogger

args = RuntimeContext()


def initial_params():
    args.knobs_setting = {'alpha': 200, 'beta': 40, 'binNum': 200}
    args.data_type = args.output_dir.split('/')[1]


def initial_logger():
    save_path = f'{args.output_dir}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger = TrainingLogger(f'{save_path}/log')
    args.logger = logger
    args.save_path = save_path


if __name__ == '__main__':
    initial_params()
    initial_logger()

    r = search_outline(args)
    # r = search_online(args)
    for knob in r.x:
        args.logger.print(f'{knob}\t')
    args.logger.print(str(r.fun))

