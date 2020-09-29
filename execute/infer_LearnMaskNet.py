import os
import argparse
import importlib
import sys

import torch
from tqdm import tqdm
import numpy as np


def infer_default(iter_max):
    # load edge prediction model checkpoint
    EDGE_MODULE = 'LearnEdgeNet'
    edge_prediction_checkpoint = torch.load(resume_edge_module)
    edge_prediction_config = edge_prediction_checkpoint['config']
    assert edge_prediction_config['module'] == EDGE_MODULE
    edge_prediction_module_arch = importlib.import_module('.model_' + EDGE_MODULE, package='model')
    edge_prediction_model_class = getattr(edge_prediction_module_arch, edge_prediction_config['model']['type'])
    edge_prediction_model = edge_prediction_model_class(**edge_prediction_config['model']['args'])
    edge_prediction_model = edge_prediction_model.to(device)
    edge_prediction_model.load_state_dict(edge_prediction_checkpoint['model'])
    edge_prediction_model.eval()

    # make dirs
    unwrapped_dir = os.path.join(result_dir, 'unwrapped')
    util.ensure_dir(unwrapped_dir)

    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(data_loader, ascii=True)):
            name = sample['name'][0]
            img_dir = os.path.join(result_dir, 'steps', name)
            util.ensure_dir(img_dir)

            # get data and send them to GPU
            modulo = sample['modulo'].to(device)  # positive int, as float32
            mask_pred = torch.ones_like(modulo).to(device)

            i = 0
            while torch.sum(mask_pred) > 0 and i <= iter_max:
                modulo_numpy = modulo.squeeze(0).permute(1, 2, 0).cpu().numpy()
                np.save(os.path.join(img_dir, str(i) + '.npy'), modulo_numpy)

                modulo_edge = torch.abs(util.torch_laplacian(modulo))  # positive int, as float32
                edge_out = edge_prediction_model(modulo / torch.max(modulo), modulo_edge / torch.max(modulo_edge))
                fold_number_edge = torch.round(torch.sigmoid(edge_out))  # binary, as float32
                output = model(modulo / torch.max(modulo), fold_number_edge)

                if confine:
                    mask_pred *= torch.round(torch.sigmoid(output))
                else:
                    mask_pred = torch.round(torch.sigmoid(output))

                modulo += 256 * mask_pred
                i += 1

            unwrapped_numpy = modulo.squeeze(0).permute(1, 2, 0).cpu().numpy()  # positive int, as float32
            np.save(os.path.join(unwrapped_dir, name + '.npy'), unwrapped_numpy)


if __name__ == '__main__':
    MODULE = 'LearnMaskNet'
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', required=True, type=str, help='path to latest checkpoint')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--data_dir', required=True, type=str, help='dir of input data')
    parser.add_argument('--result_dir', required=True, type=str, help='dir to save result')
    parser.add_argument('--data_loader_type', default='InferDataLoader', type=str, help='which data loader to use')
    parser.add_argument('--confine', default=0, type=int, help='confine mode')
    parser.add_argument('--resume_edge_module', required=True, type=str,
                        help='path to latest checkpoint of edge prediction model')
    subparsers = parser.add_subparsers(help='which func to run', dest='func')

    # add subparsers and their args for each func
    subparser = subparsers.add_parser("default")
    subparser.add_argument('--iter_max', default=15, type=int, help='iteration limit')

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root to PATH
    from utils import util

    # load checkpoint
    checkpoint = torch.load(args.resume)
    config = checkpoint['config']
    assert config['module'] == MODULE

    # setup data_loader instances
    # we choose batch_size=1(default value)
    module_data = importlib.import_module('.data_loader_' + MODULE, package='data_loader')
    data_loader_class = getattr(module_data, args.data_loader_type)
    data_loader = data_loader_class(data_dir=args.data_dir)

    # build model architecture
    module_arch = importlib.import_module('.model_' + MODULE, package='model')
    model_class = getattr(module_arch, config['model']['type'])
    model = model_class(**config['model']['args'])

    # prepare model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])

    # set the model to validation mode
    model.eval()

    # ensure result_dir
    result_dir = args.result_dir
    util.ensure_dir(result_dir)

    # use the previous mask as the confinement of the current mask
    confine = bool(args.confine)
    # path to latest checkpoint of edge prediction model
    resume_edge_module = args.resume_edge_module

    # run the selected func
    if args.func == 'default':
        infer_default(args.iter_max)
    else:
        # run the default
        infer_default(args.iter_max)
