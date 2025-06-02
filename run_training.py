#####################################
# Imports & Dependencies
#####################################
import torch

import os
import argparse
import yaml

from src import data_setup, models, loss, engine, constants
from src.utils import misc


#####################################
# Functions
#####################################
def load_configs():
    # Set configuration file as a hyperparameter
    parser = argparse.ArgumentParser(description = 'Train YOLOv1 model')
    parser.add_argument('-cf', '--config-file', 
                        help = 'Path to the configuration YAML file.',
                        type = str, 
                        default = 'configs.yaml')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.config_file):
        raise FileNotFoundError(f'Config file not found: {args.config_file}')

    with open(args.config_file, 'r') as f:
        configs = yaml.safe_load(f)

    return configs


#####################################
# Training Code
#####################################
if __name__ == '__main__':
    misc.set_seed(0) # Set seed for reproducibility
    configs = load_configs()

    # ---------------------------
    # Model
    # ---------------------------
    backbone = models.build_resnet50_backbone()
    model = models.YOLOv1(backbone = backbone, **configs['model'])

    # Device will be CUDA or MPS if they are avaliable (Change if needed)
    model = model.to(constants.DEVICE)
    # model = torch.compile(model)

    # ---------------------------
    # Dataloader (Pascal VOC)
    # ---------------------------
    train_loader, val_loader = data_setup.get_dataloaders(
        S = model.S, B = model.B,
        **configs['dataloader']
    )


    # ---------------------------
    # Loss, Optimizer, Scheduler
    # ---------------------------
    loss_fn = loss.YOLOv1Loss(S = model.S, B = model.B, C = model.C, 
                            **configs['loss_fn'])

    optimizer = torch.optim.SGD(params = model.parameters(),
                                **configs['optimizer'])

    scheduler = engine.WarmupMultiStepLR(optimizer, **configs['scheduler'])


    # ---------------------------
    # Data Class Configs
    # ---------------------------
    te_cfgs = engine.TrainEvalConfigs(**configs['train_eval'])
    ckpt_cfgs = engine.CheckpointConfigs(**configs['checkpoint'])


    # ---------------------------
    # Run Training
    # ---------------------------
    train_losses, val_losses, eval_history = engine.train(
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        loss_fn = loss_fn,
        optimizer = optimizer,
        scheduler = scheduler,
        te_cfgs = te_cfgs,
        ckpt_cfgs = ckpt_cfgs,
        device = constants.DEVICE
    )