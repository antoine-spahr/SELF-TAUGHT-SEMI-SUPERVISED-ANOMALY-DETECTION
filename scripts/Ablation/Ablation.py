import torch
import torch.cuda
import logging
import numpy as np
import pandas as pd
import random
from datetime import datetime
import os
import sys
sys.path.append('../../')
import click

from src.datasets.MURADataset import MURA_TrainValidTestSplitter, MURA_Dataset
from src.models.NearestNeighbors import NearestNeighbors
from src.models.networks.Networks import AE_net
from src.models.networks.Networks import Encoder
from src.utils.Config import Config

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--type', type=click.Choice(['SimCLR', 'AE']), default='AE',
              help='The type of model to load. Default: SimCLR.')
@click.option('--n_cluster', type=int, default=0,
              help='The number of cluster for the K-Means simplification. Default 0 : using the config file.')
@click.option('--n_neighbors', type=int, default=0,
              help='The number of neighbors to consider. Default 0 : using the config file.')
def main(config_path, type, n_cluster, n_neighbors):
    """
    Ablation study on Contrastive or AE representation.
    """
    # Load config file
    cfg = Config(settings=None)
    cfg.load_config(config_path)

    # Update n_cluster and n_neighbors if provided
    if n_cluster > 0:
        cfg.settings['n_cluster'] = n_cluster
    if n_neighbors > 0:
        cfg.settings['n_neighbors'] = n_neighbors

    # Get path to output
    OUTPUT_PATH = cfg.settings['PATH']['OUTPUT'] + cfg.settings['Experiment_Name'] + datetime.today().strftime('%Y_%m_%d_%Hh%M')+'/'
    # make output dir
    if not os.path.isdir(OUTPUT_PATH+'models/'): os.makedirs(OUTPUT_PATH+'model/', exist_ok=True)
    if not os.path.isdir(OUTPUT_PATH+'results/'): os.makedirs(OUTPUT_PATH+'results/', exist_ok=True)
    if not os.path.isdir(OUTPUT_PATH+'logs/'): os.makedirs(OUTPUT_PATH+'logs/', exist_ok=True)

    for seed_i, seed in enumerate(cfg.settings['seeds']):
        ############################### Set Up #################################
        # initialize logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        try:
            logger.handlers[1].stream.close()
            logger.removeHandler(logger.handlers[1])
        except IndexError:
            pass
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        log_file = OUTPUT_PATH + 'logs/' + f'log_{seed_i+1}.txt'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # print path
        logger.info(f"Log file : {log_file}")
        logger.info(f"Data path : {cfg.settings['PATH']['DATA']}")
        logger.info(f"Outputs path : {OUTPUT_PATH}" + "\n")

        # Set seed
        if seed != -1:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            logger.info(f"Set seed {seed_i+1:02}/{len(cfg.settings['seeds']):02} to {seed}")

        # set number of thread
        if cfg.settings['n_thread'] > 0:
            torch.set_num_threads(cfg.settings['n_thread'])

        # check if GPU available
        cfg.settings['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # Print technical info in logger
        logger.info(f"Device : {cfg.settings['device']}")
        logger.info(f"Number of thread : {cfg.settings['n_thread']}")

        ############################### Split Data #############################
        # Load data informations
        df_info = pd.read_csv(cfg.settings['PATH']['DATA_INFO'])
        df_info = df_info.drop(df_info.columns[0], axis=1)
        # remove low contrast images (all black)
        df_info = df_info[df_info.low_contrast == 0]

        # Train Validation Test Split
        spliter = MURA_TrainValidTestSplitter(df_info, train_frac=cfg.settings['Split']['train_frac'],
                                              ratio_known_normal=cfg.settings['Split']['known_normal'],
                                              ratio_known_abnormal=cfg.settings['Split']['known_abnormal'],
                                              random_state=42)
        spliter.split_data(verbose=False)
        train_df = spliter.get_subset('train')
        valid_df = spliter.get_subset('valid')
        test_df = spliter.get_subset('test')

        # print info to logger
        for key, value in cfg.settings['Split'].items():
            logger.info(f"Split param {key} : {value}")
        logger.info("Split Summary \n" + str(spliter.print_stat(returnTable=True)))

        ############################### Load model #############################
        if type == 'SimCLR':
            net = Encoder(MLP_Neurons_layer=[512, 256, 128])
            init_key = 'repr_net_dict'
        elif type == 'AE':
            net = AE_net(MLP_Neurons_layer_enc=[512, 256, 128], MLP_Neurons_layer_dec=[128,256,512], output_channels=1)
            init_key = 'ae_net_dict'

        net = net.to(cfg.settings['device'])
        pretrain_state_dict = torch.load(cfg.settings['model_path_name']+f'{seed_i+1}.pt',
                                         map_location=cfg.settings['device'])
        net.load_state_dict(pretrain_state_dict[init_key])
        logger.info('Model weights successfully loaded from ' + cfg.settings['model_path_name']+f'{seed_i+1}.pt')

        ablation_model = NearestNeighbors(net, kmeans_reduction=cfg.settings['kmeans_reduction'],
                                        batch_size=cfg.settings['batch_size'],
                                        n_job_dataloader=cfg.settings['num_worker'],
                                        print_batch_progress=cfg.settings['print_batch_progress'],
                                        device=cfg.settings['device'])

        ############################### Training ###############################
        # make dataset
        train_dataset = MURA_Dataset(train_df, data_path=cfg.settings['PATH']['DATA'], load_mask=True,
                                     load_semilabels=True, output_size=cfg.settings['Split']['img_size'],
                                     data_augmentation=False)
        valid_dataset = MURA_Dataset(valid_df, data_path=cfg.settings['PATH']['DATA'], load_mask=True,
                                     load_semilabels=True, output_size=cfg.settings['Split']['img_size'],
                                     data_augmentation=False)
        test_dataset = MURA_Dataset(test_df, data_path=cfg.settings['PATH']['DATA'], load_mask=True,
                                    load_semilabels=True, output_size=cfg.settings['Split']['img_size'],
                                    data_augmentation=False)

        logger.info("Online preprocessing pipeline : \n" + str(train_dataset.transform) + "\n")

        # Train
        ablation_model.train(train_dataset, n_cluster=cfg.settings['n_cluster'])

        # Evaluate
        logger.info(f'--- Validation with {cfg.settings["n_neighbors"]} neighbors')
        ablation_model.evaluate(valid_dataset, n_neighbors=cfg.settings['n_neighbors'],
                                mode='valid')
        logger.info(f'--- Test with {cfg.settings["n_neighbors"]} neighbors')
        ablation_model.evaluate(test_dataset, n_neighbors=cfg.settings['n_neighbors'],
                                mode='test')

        ############################## Save Results ############################
        ablation_model.save_results(OUTPUT_PATH + f'results/results_{seed_i+1}.json')
        logger.info("Results saved at " + OUTPUT_PATH + f"results/results_{seed_i+1}.json")

        ablation_model.save_model(OUTPUT_PATH + f'model/ablation_{seed_i+1}.pt')
        logger.info("model saved at " + OUTPUT_PATH + f"model/abaltion_{seed_i+1}.pt")

    # save config file
    cfg.settings['device'] = str(cfg.settings['device'])
    cfg.save_config(OUTPUT_PATH + 'config.json')
    logger.info("Config saved at " + OUTPUT_PATH + "config.json")

if __name__ == '__main__':
    main()
