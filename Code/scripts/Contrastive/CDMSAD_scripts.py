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

from src.datasets.MURADataset import MURA_TrainValidTestSplitter, MURA_Dataset, MURADataset_Contrastive
from src.models.CDMSAD import CDMSAD
from src.models.networks.Networks import Encoder
from src.utils.utils import summary_string
from src.utils.Config import Config

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def main(config_path):
    """
    Train a DMSAD on the MURA dataset using a Contrastive pretraining.
    """
    # Load config file
    cfg = Config(settings=None)
    cfg.load_config(config_path)

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

        ############################# Build Model  #############################
        # make networks
        net_CLR = Encoder(MLP_Neurons_layer=cfg.settings['SimCLR']['MLP_head'])
        net_CLR = net_CLR.to(cfg.settings['device'])
        net_DMSAD = Encoder(MLP_Neurons_layer=cfg.settings['DMSAD']['MLP_head'])
        net_DMSAD = net_DMSAD.to(cfg.settings['device'])
        # print network architecture
        net_architecture = summary_string(net_CLR, (1, cfg.settings['Split']['img_size'], cfg.settings['Split']['img_size']),
                                          batch_size=cfg.settings['SimCLR']['batch_size'], device=str(cfg.settings['device']))
        logger.info("Contrastive net architecture: \n" + net_architecture + '\n')
        net_architecture = summary_string(net_DMSAD, (1, cfg.settings['Split']['img_size'], cfg.settings['Split']['img_size']),
                                          batch_size=cfg.settings['DMSAD']['batch_size'], device=str(cfg.settings['device']))
        logger.info("DMSAD net architecture: \n" + net_architecture + '\n')

        # make model
        clr_DMSAD = CDMSAD(net_CLR, net_DMSAD, tau=cfg.settings['SimCLR']['tau'],
                               eta=cfg.settings['DMSAD']['eta'], gamma=cfg.settings['DMSAD']['gamma'])

        ########################### Train Contrastive ##########################
        # make datasets
        train_dataset_CLR = MURADataset_Contrastive(train_df, data_path=cfg.settings['PATH']['DATA'],
                                     output_size=cfg.settings['Split']['img_size'], mask_img=True)
        valid_dataset_CLR = MURADataset_Contrastive(valid_df, data_path=cfg.settings['PATH']['DATA'],
                                     output_size=cfg.settings['Split']['img_size'], mask_img=True)
        test_dataset_CLR = MURADataset_Contrastive(test_df, data_path=cfg.settings['PATH']['DATA'],
                                    output_size=cfg.settings['Split']['img_size'], mask_img=True)

        logger.info("Contrastive Online preprocessing pipeline : \n" + str(train_dataset_CLR.transform) + "\n")

        # Load model if required
        if cfg.settings['SimCLR']['model_path_to_load']:
            clr_DMSAD.load_repr_net(cfg.settings['SimCLR']['model_path_to_load'], map_location=cfg.settings['device'])
            logger.info(f"Contrastive Model Loaded from {cfg.settings['SimCLR']['model_path_to_load']}" + "\n")

        # print Train parameters
        for key, value in cfg.settings['SimCLR'].items():
            logger.info(f"Contrastive {key} : {value}")

        # Train Contrastive
        clr_DMSAD.train_contrastive(train_dataset_CLR, valid_dataset=None,
                              n_epoch=cfg.settings['SimCLR']['n_epoch'],
                              batch_size=cfg.settings['SimCLR']['batch_size'],
                              lr=cfg.settings['SimCLR']['lr'],
                              weight_decay=cfg.settings['SimCLR']['weight_decay'],
                              lr_milestones=cfg.settings['SimCLR']['lr_milestone'],
                              n_job_dataloader=cfg.settings['SimCLR']['num_worker'],
                              device=cfg.settings['device'],
                              print_batch_progress=cfg.settings['print_batch_progress'])

        # Evaluate Contrastive to get embeddings
        clr_DMSAD.evaluate_contrastive(valid_dataset_CLR, batch_size=cfg.settings['SimCLR']['batch_size'],
                                 n_job_dataloader=cfg.settings['SimCLR']['num_worker'],
                                 device=cfg.settings['device'],
                                 print_batch_progress=cfg.settings['print_batch_progress'],
                                 set='valid')

        clr_DMSAD.evaluate_contrastive(test_dataset_CLR, batch_size=cfg.settings['SimCLR']['batch_size'],
                                 n_job_dataloader=cfg.settings['SimCLR']['num_worker'],
                                 device=cfg.settings['device'],
                                 print_batch_progress=cfg.settings['print_batch_progress'],
                                 set='test')

        # save repr net
        clr_DMSAD.save_repr_net(OUTPUT_PATH + f'model/Contrastive_net_{seed_i+1}.pt')
        logger.info("Contrastive model saved at " + OUTPUT_PATH + f"model/Contrastive_net_{seed_i+1}.pt")

        # save Results
        clr_DMSAD.save_results(OUTPUT_PATH + f'results/results_{seed_i+1}.json')
        logger.info("Results saved at " + OUTPUT_PATH + f"results/results_{seed_i+1}.json")

        ######################## Transfer Encoder Weight #######################

        clr_DMSAD.transfer_encoder()

        ############################## Train DMSAD ##############################
        # make dataset
        train_dataset_AD = MURA_Dataset(train_df, data_path=cfg.settings['PATH']['DATA'], mask_img=True,
                                        output_size=cfg.settings['Split']['img_size'])
        valid_dataset_AD = MURA_Dataset(valid_df, data_path=cfg.settings['PATH']['DATA'], mask_img=True,
                                        output_size=cfg.settings['Split']['img_size'])
        test_dataset_AD = MURA_Dataset(test_df, data_path=cfg.settings['PATH']['DATA'], mask_img=True,
                                        output_size=cfg.settings['Split']['img_size'])

        logger.info("DMSAD Online preprocessing pipeline : \n" + str(train_dataset_AD.transform) + "\n")

        # Load model if required
        if cfg.settings['DMSAD']['model_path_to_load']:
            clr_DMSAD.load_AD(cfg.settings['DMSAD']['model_path_to_load'], map_location=cfg.settings['device'])
            logger.info(f"DMSAD Model Loaded from {cfg.settings['DMSAD']['model_path_to_load']} \n")

        # print Train parameters
        for key, value in cfg.settings['DMSAD'].items():
            logger.info(f"DMSAD {key} : {value}")

        # Train DMSAD
        clr_DMSAD.train_AD(train_dataset_AD, valid_dataset=valid_dataset_AD,
                          n_sphere_init=cfg.settings['DMSAD']['n_sphere_init'],
                          n_epoch=cfg.settings['DMSAD']['n_epoch'],
                          batch_size=cfg.settings['DMSAD']['batch_size'],
                          lr=cfg.settings['DMSAD']['lr'],
                          weight_decay=cfg.settings['DMSAD']['weight_decay'],
                          lr_milestone=cfg.settings['DMSAD']['lr_milestone'],
                          n_job_dataloader=cfg.settings['DMSAD']['num_worker'],
                          device=cfg.settings['device'],
                          print_batch_progress=cfg.settings['print_batch_progress'])
        logger.info('--- Validation')
        clr_DMSAD.evaluate_AD(valid_dataset_AD, batch_size=cfg.settings['DMSAD']['batch_size'],
                          n_job_dataloader=cfg.settings['DMSAD']['num_worker'],
                          device=cfg.settings['device'],
                          print_batch_progress=cfg.settings['print_batch_progress'],
                          set='valid')
        logger.info('--- Test')
        clr_DMSAD.evaluate_AD(test_dataset_AD, batch_size=cfg.settings['DMSAD']['batch_size'],
                          n_job_dataloader=cfg.settings['DMSAD']['num_worker'],
                          device=cfg.settings['device'],
                          print_batch_progress=cfg.settings['print_batch_progress'],
                          set='test')

        # save DMSAD
        clr_DMSAD.save_AD(OUTPUT_PATH + f'model/DMSAD_{seed_i+1}.pt')
        logger.info("model saved at " + OUTPUT_PATH + f"model/DMSAD_{seed_i+1}.pt")

        ########################## Save Results ################################
        # save Results
        clr_DMSAD.save_results(OUTPUT_PATH + f'results/results_{seed_i+1}.json')
        logger.info("Results saved at " + OUTPUT_PATH + f"results/results_{seed_i+1}.json")

    # save config file
    cfg.settings['device'] = str(cfg.settings['device'])
    cfg.save_config(OUTPUT_PATH + 'config.json')
    logger.info("Config saved at " + OUTPUT_PATH + "config.json")

if __name__ == '__main__':
    main()
