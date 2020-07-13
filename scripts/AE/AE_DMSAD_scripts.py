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
from src.models.AE_DMSAD import AE_DMSAD
from src.models.networks.Networks import AE_net, Encoder
from src.utils.utils import summary_string
from src.utils.Config import Config

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def main(config_path):
    """
    Train a DMSAD on the MURA dataset using a AE pretraining.
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
        net_AE = AE_net(MLP_Neurons_layer_enc=cfg.settings['AE']['MLP_head_enc'], MLP_Neurons_layer_dec=cfg.settings['AE']['MLP_head_dec'], output_channels=1)
        net_AE = net_AE.to(cfg.settings['device'])
        net_DMSAD = Encoder(MLP_Neurons_layer=cfg.settings['DMSAD']['MLP_head'])
        net_DMSAD = net_DMSAD.to(cfg.settings['device'])
        # print network architecture
        net_architecture = summary_string(net_AE, (1, cfg.settings['Split']['img_size'], cfg.settings['Split']['img_size']),
                                          batch_size=cfg.settings['AE']['batch_size'], device=str(cfg.settings['device']))
        logger.info("AE net architecture: \n" + net_architecture + '\n')
        net_architecture = summary_string(net_DMSAD, (1, cfg.settings['Split']['img_size'], cfg.settings['Split']['img_size']),
                                          batch_size=cfg.settings['DMSAD']['batch_size'], device=str(cfg.settings['device']))
        logger.info("DMSAD net architecture: \n" + net_architecture + '\n')

        # make model
        ae_DMSAD = AE_DMSAD(net_AE, net_DMSAD, eta=cfg.settings['DMSAD']['eta'], gamma=cfg.settings['DMSAD']['gamma'])

        ############################### Train AE ###############################
        # make dataset
        train_dataset_AD = MURA_Dataset(train_df, data_path=cfg.settings['PATH']['DATA'], mask_img=True,
                                        output_size=cfg.settings['Split']['img_size'])
        valid_dataset_AD = MURA_Dataset(valid_df, data_path=cfg.settings['PATH']['DATA'], mask_img=True,
                                        output_size=cfg.settings['Split']['img_size'])
        test_dataset_AD = MURA_Dataset(test_df, data_path=cfg.settings['PATH']['DATA'], mask_img=True,
                                        output_size=cfg.settings['Split']['img_size'])

        logger.info("Online preprocessing pipeline : \n" + str(train_dataset_AD.transform) + "\n")

        # Load model if required
        if cfg.settings['AE']['model_path_to_load']:
            ae_DMSAD.load_repr_net(cfg.settings['AE']['model_path_to_load'], map_location=cfg.settings['device'])
            logger.info(f"AE Model Loaded from {cfg.settings['AE']['model_path_to_load']}" + "\n")

        # print Train parameters
        for key, value in cfg.settings['AE'].items():
            logger.info(f"AE {key} : {value}")

        # Train AE
        ae_DMSAD.train_AE(train_dataset_AD, valid_dataset=None,
                              n_epoch=cfg.settings['AE']['n_epoch'],
                              batch_size=cfg.settings['AE']['batch_size'],
                              lr=cfg.settings['AE']['lr'],
                              weight_decay=cfg.settings['AE']['weight_decay'],
                              lr_milestone=cfg.settings['AE']['lr_milestone'],
                              n_job_dataloader=cfg.settings['AE']['num_worker'],
                              device=cfg.settings['device'],
                              print_batch_progress=cfg.settings['print_batch_progress'])

        # Evaluate AE to get embeddings
        ae_DMSAD.evaluate_AE(valid_dataset_AD, batch_size=cfg.settings['AE']['batch_size'],
                                 n_job_dataloader=cfg.settings['AE']['num_worker'],
                                 device=cfg.settings['device'],
                                 print_batch_progress=cfg.settings['print_batch_progress'],
                                 set='valid')

        ae_DMSAD.evaluate_AE(test_dataset_AD, batch_size=cfg.settings['AE']['batch_size'],
                                 n_job_dataloader=cfg.settings['AE']['num_worker'],
                                 device=cfg.settings['device'],
                                 print_batch_progress=cfg.settings['print_batch_progress'],
                                 set='test')

        # save repr net
        ae_DMSAD.save_ae_net(OUTPUT_PATH + f'model/AE_net_{seed_i+1}.pt')
        logger.info("AE model saved at " + OUTPUT_PATH + f"model/AE_net_{seed_i+1}.pt")

        # save Results
        ae_DMSAD.save_results(OUTPUT_PATH + f'results/results_{seed_i+1}.json')
        logger.info("Results saved at " + OUTPUT_PATH + f"results/results_{seed_i+1}.json")

        ######################## Transfer Encoder Weight #######################

        ae_DMSAD.transfer_encoder()

        ############################## Train DMSAD #############################

        # Load model if required
        if cfg.settings['DMSAD']['model_path_to_load']:
            ae_DMSAD.load_AD(cfg.settings['DMSAD']['model_path_to_load'], map_location=cfg.settings['device'])
            logger.info(f"DMSAD Model Loaded from {cfg.settings['DMSAD']['model_path_to_load']} \n")

        # print Train parameters
        for key, value in cfg.settings['DMSAD'].items():
            logger.info(f"DMSAD {key} : {value}")

        # Train DMSAD
        ae_DMSAD.train_AD(train_dataset_AD, valid_dataset=valid_dataset_AD,
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
        ae_DMSAD.evaluate_AD(valid_dataset_AD, batch_size=cfg.settings['DMSAD']['batch_size'],
                          n_job_dataloader=cfg.settings['DMSAD']['num_worker'],
                          device=cfg.settings['device'],
                          print_batch_progress=cfg.settings['print_batch_progress'],
                          set='valid')
        logger.info('--- Test')
        ae_DMSAD.evaluate_AD(test_dataset_AD, batch_size=cfg.settings['DMSAD']['batch_size'],
                          n_job_dataloader=cfg.settings['DMSAD']['num_worker'],
                          device=cfg.settings['device'],
                          print_batch_progress=cfg.settings['print_batch_progress'],
                          set='test')

        # save DMSAD
        ae_DMSAD.save_AD(OUTPUT_PATH + f'model/DMSAD_{seed_i+1}.pt')
        logger.info("model saved at " + OUTPUT_PATH + f"model/DMSAD_{seed_i+1}.pt")

        ########################## Save Results ################################
        # save Results
        ae_DMSAD.save_results(OUTPUT_PATH + f'results/results_{seed_i+1}.json')
        logger.info("Results saved at " + OUTPUT_PATH + f"results/results_{seed_i+1}.json")

    # save config file
    cfg.settings['device'] = str(cfg.settings['device'])
    cfg.save_config(OUTPUT_PATH + 'config.json')
    logger.info("Config saved at " + OUTPUT_PATH + "config.json")

if __name__ == '__main__':
    main()
