# Self-Taught Semi-Supervised Anomaly Detection on Upper Limb X-rays

This repository contains the Pytorch implementation of the method proposed  the _Self-taught Semi-Supervised Anomaly detection on Upper limb X-rays_ under submission at ISBI 2021.

____
## Abstract

Detecting anomalies in musculoskeletal radiographs, e.g., fractures can help the radiologists to speed up the radiology workflow. Supervised deep networks take for granted a large number of annotations by radiologists, which is often prohibitively very time-consuming to acquire. Moreover, these supervised systems are tailored to closed set scenarios, e.g., trained models suffer from overfitting to previously seen rare anomalies at training. Instead, our approach's rationale is using contrastive learning built upon the deep semi-supervised framework to produce representation in the embedding space, which brings together similar images and pushes apart dissimilar images. Besides, we formulate a complex distribution of normal data within our framework to avoid a potential bias on the side of anomalies. Through extensive experiments, we show that our method outperforms baselines across unsupervised and self-supervised anomaly detection settings on a real-world medical dataset, the MURA dataset.

<img src="Outputs/architecture_network_CDMSAD.png" alt="method" width="500"/>

____
## Repository Organisation
```
Code/
├── scripts                                               
│   ├── AE                                      
│   │   ├── AE_DMSAD_config.json                > config file Multi-modal experiment
│   │   ├── AE_DMSAD_scripts.py                 > Multi-modal experiment scripts
│   │   ├── AE_DSAD_config.json                 > config file Uni-modal experiment (Ruff et al.)
│   │   └── AE_DSAD_scripts.py                  > Uni-modal experiment scripts (Ruff et al.)
│   ├── Ablation
│   │   ├── Ablation.py                         > Ablation experiment scripts         
│   │   └── Ablation_config.json                > ablation exeriment config file
│   ├── Contrastive                             
│   │   ├── CDMSAD_config.json                  > config file Proposed method (Multi-Modal)
│   │   ├── CDMSAD_scripts.py                   > Proposed method (Multi-modal) experiment scripts
│   │   ├── CDSAD_config.json                   > config file Proposed method (Uni-Modal)
│   │   └── CDSAD_scripts.py                    > Proposed method (Uni-modal) experiment scripts
│   ├── postprocessing
│   │   └── process_experiement.py              > scripts to analyse an experiments
│   └── preprocessing
│       ├── generate_data_info_script.py        > scripts to generate the MURA data csv
│       └── preprocessing_script.py             > scripts to preprocess the MURA dataset
└── src
    ├── __init__.py
    ├── datasets
    │   ├── MURADataset.py                      > torch.Dataset classes for the MURA dataset
    │   ├── __init__.py
    │   └── transforms.py                       > Online transformations classes
    ├── models
    │   ├── AE_DMSAD.py                         > AE baseline Multi-Modal model
    │   ├── AE_DSAD.py                          > AE baseline Uni-Modal model
    │   ├── CDMSAD.py                           > Proposed method Multi-Modal model
    │   ├── CDSAD.py                            > Proposed method Uni-Modal model
    │   ├── NearestNeighbors.py                 > Nearest Neighbors abaltion model
    │   ├── __init__.py
    │   ├── networks
    │   │   ├── Network_Modules.py              > Network building blocks modules
    │   │   ├── Networks.py                     > Network classes
    │   │   └── __init__.py
    │   └── optim
    │       ├── AE_trainer.py                   > trainer classe for the AE
    │       ├── Contrastive_trainer.py          > trainer classe for the contrastive task
    │       ├── DMSAD_trainer.py                > trainer for the semi-supervised Multi-modal anomaly detection
    │       ├── DSAD_trainer.py                 > trainer for the semi-supervised uni-modal anomaly detection (Ruff et al.)
    │       ├── Loss_Functions.py               > loss functions modules classes
    │       └── __init__.py
    ├── preprocessing
    │   ├── __init__.py
    │   ├── cropping_rect.py                    > functions to extract X-ray image carrier
    │   ├── get_data_info.py                    > functions to make the MURA info csv
    │   └── segmentation.py                     > functions to segment the body parts
    └── utils
        ├── Config.py                           > classe defining a config file
        ├── __init__.py
        ├── plot_utils.py                       > utility functions for plotting
        └── utils.py                            > general utility functions
```
----
### Licence
MIT
