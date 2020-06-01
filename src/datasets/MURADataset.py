import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as TF
from torch.utils import data
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable

import src.datasets.transforms as tf

class MURA_Dataset(data.Dataset):
    """
    Implementation of a torch.utils.data.Dataset for the MURA-dataset that can
    handle the loading of a mask and semi.supervized labels.
    """
    def __init__(self, sample_df, data_path, mask_img=True, output_size=512,
                 data_augmentation=True):
        """
        Constructor of the dataset.
        ----------
        INPUT
            |---- sample_df (pandas.DataFrame) the dataframe containing the
            |           samples' filenames, labels, (semi-labels and mask-filename).
            |---- data_path (str) the path to the data filenames specified in the sample_df
            |---- mask_img (bool) whether to mask the image.
            |---- output_size (int) the size of the output squared image.
            |---- data_augmentation (bool) whether to perform data augmentation.
        """
        data.Dataset.__init__(self)
        self.sample_df = sample_df
        self.data_path = data_path
        if data_augmentation:
            self.transform = tf.Compose(tf.Grayscale(), \
                                        tf.AutoContrast(cutoff=1), \
                                        tf.RandomHorizontalFlip(p=0.5), \
                                        tf.RandomVerticalFlip(p=0.5), \
                                        tf.RandomBrightness(lower=0.8, upper=1.2), \
                                        tf.RandomScaling(scale_range=(0.8,1.2)), \
                                        tf.RandomRotation(degree_range=(-20,20)), \
                                        tf.ResizeMax(output_size), \
                                        tf.PadToSquare(), \
                                        tf.MinMaxNormalization(), \
                                        tf.MaskImage(mask_img), \
                                        tf.ToTorchTensor())
        else:
            self.transform = tf.Compose(tf.Grayscale(), \
                                        tf.AutoContrast(cutoff=1), \
                                        tf.ResizeMax(output_size), \
                                        tf.PadToSquare(), \
                                        tf.MinMaxNormalization(), \
                                        tf.MaskImage(mask_img), \
                                        tf.ToTorchTensor())

    def __len__(self):
        """
        Get the number of samples in the dataset.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- len (int) the number of samples.
        """
        return self.sample_df.shape[0]

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        ----------
        INPUT
            |---- idx (int) the index of the sample to get (i.e. row of the
            |           self.sample_df get).
        OUTPUT
            |---- image (torch.Tensor) the image with online transformations.
            |---- label (torch.Tensor) the label (normal vs abnormal)
            |---- mask (torch.Tensor) the mask with online transfromations.
            |---- semi_label (torch.Tensor) the semi-supervised labels (normal-known,
            |           abnormal-known or unknown)
        """
        im = Image.open(self.data_path + self.sample_df.loc[idx,'filename'])
        # load label
        label = torch.tensor(self.sample_df.loc[idx,'abnormal_XR'])
        # load mask
        mask = Image.open(self.data_path + self.sample_df.loc[idx,'mask_filename'])
        # load semi-label
        semi_label = torch.tensor(self.sample_df.loc[idx, 'semi_label'])
        # Online transformation
        im, mask = self.transform(im, mask)

        return im, label, mask, semi_label, torch.tensor(idx)

class MURADataset_SimCLR(data.Dataset):
    """
    MURA dataset for the SimCLR model which return two replicate of the image
    with heavy data augmentation.
    """
    def __init__(self, sample_df, data_path, output_size=512, mask_img=True):
        """
        Constructor of the dataset.
        ----------
        INPUT
            |---- sample_df (pandas.DataFrame) the dataframe containing the
            |           samples' filenames, labels, (semi-labels and mask-filename).
            |---- data_path (str) the path to the data filenames specified in the sample_df
            |---- output_size (int) the size of the output squared image.
            |---- mask_img (bool) whether to mask image.
        """
        data.Dataset.__init__(self)
        self.sample_df = sample_df
        self.data_path = data_path

        self.transform = tf.Compose(tf.Grayscale(),
                                    tf.AutoContrast(cutoff=1),
                                    tf.RandomHorizontalFlip(p=0.5),
                                    tf.RandomScaling(scale_range=(0.8,1.2)),
                                    tf.RandomRotation(degree_range=(-45,45)),
                                    tf.ResizeMax(max_len=output_size),
                                    tf.PadToSquare(),
                                    tf.RandomCropResize((output_size, output_size), scale=(0.4, 1.0), ratio=(4./5., 5./4.)),
                                    tf.ColorDistorsion(s=0.5),
                                    tf.GaussianBlur(p=0.5, sigma=(0.1, 2.0)),
                                    tf.MinMaxNormalization(),
                                    tf.MaskImage(mask_img),
                                    tf.ToTorchTensor())

    def __len__(self):
        """
        Get the number of samples in the dataset.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- len (int) the number of samples.
        """
        return self.sample_df.shape[0]

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        ----------
        INPUT
            |---- idx (int) the index of the sample to get (i.e. row of the
            |           self.sample_df get).
        OUTPUT
            |---- image_1 (torch.Tensor) the image transformed once.
            |---- image_2 (torch.Tensor) the same image transfromed differently.
            |---- semi_label (torch.Tensor) the semi-supervised label. 0 if unknown,
            |           1 if known normal and -1 if known abnormal.
        """
        im = Image.open(self.data_path + self.sample_df.loc[idx,'filename'])
        mask = Image.open(self.data_path + self.sample_df.loc[idx,'mask_filename'])
        semi_label = torch.tensor(self.sample_df.loc[idx,'semi_label'])

        im1, _ = self.transform(im, mask)
        im2, _ = self.transform(im, mask)

        return im1, im2, semi_label, idx

class MURA_TrainValidTestSplitter:
    """
    A train-validation-test splitter for the MURA-dataset splitting at the level
    of patient's bodypart to avoid test leakage.
    """
    def __init__(self, data_info, train_frac=0.5,
                 ratio_known_normal=0.0, ratio_known_abnormal=0.0, random_state=42):
        """
        Constructor of the splitter.
        ----------
        INPUT
            |---- data_info (pd.DataFrame) the whole data with columns generated
            |           with generate_data_info function
            |---- train_frac (float) define the fraction of the data to use as
            |           train set. The train set is mostly composed of normal
            |           sample. There must thus be enough of them.
            |---- ratio_known_normal (float) the fraction of knwon normal samples
            |---- ratio_known_abnormal (float) the fraction of knwon abnormal samples
            |---- random_state (int) the seed for reproducibility
        OUTPUT
            |---- None
        """
        # input
        self.subsets = {}
        self.data_info = data_info
        assert train_frac <= 1, f'Input Error. The train fraction must larger than one. Here it is {train_frac}'
        self.train_frac = train_frac
        assert ratio_known_normal <= 1, f'Input Error. The ratio_known_normal must be smaller than one. Here it is {ratio_known_normal}'
        self.ratio_known_normal = ratio_known_normal
        assert ratio_known_abnormal <= 1, f'Input Error. The ratio_known_abnormal must be smaller than one. Here it is {ratio_known_abnormal}'
        self.ratio_known_abnormal = ratio_known_abnormal
        self.random_state = random_state

    def split_data(self, verbose=False):
        """
        Split the MURA dataset into a train, validation and test sets. To avoid
        test leakage, the split is made at the level of patients bodypart (all
        XR from a patient's hand will be on the same set).
        1) The train contains train_frac samples, of which ratio_known_abnormal
        are abnormal and the rest are normal XR. All the abnormal train XR are
        considered as known.
        2) the rest of the normal/mixt and abnormal are equally shared between
        the validation and test set. Mixt patient bodypart (patient hand with both
        normal and abnormal XR) are considered abnormal for the spliting.
        3) For each set, a fraction of normal and abnormal (in the case of the
        train, all abnormal are considered as known) are labeled. The resulting
        semi-supervised labelling is : 0 = unknown ; 1 = known normal ;
        -1 = known abnormal
        ----------
        INPUT
            |---- verbose (bool) whether to display a summary at the end
        OUTPUT
            |---- None
        """
        # group sample by patient and body part
        tmp = self.data_info.groupby(['patientID', 'body_part']).max()
        # get the index (i.e. patient and bodypart) where none of the body part XR of a given patient are abnormal
        idx_list_normal = tmp[tmp.body_part_abnormal == 0].index.to_list()
        # get the index (i.e. patient and bodypart) where at least one but not all of the body part XR of a given patient are abnormal
        idx_list_mixt = tmp[tmp.body_part_abnormal == 0.5].index.to_list()
        # get the index (i.e. patient and bodypart) where all one of the body part XR of a given patient are abnormal
        idx_list_abnormal = tmp[tmp.body_part_abnormal == 1].index.to_list()
        total = len(idx_list_normal)+len(idx_list_mixt)+len(idx_list_abnormal)
        train_size = self.train_frac*total
        assert train_size < len(idx_list_normal), f'There are not enough normal sample for the given train_frac : {self.train_frac}. \
                                                    There are {len(idx_list_normal)} normal sample over {total} total samples.'
        valid_size = (1-self.train_frac)*0.5*total
        test_size = (1-self.train_frac)*0.5*total
        # randomly pick (1-ratio_known_abnormal)*train_frac*total from the normal index for the train set
        train_idx_normal, remain = train_test_split(idx_list_normal, \
                                                    train_size=int((1-self.ratio_known_abnormal)*train_size),\
                                                    random_state=self.random_state)
        # split the rest equally in the validation and test set
        valid_idx_normal, test_idx_normal = train_test_split(remain, test_size=0.5, random_state=self.random_state)
        # add ratio_known_abnormal*train_frac*total from the abnormal index
        if self.ratio_known_abnormal == 0.0:
            train_idx_abnormal, remain = [], idx_list_abnormal
        else:
            train_idx_abnormal, remain = train_test_split(idx_list_abnormal, \
                                                          train_size=int(self.ratio_known_abnormal*train_size),\
                                                          random_state=self.random_state)
        # split the rest equally in the validation and test set
        valid_idx_abnormal, test_idx_abnormal = train_test_split(remain, test_size=0.5, random_state=self.random_state)
        # split the mixt between test and validation and consider them as abnormal patients bodypart
        valid_idx_mixt, test_idx_mixt = train_test_split(idx_list_mixt, test_size=0.5, random_state=self.random_state)
        valid_idx_abnormal += valid_idx_mixt
        test_idx_abnormal += test_idx_mixt
        # get the known and unknown index for each sets
        # get a fraction of normal known
        if self.ratio_known_normal == 0.0:
            train_idx_known, train_idx_unknown = [], train_idx_normal
            valid_idx_known, valid_idx_unknown = [], valid_idx_normal
            test_idx_known, test_idx_unknown = [], test_idx_normal
        else:
            train_idx_known, train_idx_unknown = train_test_split(train_idx_normal, \
                                                            train_size=int(self.ratio_known_normal*train_size),\
                                                            random_state=self.random_state)
            valid_idx_known, valid_idx_unknown = train_test_split(valid_idx_normal, \
                                                            train_size=int(self.ratio_known_normal*valid_size),\
                                                            random_state=self.random_state)
            test_idx_known, test_idx_unknown = train_test_split(test_idx_normal, \
                                                            train_size=int(self.ratio_known_normal*test_size), \
                                                            random_state=self.random_state)
        # get the abnormal known
        # all abnormal in train are known
        train_idx_known += train_idx_abnormal
        if self.ratio_known_abnormal == 0.0:
            valid_idx_unknown += valid_idx_abnormal
            test_idx_unknown += test_idx_abnormal
        else:
            valid_idx_known_abnormal, valid_idx_unknown_abnormal = train_test_split(valid_idx_abnormal, \
                                                                        train_size=int(self.ratio_known_abnormal*valid_size), \
                                                                        random_state=self.random_state)
            valid_idx_known += valid_idx_known_abnormal
            valid_idx_unknown += valid_idx_unknown_abnormal
            test_idx_known_abnormal, test_idx_unknown_abnormal = train_test_split(test_idx_abnormal, \
                                                                        train_size=int(self.ratio_known_abnormal*test_size),\
                                                                        random_state=self.random_state)
            test_idx_known += test_idx_known_abnormal
            test_idx_unknown += test_idx_unknown_abnormal

        # get the subsample dataframe with semi-label
        train_df = self.generate_semisupervized_label(train_idx_known, train_idx_unknown)
        valid_df = self.generate_semisupervized_label(valid_idx_known, valid_idx_unknown)
        test_df = self.generate_semisupervized_label(test_idx_known, test_idx_unknown)
        # shuffle the dataframes
        self.subsets['train'] = train_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        self.subsets['valid'] = valid_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        self.subsets['test'] = test_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        # Print summary
        if verbose:
            self.print_stat()

    def generate_semisupervized_label(self, idx_known, idx_unknown):
        """
        Assigne the semi-supervized labels at the given indices
        0 = unknown ; 1 = known normal ; -1 = known abnormal
        ----------
        INPUT
            |---- idx_known (list of tuple) the multiindex (patient, bodypart)
            |           of the known elements
            |---- idx_unknown (list of tuple) the multiindex (patient, bodypart)
            |           of the unknown elements
        OUTPUT
            |---- df (pd.DataFrame) the dataframe at the passed index with semi-
            |           supervised labels
        """
        tmp_df = self.data_info.set_index(['patientID','body_part'])
        # associate semi-supervized settings
        if len(idx_known) > 0:
            df_known = tmp_df.loc[idx_known,:]
            df_known['semi_label'] = df_known.abnormal_XR.apply(lambda x: -1 if x==1 else 1)
            df_unknown = tmp_df.loc[idx_unknown,:]
            df_unknown['semi_label'] = 0
            return pd.concat([df_known, df_unknown], axis=0).reset_index()
        else:
            df_unknown = tmp_df.loc[idx_unknown,:]
            df_unknown['semi_label'] = 0
            return df_unknown.reset_index()

    def get_subset(self, name):
        """
        Return the data subset requested.
        ----------
        INPUT
            |---- name (str) the subset name : 'train', 'valid' or 'test'
        OUTPUT
            |---- subset (pd.DataFrame) the subset dataframe with semi-lables
        """
        assert name in ['train', 'valid', 'test'], f'Invalid dataset name! {name} has been provided but must be one of [train, valid, test]'
        return self.subsets[name]

    def print_stat(self, returnTable=False):
        """
        Display a summary table of the splitting with the number and fractions of
        normal and abnormal, as well as the number and fraction of semisupervized
        labels.
        ----------
        INPUT
            |---- returnTable (bool) whether to return table as a string.
        OUTPUT
            |---- (summary) (str) the table as a string if returnTable = True.
        """
        summary = PrettyTable(["Set", "Name", "Number [-]", "Fraction [%]"])
        summary.align = 'l'
        for name, df in self.subsets.items():
            summary.add_row([name, 'Normal', df[df.abnormal_XR == 0].shape[0], '{:.2%}'.format(df[df.abnormal_XR == 0].shape[0] / df.shape[0])])
            summary.add_row([name, 'Abnormal', df[df.abnormal_XR == 1].shape[0], '{:.2%}'.format(df[df.abnormal_XR == 1].shape[0] / df.shape[0])])
            summary.add_row([name, 'Normal known', df[df.semi_label == 1].shape[0], '{:.2%}'.format(df[df.semi_label == 1].shape[0] / df.shape[0])])
            summary.add_row([name, 'Abnormal known', df[df.semi_label == -1].shape[0], '{:.2%}'.format(df[df.semi_label == -1].shape[0] / df.shape[0])])
            summary.add_row([name, 'Unknown', df[df.semi_label == 0].shape[0], '{:.2%}'.format(df[df.semi_label == 0].shape[0] / df.shape[0])])
            if name != 'test' : summary.add_row(['----']*4)
        if returnTable:
            return summary
        else:
            print(summary)
