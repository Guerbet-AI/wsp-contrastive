from os.path import join
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from dataset import Dataset
from sklearn import svm


class Config:

    def __init__(self,
                 mode: str = 'finetuning',
                 rep_dim: int = 512,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 num_classes: int = 4,
                 encoder: str = 'tiny',
                 n_layer: int = 18,
                 lr: float = 1e-5,
                 weight_decay: float = 1e-5,
                 label_name: str = 'label',
                 n_fold: int = 4,
                 cross_val: bool = False,
                 pretrained_path: str = None,
                 sigma: float = 0.85,
                 temperature: float = 0.1,
                 kernel: str = 'rbf',
                 max_epochs: int = 40,
                 batch_size: int = 64,
                 pretrained: bool = False,
                 path_to_data: str = "path_to_data",
                 lght_dir: str = "path_to_models"):

        """PyTorch Dataset Module for training and validation
        Keywords arguments:
        :param mode: finetuning for classification and pretraining for pretraining models.
        :param rep_dim: representation space dimension.
        :param hidden_dim: hidden space dimension (between rep and out spaces).
        :param output_dim: output space dimension (space where loss is computed for CL)
        :param num_classes: number of classes in the discrete variable y.
        :param encoder: type of encoder: TinyNet or ResNet.
        :param lr: learning rate.
        :param weight_decay: weight decay for learning.
        :param label_name: y data name for WSP loss.
        :param n_fold: number of folds if cross-validation procedure in finetuning.
        :param cross_val: called if cross-validation.
        :param pretrained_path: path to pretrained model if you run model with pretrained weights.
        :param sigma: sigma parameter in the Gaussian kernel of the WSP loss.
        :param temperature: tau parameter in the original NTXentLoss.
        :param kernel: type of kernel used for variable d in the WSP loss.
        :param max_epochs: number of epochs during training.
        :param batch_size: batch size for training.
        :param pretrained: boolean to indicate if you used the pretrained weights from ImageNet (if using ResNet).
        :param path_to_data: path to where the dataframe and the nifty scans are stored.
        :return: a PyTorch Dataset
        """

        assert mode in {"pretraining", "finetuning", "test"}, "Unknown mode: %i"%mode

        self.mode = mode
        self.input_size = (1, 512, 512)
        self.nb_cpu = 6
        self.scheduler = 0.987
        self.path_to_data = path_to_data
        self.lght_dir = lght_dir

        # Encoder parameters
        self.input_dim = 1
        self.encoder = encoder
        self.pretrained = pretrained
        self.n_layer = n_layer
        self.rep_dim = rep_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_classes = num_classes
        # Cross-validation parameters
        self.cross_val = cross_val
        self.n_fold = n_fold
        self.val_rate = 1.0

        # Optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        # Checkpoint
        self.pretrained_path = pretrained_path
        self.train_set = 'train_set'
        self.val_set = 'val_set'

        if self.mode == "pretraining":

            self.label_name = label_name
            self.pin_mem = True

            # Hyperparameters for our WSP loss
            self.kernel = kernel
            self.sigma = sigma
            self.temperature = temperature


        elif self.mode == "finetuning":

            # Saving csv files
            self.df_train_name = 'finetuning_train.csv'
            self.df_val_name = 'finetuning_val.csv'