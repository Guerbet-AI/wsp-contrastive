import sklearn.metrics
import torch
from torch.optim.lr_scheduler import ExponentialLR

import pytorch_lightning as pl
from torch.autograd import Variable
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler, Subset
from dataset import Dataset
from sklearn.decomposition import PCA
import time
import cv2
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, balanced_accuracy_score, roc_curve, auc
import numpy as np
from dataset import DatasetCL
import seaborn as sns
import pandas as pd
import itertools
from os.path import join
from os import listdir
from collections import Counter
from torch import Tensor
import models.network as model
import kornia



class DataAugmentation(torch.nn.Module):

    """Module to perform data augmentation using Kornia on torch tensors.
    :return: a torch tensor with Kornia GPU augmentation module."""

    def __init__(self) -> None:
        super().__init__()

        self.transforms = torch.nn.Sequential(
            #kornia.augmentation.RandomGaussianNoise(p=0.5, mean=0.0, std=0.1),
            #kornia.augmentation.RandomGaussianBlur(p=0.5, kernel_size=(3, 3), sigma=(0.1, 2)),
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            kornia.augmentation.RandomVerticalFlip(p=0.5),
            #kornia.augmentation.RandomErasing(p=0.5, scale=(0.05, 0.05), ratio=(1, 1)),
            kornia.augmentation.RandomResizedCrop(p=0.5, size=(512, 512), scale=(0.7, 0.7)),
            kornia.augmentation.RandomAffine(p=0.5, degrees=0, translate=(0.2, 0.2)),
            kornia.augmentation.RandomRotation(p=0.5, degrees=30)
        )

    @torch.no_grad()                # disable gradients for efficiency
    def forward(self, x: Tensor) -> Tensor:
        x_out = self.transforms(x)  # BxCxHxW
        return x_out


def cutoff_youdens_j(fpr,tpr,thresholds):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]



class wspContrastiveModel(pl.LightningModule):

    def __init__(self, net, loss, config, data_train, data_val):

        """PyTorch Lightning model
        Keywords arguments:
        :param net: type of network used for training
        :param loss: loss used for training
        :param loader_train: PyTorch DataLoader for training
        :param loader_val: PyTorch DataLoader for validation
        :param config: Config object with hyperparameters
        :return: a pl module
        """

        super().__init__()
        self.loss = loss
        self.net = net
        self.config = config
        self.data_train = data_train
        self.data_val = data_val
        self.mode = config.mode
        self.transforms = DataAugmentation()

        self.nb_steps = len(data_val) // self.config.batch_size + 1
        self.val_loss_step = 0
        self.cutoff = 0.5
        self.vols_train = []

        assert self.mode in ['finetuning', 'pretraining', 'test'], ('self.mode =', self.mode)

        if self.config.mode == 'finetuning':

            # TRAINING PART
            index_train = self.data_train.labels.index                                # list of the training volumes
            self.df_train = pd.DataFrame({'epoch_'+str(i): [np.zeros(self.config.num_classes) for x in index_train] for i in range(500)},
                                         index=index_train)
                                                                                      # df_train: contains the avg of the probabilities by patient over the whole epoch
            dic_train = dict(Counter(self.data_train.volumes))                        # a dictionary that maps the volumes to their occurrences inside the training dataset
            self.df_train['nb_slices'] = [dic_train[x] for x in self.df_train.index]  # nb of slices associated to each volume in the training dataset
            self.df_train['class'] = self.data_train.labels['class']                  # class associated to each volume in the training dataset
            self.df_train['label'] = self.data_train.labels['label']                  # label (binarized class) associated to each volume in the training dataset

            # VALIDATION PART
            index_val = self.data_val.labels.index
            self.df_val = pd.DataFrame({'epoch_'+str(i): [np.zeros(self.config.num_classes) for x in index_val] for i in range(500)},
                                       index=index_val)

            dic_val = dict(Counter(self.data_val.volumes))                            # a dictionary that maps the volumes to their occurrences inside the validation dataset
            self.df_val['nb_slices'] = [dic_val[x] for x in self.df_val.index]        # nb of slices associated to each volume in the validation dataset
            self.df_val['class'] = self.data_val.labels['class']                      # class associated to each volume in the validation dataset
            self.df_val['label'] = self.data_val.labels['label']                      # label (binarized class) associated to each volume in the validation dataset


        if hasattr(config, 'pretrained_path') and config.pretrained_path is not None:
            self.load_model(config.pretrained_path)


    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        # discard the version number
        items.pop("v_num", None)
        return items

    def forward(self, x, mode=None):
        y = self.net(x, mode)
        return y

    def training_step(self, batch, batch_idx):

        inputs, labels, subjects_id, z = batch          # batch : N x (augmented images (1 et 2), label (cirrhosis), subject_id, z-position)
                                                        # inputs shape      (batch_size,1 or 3,1,512,512)
                                                        # labels shape      (batch_size,1)
                                                        # subjects_id shape (batch_size,1)
                                                        # z shape (batch_size,1)

        assert self.mode in ['finetuning', 'pretraining'], ('self.mode =', self.mode)

        if self.mode == "pretraining":

            aug_i, aug_j = self.transforms(inputs / 250.), self.transforms(inputs / 250.)

            ## Forward pass
            z_i = self(aug_i)  # z_i : first augmented image
            z_j = self(aug_j)  # z_j : second augmented image
            # aug_i : shape (batch_size, 1, 512, 512)
            # aug_j : shape (batch_size, 1, 512, 512)
            # z_i : shape (batch_size,d=128)
            # z_j : shape (batch_size,d=128)

            ## Compute the Loss
            loss, logits, target = self.loss(z_i, z_j, labels, z)



            if batch_idx == 0:
                self.logger.experiment.add_image("Images/First augmentation", (aug_i)[batch_idx, :].cpu(),
                                                 self.current_epoch)
                self.logger.experiment.add_image("Images/Second augmentation", (aug_j)[batch_idx, :].cpu(),
                                                 self.current_epoch)


        elif self.mode == "finetuning":

            aug = self.transforms(inputs / 250.)                # inputs shape: (batch_size,1,512,512)

            ## Forward pass
            y = self(aug)                                       # y shape (batch_size,C)
            ## Compute the Loss
            loss = self.loss(y, labels)                         # loss : cross-entropy loss

            ## Output probabilities of class 1 on the batch
            y_prob = softmax(y, dim=1).detach().cpu().numpy()   # shape (batch_size,C)
            ## Fill the epoch-level metrics dataframe : we compute the average probability over the whole epoch
            ep = 'epoch_' + str(self.current_epoch)
            for i, subject in enumerate(subjects_id):
                for k in range(self.config.num_classes):
                    self.df_train.loc[subject, ep][k] += np.nan_to_num(y_prob[i, k])

            if batch_idx == 0:
                self.logger.experiment.add_image("Images/OriginalImage",
                                                 (inputs / 250.)[batch_idx, :].cpu(),
                                                 self.current_epoch)
                self.logger.experiment.add_image("Images/Augmentation", (aug)[batch_idx, :].cpu(),
                                                 self.current_epoch)


        # accumulate the sampled volumes through the epoch
        self.vols_train.append(subjects_id)

        ### LOGS ###
        self.logger.experiment.add_scalar("Training/training_loss_step", loss, self.global_step)
        self.logger.experiment.add_scalars("TrainVal/training_val_loss_step", {'train': loss}, self.global_step)
        ## we log the loss of the first batch as the training loss at iteration 0 (before optimization)
        if self.global_step == 0:
            self.logger.experiment.add_scalar("Training/training_loss_epoch", loss, 0)
            self.logger.experiment.add_scalars("TrainVal/training_val_loss_epoch", {'train': loss}, 0)
        ### END LOGS ###

        return loss


    def training_epoch_end(self, outputs):

        train_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean().item()
        self.log('tl_epoch', train_loss_epoch)

        if self.mode == 'pretraining':
            self.logger.experiment.add_scalar("Training/training_loss_epoch", train_loss_epoch, self.current_epoch)
            self.logger.experiment.add_scalars("TrainVal/training_val_loss_epoch", {'train': train_loss_epoch}, self.current_epoch)

        elif self.mode == 'finetuning':

            # in training mode, the number of slices per patient that are sampled during each epoch vary,
            # so the 'nb_slices' data is being updated at each end of epoch to obtain the loss by subject.

            vols_train = list(itertools.chain(*self.vols_train))
            dic_train = dict(Counter(vols_train))
            self.df_train['nb_slices'] = [dic_train[x] if x in list(dic_train.keys()) else 1 for x in
                                          self.df_train.index]
            self.df_train['epoch_' + str(self.current_epoch)] /= self.df_train['nb_slices']
            self.vols_train = []

            y_true = torch.tensor(self.df_train['label'], dtype=torch.long)
            y_pred = torch.tensor(np.vstack(self.df_train['epoch_' + str(self.current_epoch)])[:, 1])
            fpr_train, tpr_train, thresholds = roc_curve(y_true, y_pred)
            self.cutoff = cutoff_youdens_j(fpr_train, tpr_train, thresholds)

            ## Training loss per subject
            loss_vector = torch.mul(torch.log(y_pred), y_true) + torch.mul(torch.log(1 - y_pred), 1 - y_true)

            train_loss_sub = -torch.mean(torch.nan_to_num(loss_vector, neginf=0, posinf=0))             # shape 1
            ### ROC-AUC score per subject
            train_auc = roc_auc_score(y_true, y_pred)
            ### END METRICS ###

            ### LOGS ###
            self.logger.experiment.add_scalar("Training/training_AUC", train_auc, self.current_epoch)
            self.logger.experiment.add_scalar("Training/training_loss_epoch", train_loss_epoch, self.current_epoch)
            self.logger.experiment.add_scalar("Training/training_loss_subject", train_loss_sub, self.current_epoch)
            self.logger.experiment.add_scalars("TrainVal/training_val_loss_epoch", {'train': train_loss_epoch},
                                               self.current_epoch)
            self.logger.experiment.add_scalars("TrainVal/training_val_loss_subject", {'train': train_loss_sub},
                                               self.current_epoch)
            ### END LOGS ###



    def validation_step(self, batch, batch_idx):

        inputs, labels, subjects_id, z = batch          # batch : N x (augmented images (1 and 2), label (cirrhosis), subject_id, z-position)
                                                        # inputs shape      (batch_size, 1 or 3, 1, 512,512)
                                                        # labels shape      (batch_size)
                                                        # subjects_id       (batch_size)
                                                        # z       (batch_size)


        assert self.mode in ['finetuning', 'pretraining'], ('self.mode =', self.mode)

        if self.mode == "pretraining":

            aug_i, aug_j = self.transforms(inputs / 250.), self.transforms(inputs / 250.)

            ## Forward pass
            z_i = self(aug_i)  # z_i : first augmented image
            z_j = self(aug_j)  # z_j : second augmented image
            # aug_i : shape (batch_size, 1, 512, 512)
            # aug_j : shape (batch_size, 1, 512, 512)
            # z_i : shape (batch_size,d=128)
            # z_j : shape (batch_size,d=128)

            ## Compute the Loss
            val_loss_step, logits, target = self.loss(z_i, z_j, labels, z)

            self.val_loss_step += val_loss_step / self.nb_steps          # for the log of the avg
            # no separation with the sanity check because we run the forward on the whole validation set

            if batch_idx == 0:
                self.logger.experiment.add_image("Images/First augmentation_val", (aug_i)[batch_idx, :].cpu(),
                                                 self.current_epoch)
                self.logger.experiment.add_image("Images/Second augmentation_val", (aug_j)[batch_idx, :].cpu(),
                                                 self.current_epoch)


        elif self.mode == "finetuning":

            ## Forward pass
            y = self(inputs / 250.)                             # y shape (batch_size,2)
            ## Compute the Loss
            val_loss_step = self.loss(y, labels)                # loss : cross-entropy loss
            self.val_loss_step += val_loss_step / self.nb_steps # for the log of the avg

            ## Output probabilities of class 1 on the batch
            y_prob = softmax(y, dim=1).detach().cpu().numpy()

            ## Fill the epoch-level metrics dataframe : we compute the average probability over the whole epoch
            ep = 'epoch_' + str(self.current_epoch)
            for i, subject in enumerate(subjects_id):
                for k in range(self.config.num_classes):
                    self.df_val.loc[subject, ep][k] += np.nan_to_num(
                        y_prob[i, k] / self.df_val.loc[subject, 'nb_slices'])

        ### LOGS ###
        self.logger.experiment.add_scalar("Validation/val_loss_step", self.val_loss_step, self.global_step+1)
        self.logger.experiment.add_scalars("TrainVal/training_val_loss_step", {'val': self.val_loss_step},
                                           self.global_step+1)
        # we log the first forward pass on 2 batches from the sanity check before any optimization
        if self.trainer.sanity_checking:
            self.logger.experiment.add_scalar("Validation/val_loss_epoch", self.val_loss_step, 0)
            self.logger.experiment.add_scalars("TrainVal/training_val_loss_epoch", {'val': self.val_loss_step}, 0)
        ### END LOGS ###

        return val_loss_step

    def validation_epoch_end(self, outputs):

        self.val_loss_step = 0
        val_loss_epoch = np.mean([x.item() for x in outputs])
        self.log('vl_epoch',val_loss_epoch)

        if self.mode == 'pretraining':

            if self.trainer.sanity_checking:
                self.logger.experiment.add_scalar("Validation/val_loss_epoch", val_loss_epoch, 0)
                self.logger.experiment.add_scalars("TrainVal/training_val_loss_epoch", {'val': val_loss_epoch}, 0)

            else:
                self.logger.experiment.add_scalar("Validation/val_loss_epoch", val_loss_epoch, self.current_epoch + 1)
                self.logger.experiment.add_scalars("TrainVal/training_val_loss_epoch", {'val': val_loss_epoch},
                                               self.current_epoch + 1)


        elif self.mode == 'finetuning':       # loss_P = 1/nb_slices_P sum_{s=1^S} Softmax(Net(slices_P_s))

            if self.config.cross_val:
                self.df_val.to_csv(join(self.config.lght_dir, self.config.df_val_name+str(self.config.cv_fold)), sep=';', index='subject')
            else:
                self.df_val.to_csv(join(self.config.lght_dir, self.config.df_val_name),
                                   sep=';', index='subject')

            y_true = torch.tensor(self.df_val['label'], dtype=torch.long)
            y_pred = torch.tensor(np.vstack(self.df_val['epoch_' + str(self.current_epoch)])[:, 1])

            ## Training loss per subject
            loss_vector = torch.mul(torch.log(y_pred), y_true) + torch.mul(torch.log(1 - y_pred), 1 - y_true)
            val_loss_sub = -torch.mean(torch.nan_to_num(loss_vector, neginf=0, posinf=0))   # shape 1
            ### ROC-AUC score per subject
            val_auc = roc_auc_score(y_true, y_pred)
            ## Balanced Accuracy per subject
            y_pred_youden = [1 if x > self.cutoff else 0 for x in y_pred]
            val_acc = balanced_accuracy_score(y_true, y_pred_youden)
            ### END METRICS ###

            ### LOGS ###

            self.log("val_auc", val_auc) # for monitoring

            self.logger.experiment.add_scalar("Validation/val_loss_epoch", val_loss_epoch, self.current_epoch)
            self.logger.experiment.add_scalar("Validation/val_loss_subject", val_loss_sub, self.current_epoch)
            self.logger.experiment.add_scalar("Validation/val_AUC", val_auc, self.current_epoch)
            self.logger.experiment.add_scalar("Validation/val_accuracy", val_acc, self.current_epoch)
            self.logger.experiment.add_scalars("TrainVal/training_val_loss_epoch", {'val': val_loss_epoch},
                                               self.current_epoch)
            self.logger.experiment.add_scalars("TrainVal/training_val_loss_subject", {'val': val_loss_sub},
                                               self.current_epoch)

            ### END LOGS ###


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.config.max_epochs, eta_min=0, last_epoch=-1
        )

        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                "monitor":'vl_epoch'}

    def load_model(self, checkpoint):

        if checkpoint is not None:

            checkpoint_ = torch.load(checkpoint)

            model_dict = self.state_dict()
            new_state_dict = checkpoint_['state_dict']

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.load_state_dict(pretrained_dict, strict=False)
            print("Pretrained model loaded!")
