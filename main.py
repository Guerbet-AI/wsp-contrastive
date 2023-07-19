#import torch.cuda


from dataset import DatasetCL
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler
from losses import WSPContrastiveLoss
from torch.nn import CrossEntropyLoss
import itertools
import models.network as model_
from sampler import CustomSampler
import argparse
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from config import Config
from wspContrastiveLearning import wspContrastiveModel
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, train_test_split
import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from os.path import join
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
import warnings


if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning", "autoencoder"], required=True,
                        help="Set the training mode. Do not forget to configure config.py accordingly!")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--cross_val", dest="cross_val", action="store_true")
    parser.add_argument("--no-cross_val", dest="cross_val", action="store_false")
    parser.add_argument("--n_fold", type=int)
    parser.add_argument("--encoder", type=str)
    parser.add_argument("--n_layer", type=int)
    parser.add_argument("--pretrained_path", default=None, type=str)
    parser.add_argument("--sigma", default=0.5, type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--label_name", default="label", type=str)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--kernel", default='rbf', type=str)
    parser.add_argument("--max_epochs", default=40, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--rep_dim", default=256, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--output_dim", default=64, type=int)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")

    args = parser.parse_args()

    mode = args.mode
    lr = args.lr
    weight_decay = args.weight_decay
    cross_val = args.cross_val
    n_fold = args.n_fold
    encoder = args.encoder
    pretrained_path = args.pretrained_path
    sigma = args.sigma
    temperature = args.temperature
    label_name = args.label_name
    num_classes = args.num_classes
    kernel = args.kernel
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    pretrained = args.pretrained
    rep_dim = args.rep_dim
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    n_layer = args.n_layer


    config = Config(mode=mode,
                    lr=lr,
                    weight_decay=weight_decay,
                    cross_val=cross_val,
                    n_fold=n_fold,
                    encoder=encoder,
                    n_layer=n_layer,
                    dir=dir,
                    pretrained_path=pretrained_path,
                    sigma=sigma,
                    temperature=temperature,
                    label_name=label_name,
                    num_classes=num_classes,
                    kernel=kernel,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    rep_dim=rep_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    pretrained=pretrained)

    print('label name',config.label_name)


    if config.cross_val:

        df = pd.read_csv(join(config.path_to_data, '1_subjects_label.csv'), delimiter=",").drop(['Unnamed: 0'], axis=1)

        skf = StratifiedKFold(n_splits=config.n_fold)

        for j, (train_index, val_index) in enumerate(skf.split(df['subject'],df['class'])):#enumerate(skf.split(df['subject'],df['label'])):

            train_index_ = df.loc[train_index, 'subject']
            train_labs = df.loc[train_index, config.label_name]
            val_index_ = df.loc[val_index, 'subject']
            val_labs = df.loc[val_index, config.label_name]

            train = [1 if x in list(train_index_) else 0 for x in df['subject']]
            val = [1 if x in list(val_index_) else 0 for x in df['subject']]

            df['train_set'] = train
            df['val_set'] = val

            ## re enregistrer le dataset en cross val
            df.to_csv(join(config.lght_dir, '1_subjects_label_cv.csv'))
            df.to_csv(join(config.lght_dir, '1_subjects_label_cv_'+str(j)+'.csv'))

            dataset_train = DatasetCL(config, training=True)
            dataset_val = DatasetCL(config, validation=True)

            #sampler = WeightedRandomSampler(dataset_train.samples_weight.type('torch.DoubleTensor'),
            #                                len(dataset_train))

            indices = {}  # indices of all the slices for each volume (patient) in the dataset
            for z in dataset_train.volumes:
                indices[z] = [i for i, x in enumerate(dataset_train.volumes) if x == z]

            k = len(dataset_train)
            sampler = CustomSampler(dataset_train, config.batch_size, indices,
                                    weights=dataset_train.samples_weight.numpy(),
                                    k=k)
            print('Weights assigned to each class', dataset_train.weight)

            loader_train = DataLoader(dataset_train, batch_size=config.batch_size,
                                      sampler=sampler,
                                      collate_fn=dataset_train.collate_fn,
                                      pin_memory=config.pin_mem,
                                      num_workers=config.nb_cpu,
                                      drop_last=False)

            loader_val = DataLoader(dataset_val, batch_size=config.batch_size,
                                    sampler=SequentialSampler(dataset_val),
                                    collate_fn=dataset_val.collate_fn,
                                    pin_memory=config.pin_mem,
                                    num_workers=config.nb_cpu,
                                    drop_last=False)

            print('Ready to download the model!')
            print('Pretrained on ImageNet?', config.pretrained)
            net = model_.network(mode="classifier",
                                 net=config.encoder,
                                 pretrained=config.pretrained,
                                 n_layer=config.n_layer,
                                 num_classes=config.num_classes,
                                 rep_dim=config.rep_dim,
                                 hidden_dim=config.hidden_dim)
            print('Network downloaded!')

            loss = CrossEntropyLoss()
            lr_logger = LearningRateMonitor(logging_interval='epoch')

            # Folder hack
            tb_logger = TensorBoardLogger(save_dir=config.lght_dir,
                                          version=f'fold_{j + 1}')
            checkpoint_callback = ModelCheckpoint(monitor="val_auc",
                                                  save_top_k=1,
                                                  every_n_epochs=1,
                                                  save_last=True,
                                                  mode="max",
                                                  dirpath=tb_logger.log_dir,
                                                  filename='best')

            trainer = pl.Trainer(gpus=([0]), default_root_dir=config.lght_dir, max_epochs=config.max_epochs,
                                 callbacks=[checkpoint_callback, lr_logger],
                                 val_check_interval=config.val_rate,
                                 amp_backend="native", precision=16,
                                 reload_dataloaders_every_epoch=False,
                                 num_sanity_val_steps=-1,
                                 logger=tb_logger,
                                 )

            model = wspContrastiveModel(net, loss, config, dataset_train, dataset_val, config.mode)

            # we check the number of trainable parameters
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            model_params = sum([np.prod(p.size()) for p in model_parameters])
            print('Number of trainable parameters in the whole model: ' + str(model_params))
            config.cv_fold = j
            trainer.fit(model, loader_train, loader_val)

    else:

        dataset_train = DatasetCL(config, training=True)
        dataset_val = DatasetCL(config, validation=True)
        print('ok')

        indices = {}  # indices of each volume (patient) in the dataset
        for z in dataset_train.volumes:
            indices[z] = [i for i, x in enumerate(dataset_train.volumes) if x == z]

        sampler = CustomSampler(dataset_train, config.batch_size, indices,
                                weights=dataset_train.samples_weight.numpy(),
                                k=10000)
        print('Weights assigned to each class', dataset_train.weight)

        loader_train = DataLoader(dataset_train, batch_size=config.batch_size,
                                  sampler=sampler,
                                  collate_fn=dataset_train.collate_fn,
                                  pin_memory=config.pin_mem,
                                  num_workers=config.nb_cpu,
                                  drop_last=False)

        loader_val = DataLoader(dataset_val, batch_size=config.batch_size,
                                sampler=SequentialSampler(dataset_val),
                                collate_fn=dataset_val.collate_fn,
                                pin_memory=config.pin_mem,
                                num_workers=config.nb_cpu,
                                drop_last=False)

        ### DENSENET AND LOSS

        if config.mode == "pretraining":

            net = model_.network(mode="encoder",
                                 net=config.encoder,
                                 pretrained=config.pretrained,
                                 n_layer=config.n_layer,
                                 rep_dim=config.rep_dim,
                                 hidden_dim=config.hidden_dim)

            loss = WSPContrastiveLoss(config=config,
                                      temperature=config.temperature,
                                      kernel=config.kernel,
                                      sigma=config.sigma,
                                      return_logits=True)

        elif config.mode == "finetuning":

            net = model_.network(mode="classifier",
                                 net=config.encoder,
                                 pretrained=config.pretrained,
                                 n_layer=config.n_layer,
                                 num_classes=config.num_classes,
                                 rep_dim=config.rep_dim,
                                 hidden_dim=config.hidden_dim,
                                 output_dim=config.output_dim)

            loss = CrossEntropyLoss()

        lr_logger = LearningRateMonitor(logging_interval='epoch')
        # Folder hack
        tb_logger = TensorBoardLogger(save_dir=config.lght_dir)
        checkpoint_callback = ModelCheckpoint(monitor="val_auc",
                                              save_top_k=1,
                                              every_n_epochs=1,
                                              save_last=True,
                                              mode="max",
                                              dirpath=tb_logger.log_dir,
                                              filename='best')
        trainer = pl.Trainer(gpus=([0]), default_root_dir=config.lght_dir, max_epochs=config.max_epochs,
                             callbacks=[checkpoint_callback, lr_logger],
                             val_check_interval=config.val_rate,
                             reload_dataloaders_every_epoch=False,
                             num_sanity_val_steps=-1,
                             logger=tb_logger,
                             )
                # The monitor argument name corresponds to the scalar value that you log
                # when using the self.log method within the LightningModule hooks.

        model = wspContrastiveModel(net, loss, config, dataset_train, dataset_val, config.mode)

        # we check the number of trainable parameters
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        model_params = sum([np.prod(p.size()) for p in model_parameters])
        print('Number of trainable parameters in the whole model: ' + str(model_params))

        trainer.fit(model, loader_train, loader_val)

