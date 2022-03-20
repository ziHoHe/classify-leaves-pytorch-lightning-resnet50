import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
import albumentations
from albumentations import pytorch as AP
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import *
import torchmetrics
from sklearn.preprocessing import OneHotEncoder
import timm
import re


def get_latest_ckpt(lightning_logs_root: str):
    """
    :param lightning_logs_root: just like './resume' or './train'
    :return: path
    """
    lightning_logs_root += '/lightning_logs'
    patten = re.compile(r'\d+.\d+')
    version_list = os.listdir(lightning_logs_root)
    version_list = sorted(version_list, key=lambda x: int(x[-1]), reverse=True)
    ckpt_list = os.listdir(os.path.join(lightning_logs_root, version_list[0], 'checkpoints'))
    result_dict = {}
    for ckpt in ckpt_list:
        result_dict[float(patten.findall(ckpt)[0])] = ckpt
    result_list = sorted(result_dict, reverse=True)
    return os.path.join(lightning_logs_root, version_list[0], 'checkpoints',
                        result_dict[result_list[0]])


RANDOM_SEED = 107
pl.seed_everything(RANDOM_SEED)

RESUME_TRAIN = True
MODEL_NAME = 'resnet50'
IMAGE_ROOT_PATH = 'E:\\dataset'
DATASET_NAME = 'classify-leaves'
CKPT_RESUME_PATH = './resume'
RESUME_PATH = get_latest_ckpt('./resume')
if RESUME_TRAIN:
    CKPT_SAVE_PATH = CKPT_RESUME_PATH
else:
    CKPT_SAVE_PATH = './train'

MAX_EPOCHS = 30
LABEL_SMOOTH = 0.05
LR = 3e-4
BATCH_SIZE = 64


class LeaDataSet(Dataset):
    def __init__(self, img_nemes=None, labels: torch.Tensor = None, transform: Callable = None, train: bool = True):
        super(LeaDataSet, self).__init__()
        self.img_names = img_nemes
        if train:
            self.labels = labels
        else:
            self.labels = None
        self.transform = transform

    def __getitem__(self, item):
        img = cv2.imread(os.path.join(IMAGE_ROOT_PATH, DATASET_NAME, self.img_names[item]), cv2.IMREAD_COLOR)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']

        if self.labels is not None:
            return img, self.labels[item]
        else:
            return img, torch.zeros((len(self.img_names),), dtype=torch.float32)

    def __len__(self):
        return len(self.img_names)


class LeaDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_img_names=None, train_labels=None,
                 val_img_names=None, val_labels=None,
                 batch_size=32,
                 test_img_names=None):
        super(LeaDataModule, self).__init__()
        self.training_transforms = albumentations.Compose([
            albumentations.RandomRotate90(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.RandomBrightnessContrast(p=0.5),
            albumentations.RandomGamma(p=0.5),
            albumentations.OneOf([
                albumentations.GaussNoise(),
                albumentations.GlassBlur(),
            ]),
            albumentations.Normalize(),
            AP.ToTensorV2(),
        ])

        self.valing_transforms = albumentations.Compose([
            albumentations.Normalize(),
            AP.ToTensorV2(),
        ])

        self.train_img_names = train_img_names
        self.train_labels = train_labels
        self.test_img_names = test_img_names
        self.val_img_names = val_img_names
        self.val_labels = val_labels

        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = LeaDataSet(self.train_img_names, self.train_labels, self.training_transforms)
        self.val_dataset = LeaDataSet(self.val_img_names, self.val_labels, self.valing_transforms)
        self.predict_dataset = LeaDataSet(self.test_img_names, None, transform=self.valing_transforms, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False)


class LeaNetModule(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 pretrained: bool,
                 num_classes: int,
                 learning_rate: float = 3e-4,
                 weight_decay: float = 1e-4,
                 t_max: float = 20,
                 label_smooth: float = 0.05
                 ):
        super(LeaNetModule, self).__init__()
        self.arch = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.save_hyperparameters()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, x) -> torch.Tensor:
        out = self.arch(x)
        return out

    def training_step(self, batch, batch_idx):
        return self._setup('train', batch)

    def validation_step(self, batch, batch_idx):
        self._setup('val', batch)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        x, y = batch
        y_hat = self(x)
        return F.softmax(y_hat, dim=1)

    def _setup(self, mode, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, label_smoothing=LABEL_SMOOTH)
        y = y.type(torch.int64)
        if mode == 'train':
            self.log('train_loss', loss, prog_bar=True, on_step=True)
            self.train_acc(y_hat.argmax(1), y.argmax(1))
            self.log('train_acc', self.train_acc, prog_bar=True, on_step=True)
            return loss

        if mode == 'val':
            self.log('val_loss', loss, prog_bar=True, on_epoch=True)
            self.val_acc(y_hat.argmax(1), y.argmax(1))
            self.log('val_acc', self.val_acc, prog_bar=True, on_epoch=True, on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,
                                     weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.t_max)
        return [optimizer], [lr_scheduler]


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(IMAGE_ROOT_PATH, DATASET_NAME, 'train.csv'))
    print(df.info())
    print(df.describe())

    label_sort_list = sorted(set(list(df.label)))
    n_class = len(label_sort_list)
    label_map_dict = dict(zip(label_sort_list, range(n_class)))
    label_inv_map_dict = {v: k for k, v in label_map_dict.items()}
    print(label_map_dict)
    print(label_inv_map_dict)

    label_map = df['label'].apply(lambda x: label_map_dict[x]).to_numpy()
    img_names = df['image'].to_numpy()
    print(label_map)
    print(img_names)

    onthot_encoder = OneHotEncoder()
    label_reshape = label_map.reshape(len(label_map), -1)
    label_array = onthot_encoder.fit_transform(label_reshape).toarray()
    print(np.argmax(label_array, axis=-1))

    # use sklearn

    train_data, val_data, train_labels, val_labels = train_test_split(img_names, label_array, test_size=0.2,
                                                                      random_state=RANDOM_SEED)

    train_labels, val_labels = torch.tensor(train_labels, dtype=torch.float32), torch.tensor(val_labels,
                                                                                             dtype=torch.float32)

    assert len(train_data) == len(train_labels)

    dm = LeaDataModule(train_data, train_labels, val_data, val_labels, batch_size=BATCH_SIZE)
    module = LeaNetModule(MODEL_NAME, True, n_class, learning_rate=LR)
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=1, mode='max', filename='sample_{val_acc:.6f}')

    if not os.path.exists(CKPT_SAVE_PATH):
        os.mkdir(CKPT_SAVE_PATH)

    trainer = pl.Trainer(default_root_dir=CKPT_SAVE_PATH,
                         callbacks=[checkpoint_callback],
                         gpus=torch.cuda.device_count(),
                         num_sanity_val_steps=1,
                         precision=16,
                         max_epochs=MAX_EPOCHS,
                         )

    if not RESUME_TRAIN:
        trainer.fit(model=module, datamodule=dm)
    else:
        trainer.fit(model=module, datamodule=dm, ckpt_path=RESUME_PATH)
