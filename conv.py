'''
Model ResNet do przewidywania kolejnej nuty z sekwencji X nut
'''

import torch
import pytorch_lightning as pl
from torch import nn
import torchmetrics
import torch.nn.functional as F
from torch import optim
from pytorch_lightning.loggers import TensorBoardLogger

from common import ZeroLoss, generate_sample_song
from songs_data import SongsDataModule


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)


class SongResNet(pl.LightningModule):
    def __init__(self, in_channels, resblock, pitches_num=128):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )

        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fcPitch = torch.nn.Linear(512, pitches_num)
        self.fcStep = torch.nn.Linear(512, 1)
        self.fcDuration = torch.nn.Linear(512, 1)

        self.in_channels = in_channels

        self.pitch_loss_function = nn.CrossEntropyLoss()
        self.step_loss_function = nn.MSELoss()
        self.step_loss_function_2 = ZeroLoss()
        self.duration_loss_function = nn.MSELoss()
        self.duration_loss_function_2 = ZeroLoss(scale=True)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self.train_macro_f1 = torchmetrics.F1Score(num_classes=pitches_num, average='macro')
        self.val_macro_f1 = torchmetrics.F1Score(num_classes=pitches_num, average='macro')
        self.val_epoch_num = 0


    def forward(self, x):
        '''
        :param x: (N, seq_len, 3)
        :return:
        '''
        N, H, W = x.size()
        x = torch.reshape(x, (N, self.in_channels, H, W))

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)

        pitch = self.fcPitch(x)
        step = nn.LeakyReLU()(self.fcStep(x))
        duration = nn.LeakyReLU()(self.fcDuration(x))

        return pitch, step, duration

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        return optimizer

    def validation_epoch_end(self, _):
        generate_sample_song(self, filename=f"song_{self.val_epoch_num}.midi", seq_len=25)

        self.val_epoch_num += 1

    def training_step(self, train_batch, batch_idx):
        input_notes, label_note = train_batch

        label_pitch = label_note[:, 0].long()
        val_step = label_note[:, 1]
        val_duration = label_note[:, 2]

        pitch, step, duration = self.forward(input_notes.float())

        step = torch.flatten(step)
        duration = torch.flatten(duration)

        loss_pitch = self.pitch_loss_function(pitch, label_pitch)
        loss_step = self.step_loss_function(step , val_step)
        loss_step_2 = self.step_loss_function_2(step)
        loss_duration = self.duration_loss_function(duration, val_duration)
        loss_duration_2 = self.duration_loss_function_2(duration)

        self.log('pitch_train_loss', loss_pitch, on_step=True, on_epoch=True)
        self.log('step_train_loss', loss_step, on_step=True, on_epoch=True)
        self.log('duration_train_loss', loss_duration, on_step=True, on_epoch=True)
        self.log('duration_max_train', torch.max(duration), on_step=True, on_epoch=True)
        self.log('duration_min_train', torch.min(duration), on_step=True, on_epoch=True)
        self.log('step_max_train', torch.max(step), on_step=True, on_epoch=True)
        self.log('step_min_train', torch.min(step), on_step=True, on_epoch=True)
        self.log('step_zero_loss_train', loss_step_2, on_step=True, on_epoch=True)
        self.log('duration_zero_loss_train', loss_duration_2, on_step=True, on_epoch=True)

        loss = loss_pitch + loss_step + loss_duration + loss_step_2 + loss_duration_2

        self.log('train_loss', loss, on_step=True, on_epoch=True)

        pitch_outputs = F.softmax(pitch, dim=1)

        acc = self.train_acc(pitch_outputs, label_pitch)
        self.log('train_pitch_acc', acc, on_epoch=True, on_step=False)

        f1 = self.train_macro_f1(pitch_outputs, label_pitch)
        self.log('train_pitch_macro_f1', f1, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, val_batch, batch_idx):
        input_notes, label_note = val_batch

        label_pitch = label_note[:, 0].long()
        val_step = label_note[:, 1]
        val_duration = label_note[:, 2]

        pitch, step, duration = self.forward(input_notes.float())

        step = torch.flatten(step)
        duration = torch.flatten(duration)

        loss_pitch = self.pitch_loss_function(pitch, label_pitch)
        loss_step = self.step_loss_function(step , val_step)
        loss_step_2 = self.step_loss_function_2(step)
        loss_duration = self.duration_loss_function(duration, val_duration)
        loss_duration_2 = self.duration_loss_function_2(duration)

        self.log('pitch_train_loss', loss_pitch, on_step=True, on_epoch=True)
        self.log('step_train_loss', loss_step, on_step=True, on_epoch=True)
        self.log('duration_train_loss', loss_duration, on_step=True, on_epoch=True)
        self.log('duration_max_train', torch.max(duration), on_step=True, on_epoch=True)
        self.log('duration_min_train', torch.min(duration), on_step=True, on_epoch=True)
        self.log('step_max_train', torch.max(step), on_step=True, on_epoch=True)
        self.log('step_min_train', torch.min(step), on_step=True, on_epoch=True)
        self.log('step_zero_loss_train', loss_step_2, on_step=True, on_epoch=True)
        self.log('duration_zero_loss_train', loss_duration_2, on_step=True, on_epoch=True)

        loss = loss_pitch + loss_step + loss_duration + loss_step_2 + loss_duration_2

        self.log('val_loss', loss, on_step=True, on_epoch=True)

        pitch_outputs = F.softmax(pitch, dim=1)

        acc = self.val_acc(pitch_outputs, label_pitch)
        self.log('val_pitch_acc', acc, on_epoch=True, on_step=False)

        f1 = self.val_macro_f1(pitch_outputs, label_pitch)
        self.log('val_pitch_macro_f1', f1, on_epoch=True, on_step=False)

        return loss


if __name__ == '__main__':
    dm = SongsDataModule(num_train_songs=10, num_val_songs=5, split_label=True)
    model = SongResNet(1, ResBlock)
    logger = TensorBoardLogger("lightning_logs", name="resnet_model", version="LeakyRELU for step and duration fc, but abs them when generating")

    trainer = pl.Trainer(logger=logger, max_epochs = 10, log_every_n_steps=1)
    trainer.fit(model, dm)
