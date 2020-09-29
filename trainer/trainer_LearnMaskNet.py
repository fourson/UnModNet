import numpy as np
import torch
from torchvision.utils import make_grid

from base.base_trainer import BaseTrainer
from utils import util


class DefaultTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, config, model, loss, metrics, optimizer, lr_scheduler, resume, data_loader,
                 valid_data_loader=None, train_logger=None):
        super(DefaultTrainer, self).__init__(config, model, loss, metrics, optimizer, lr_scheduler, resume,
                                             train_logger)

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, mask_pred, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(mask_pred, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        # set the model to train mode
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        # start training
        for batch_idx, sample in enumerate(self.data_loader):
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)

            # get data and send them to GPU
            # (N, C, H, W) GPU tensor
            modulo = sample['modulo'].to(self.device)  # [0, 1] float, as float32
            fold_number_edge = sample['fold_number_edge'].to(self.device)  # binary, as float32
            mask = sample['mask'].to(self.device)  # binary, as float32

            # get network output
            # (N, C, H, W) GPU tensor
            output = self.model(modulo, fold_number_edge)

            # visualization
            with torch.no_grad():
                # (N, C, H, W) GPU tensor
                mask_pred = torch.round(torch.sigmoid(output))

                if batch_idx % 100 == 0:
                    modulo_tonemapped = util.tonemap(modulo.cpu())

                    # save images to tensorboardX
                    self.writer.add_image('1_modulo', make_grid(modulo_tonemapped))
                    self.writer.add_image('2_fold_number_edge', make_grid(fold_number_edge))
                    self.writer.add_image('3_mask_pred', make_grid(mask_pred))
                    self.writer.add_image('4_mask', make_grid(mask))

            # train model
            self.optimizer.zero_grad()
            model_loss = self.loss(output, mask)
            model_loss.backward()
            self.optimizer.step()

            # calculate total loss/metrics and add scalar to tensorboard
            self.writer.add_scalar('loss', model_loss.item())
            total_loss += model_loss.item()
            total_metrics += self._eval_metrics(mask_pred, mask)

            # show current training step info
            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] loss: {:.6f}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        model_loss.item(),  # it's a tensor, so we call .item() method
                    )
                )

        # turn the learning rate
        self.lr_scheduler.step()

        # get batch average loss/metrics as log and do validation
        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        # set the model to validation mode
        self.model.eval()

        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))

        # start validating
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.valid_data_loader):
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')

                # get data and send them to GPU
                # (N, C, H, W) GPU tensor
                modulo = sample['modulo'].to(self.device)  # [0, 1] float, as float32
                fold_number_edge = sample['fold_number_edge'].to(self.device)  # binary, as float32
                mask = sample['mask'].to(self.device)  # binary, as float32

                # infer and calculate the loss
                # (N, C, H, W) GPU tensor
                output = self.model(modulo, fold_number_edge)
                mask_pred = torch.round(torch.sigmoid(output))
                loss = self.loss(output, mask)

                # calculate total loss/metrics and add scalar to tensorboardX
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(mask_pred, mask)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
