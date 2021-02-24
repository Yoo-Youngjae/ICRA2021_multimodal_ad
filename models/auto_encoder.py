#
#  MAKINAROCKS CONFIDENTIAL
#  ________________________
#
#  [2017] - [2020] MakinaRocks Co., Ltd.
#  All Rights Reserved.
#
#  NOTICE:  All information contained herein is, and remains
#  the property of MakinaRocks Co., Ltd. and its suppliers, if any.
#  The intellectual and technical concepts contained herein are
#  proprietary to MakinaRocks Co., Ltd. and its suppliers and may be
#  covered by U.S. and Foreign Patents, patents in process, and
#  are protected by trade secret or copyright law. Dissemination
#  of this information or reproduction of this material is
#  strictly forbidden unless prior written permission is obtained
#  from MakinaRocks Co., Ltd.
import torch
from models.abstract_model import AbstractModel


class AutoEncoder(AbstractModel):

    def __init__(
            self,
            encoder,
            decoder,
            recon_loss,
    ):
        super(AutoEncoder, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
        self.recon_loss = recon_loss

    def encode(self, x):
        # |x| = (batch_size, #c, w, h)
        z = self.encoder(x)
        return z.view(x.size(0), -1)

    def decode(self, z):
        # |z| = (batch_size, btl_size)
        y = self.decoder(z)
        return y

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        x_hat = x_hat
        return x_hat.view(x.size(0), -1)

    def get_loss_value(self, x, y, *args, **kwargs):
        output = self(x)
        loss = self.recon_loss(output, x)
        return loss

    @staticmethod
    def step(engine, mini_batch):
        # set models to train
        engine.model.train()
        engine.optimizer.zero_grad()

        # get hsr_objectdrop from mini_batch
        x, _ = mini_batch
        if engine.config.gpu_id >= 0:
            x = x.cuda(engine.config.gpu_id)
        x = x.view(x.size(0), -1)

        # get loss value from models
        loss = engine.model.get_loss_value(x, x)

        # update parameters with loss
        loss.backward(retain_graph=True)

        engine.optimizer.step()

        return (float(loss), )

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x, _ = mini_batch
            if engine.config.gpu_id >= 0:
                x = x.cuda(engine.config.gpu_id)
            x = x.view(x.size(0), -1)

            loss = engine.model.get_loss_value(x, x)

        return (float(loss), )

    @staticmethod
    def attach(trainer, evaluator, config):
        from ignite.engine import Events
        from ignite.metrics import RunningAverage
        from ignite.contrib.handlers.tqdm_logger import ProgressBar

        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'recon')

        if config.verbose >= 2:
            pbar = ProgressBar()
            pbar.attach(trainer, ['recon'])

        if config.verbose >= 1:
            @trainer.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_recon = engine.state.metrics['recon']
                print("Epoch {} - loss={:.4e}"
                      .format(engine.state.epoch, avg_recon))

        RunningAverage(output_transform=lambda x: x[0]).attach(evaluator, 'recon')

        if config.verbose >= 2:
            pbar = ProgressBar()
            pbar.attach(evaluator, ['recon'])

        if config.verbose >= 1:
            @evaluator.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_recon = engine.state.metrics['recon']
                print("Validation - recon={:.4e} lowest_recon={:.4e}"
                      .format(avg_recon, engine.lowest_loss))