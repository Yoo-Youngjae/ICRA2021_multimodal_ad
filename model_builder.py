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
import torch.nn as nn

from modules import FCModule, Loss
from utils.common_utils import get_hidden_layer_sizes


def ae_wrapper(config):
    from models.auto_encoder import AutoEncoder

    # args
    input_size = config.input_size
    btl_size = config.btl_size
    n_layers = config.n_layers

    # input_size -> flatten
    if type(input_size) != int:
        C, H, W = input_size
        input_size = C * H * W
    else:
        input_size = input_size

    encoder = FCModule(
        input_size=input_size,
        output_size=btl_size,
        hidden_sizes=get_hidden_layer_sizes(input_size, btl_size, n_hidden_layers=n_layers - 1),
        use_batch_norm=True,
        act="leakyrelu",
        last_act=None,
    )

    decoder = FCModule(
        input_size=btl_size,
        output_size=input_size,
        hidden_sizes=get_hidden_layer_sizes(btl_size, input_size, n_hidden_layers=n_layers - 1),
        use_batch_norm=True,
        act="leakyrelu",
        last_act=None,
    )

    model = AutoEncoder(
        encoder=encoder,
        decoder=decoder,
        recon_loss=Loss("mse", reduction="sum"),
    )

    return model


def get_model(config):
    if config.model == 'ae':
        model = ae_wrapper(config)
    elif config.model == 'vae':
        from models.variational_auto_encoder import VariationalAutoEncoder as VAE
        if type(config.input_size) != int:
            input_size = 1
            for i in config.input_size:
                input_size *= i
        else:
            input_size = config.input_size
        model = VAE(
            input_size=input_size,
            btl_size=config.btl_size,
            n_layers=config.n_layers,
            k=10
        )
    elif config.model == 'aae':
        from models.adversarial_auto_encoder import AdversarialAutoEncoder as AAE
        if 'dropout' in vars(config).keys():
            dropout = config.dropout
        else:
            dropout = 0.2
        if type(config.input_size) != int:
            input_size = 1
            for i in config.input_size:
                input_size *= i
        else:
            input_size = config.input_size
        model = AAE(
                    input_size=input_size,
                    btl_size=config.btl_size,
                    n_layers=config.n_layers,
                )
    else:
        raise NotImplementedError

    if config.gpu_id >= 0:
        model = model.cuda(config.gpu_id)

    return model