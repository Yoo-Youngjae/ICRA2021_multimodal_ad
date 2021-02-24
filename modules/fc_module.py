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
import torch.nn as nn

from decorators import variational_info_bottleneck as vib
from layers import FCLayer


class FCModule(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes=None,
                 use_batch_norm=True,
                 dropout_p=0,
                 act='leakyrelu',
                 last_act=None
                 ):

        super().__init__()

        self.layer_list = []

        if use_batch_norm and dropout_p > 0:
            raise Exception("Either batch_norm or dropout is allowed, not both")

        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for idx, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            if idx < len(hidden_sizes):
                layer = FCLayer(input_size=in_size,
                                output_size=out_size,
                                act=act,
                                bn=use_batch_norm,
                                dropout_p=dropout_p
                                )
            else:
                layer = FCLayer(input_size=in_size,
                                output_size=out_size,
                                act=last_act,
                                )

            self.layer_list.append(layer)
        self.net = nn.Sequential(*self.layer_list)

    @vib
    def forward(self, x):
        return self.net(x)
