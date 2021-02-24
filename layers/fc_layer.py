
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
from torch import nn

from modules import Activation


class FCLayer(nn.Module):
    def __init__(self,
                 input_size,
                 output_size=1,
                 bias=True,
                 act='relu',
                 bn=False,
                 dropout_p=0):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size, bias)
        self.bn = nn.BatchNorm1d(output_size) if bn else None
        self.dropout = nn.Dropout(dropout_p) if dropout_p else None
        self.act = Activation(act) if act else None

    def forward(self, x):
        y = self.act(self.layer(x)) if self.act else self.layer(x)
        if self.bn:
            # In case of expansion(k) in Information bottleneck
            if y.dim() > 2:
                original_y_size = y.size()
                y = self.bn(y.view(-1, y.size(-1))).view(*original_y_size)
            else:
                y = self.bn(y)
        y = self.dropout(y) if self.dropout else y

        return y
