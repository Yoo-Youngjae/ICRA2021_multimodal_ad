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


class Activation(nn.Module):

    def __init__(self, act):
        super().__init__()

        if act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'logsigmoid':
            self.act = nn.LogSigmoid()
        elif act == 'softmax':
            self.act = nn.Softmax(dim=-1)
        elif act == 'logsoftmax':
            self.act = nn.LogSoftmax(dim=-1)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'leakyrelu':
            self.act = nn.LeakyReLU(.2)
        else:
            self.act = None

    def forward(self, x):
        if self.act is not None:
            return self.act(x)
        return x
