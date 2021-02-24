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


class Loss(nn.Module):

    def __init__(self, loss, weight=None, reduction='sum'):
        self.reduction = reduction

        super().__init__()

        if loss == 'bce':
            self.loss = nn.BCELoss(weight=weight, reduction=reduction)
        elif loss == 'bce_with_logit':
            self.loss = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)
        elif loss == 'mse':
            self.loss = nn.MSELoss(reduction=reduction)
        elif loss == 'l1':
            self.loss = nn.L1Loss(reduction=reduction)
        elif loss == 'ce':
            self.loss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        elif loss == 'nll':
            self.loss = nn.NLLLoss(weight=weight, reduction=reduction)
        else:
            self.loss = None

    def is_classification_task(self):
        if isinstance(self.loss, nn.NLLLoss) or isinstance(self.loss, nn.CrossEntropyLoss):
            return True
        return False

    def forward(self, y_hat, y):
        if self.loss is not None:
            if self.is_classification_task():
                y = y.long()

            return self.loss(y_hat, y)
        return y_hat.mean()