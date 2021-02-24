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


class AbstractModel(nn.Module):

    def __init__(self, *model_and_opts):
        super().__init__()

        from collections.abc import Iterable
        for model_and_opt in model_and_opts:
            if not (model_and_opt is None or isinstance(model_and_opt, Iterable)):
                raise Exception("model_and_opt arg should be None or iterable objects")

        # For saving best model's optimizers
        self.optimizer_list = []

    def forward(self):
        raise NotImplementedError

    def get_loss_value(self, x, y, *args, **kwargs):
        raise NotImplementedError

    def get_all_optimizers_state_dicts(self):
        return [opt.state_dict() for opt in self.optimizer_list]
