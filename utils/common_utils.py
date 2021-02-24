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
import random
import numpy as np


def get_hidden_layer_sizes(start_size, end_size, n_hidden_layers):
    """
    It can handle both increasing & decreasing sizes automatically
    """
    sizes = []
    diff = (start_size - end_size) / (n_hidden_layers + 1)

    for idx in range(n_hidden_layers):
        sizes.append(int(start_size - (diff * (idx + 1))))
    return sizes