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
import numpy as np
import torch


def get_diffs(x, model, batch_size=698):
    model.eval()
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    # x.shape == [3846, 1728]

    batchified = x.split(batch_size)
    # batchified = [6, batch_size, 1728)
    # batchified = x.view(-1, 1, 1) # when hand weight test

    stacked = []
    for _x in batchified:
        # _x.shape == batch_size, 1728
        model.eval()
        diffs = []
        _x = _x.to(next(model.parameters()).device).float()
        x_tilde = model(_x)
        diffs.append((x_tilde - _x).cpu())

        for layer in model.encoder.layer_list:
            _x = layer(_x)
            x_tilde = layer(x_tilde)
            diffs.append((x_tilde - _x).cpu())

        # diffs.shape == 6, batch_size, 1728
        stacked.append(diffs)
    # stacked.shape == 6, 6, batch_size, 1728
    stacked = list(zip(*stacked))
    # stacked.shape == 3, 6, batch_size, 1728
    diffs = [torch.cat(s, dim=0).numpy() for s in stacked]

    return diffs
