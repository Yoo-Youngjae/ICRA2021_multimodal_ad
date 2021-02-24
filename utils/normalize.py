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

class Standardizer():

    def __init__(self, *args, **kwargs):
        self.mu, self.var = None, None

    def fit(self, x):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.float()
            # |x| = (batch_size, dim)

            self.mu = x.mean(dim=0)
            x = x - self.mu
            self.var = torch.from_numpy(np.cov(x.numpy().T)).diagonal().float()

    def run(self, x):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.float()
            # |x| = (batch_size, dim)

            x = (x - self.mu) / self.var**.5

        return x.numpy()

class Rotater():

    def __init__(self, *args, **kwargs):
        self.mu, self.v = None, None

    def fit(self, x, gpu_id=-1):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.float()
            # |x| = (batch_size, dim)

            self.mu = x.mean(dim=0)
            x = x - self.mu

            if gpu_id >= 0:
                device = torch.device('cpu') if gpu_id < 0 else torch.device('cuda:%d' % gpu_id)

                x = x.to(device)

            u, s, self.v = x.svd()

            if gpu_id >= 0:
                self.v = self.v.cpu()

    def run(self, x, gpu_id=-1, max_size=20000):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.float()
            # |x| = (batch_size, dim)

            x = x - self.mu

            if gpu_id >= 0:
                device = torch.device('cpu') if gpu_id < 0 else torch.device('cuda:%d' % gpu_id)

                x = x.to(device)
                v = self.v.to(device)
            else:
                v = self.v

            if len(x) > max_size:
                x_tilde = []
                for x_i in x.split(max_size, dim=0):
                    x_i_tilde = torch.matmul(x_i, v)

                    x_tilde += [x_i_tilde]

                x_tilde = torch.cat(x_tilde, dim=0)
            else:
                x_tilde = torch.matmul(x, v)

            if gpu_id >= 0:
                x_tilde = x_tilde.cpu()

            return x_tilde.numpy()

class Truncater(Rotater):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, x, trunc, gpu_id=-1, max_size=20000):
        if trunc <= 0:
            return x

        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.float()
            # |x| = (batch_size, dim)

            x = x - self.mu

            if gpu_id >= 0:
                device = torch.device('cpu') if gpu_id < 0 else torch.device('cuda:%d' % gpu_id)

                x = x.to(device)
                v = self.v.to(device)[:, :trunc]
            else:
                v = self.v[:, :trunc]
            v_t = v.transpose(0, 1)

            if len(x) > max_size:
                x_tilde = []
                for x_i in x.split(max_size, dim=0):
                    x_i_tilde = torch.matmul(torch.matmul(x_i, v), v_t)

                    x_tilde += [x_i_tilde]

                x_tilde = torch.cat(x_tilde, dim=0)
            else:
                x_tilde = torch.matmul(torch.matmul(x, v), v_t)

            if gpu_id >= 0:
                x_tilde = x_tilde.cpu()
            x_tilde = x_tilde + self.mu

            return x_tilde.numpy()
