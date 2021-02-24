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
import argparse
import itertools
import random

import multiprocessing
from multiprocessing import Pool


class Repeater():

    def __init__(self, config, f, interval=0):
        self.config = config
        self.f = f
        self.interval = interval

    def run(self):
        with Pool(processes=len(self.config.gpu_id)) as p:
            original_gpu_id = self.config.gpu_id
            self.config.gpu_id = [-1]

            params_list = itertools.product(*[v for _, v in vars(self.config).items()])
            titles = [k for k, _ in vars(self.config).items()]

            def build_config(titles, params):
                config = argparse.Namespace()
                d = vars(config)

                for t, p in zip(titles, params):
                    d[t] = p

                return config

            params_list = [build_config(titles, params) for params in list(params_list)]
            random.shuffle(params_list)
            print('We have %d processes.' % len(params_list))
            
            for i, c in enumerate(params_list):
                c.progress = i + 1
                c.gpu_id = original_gpu_id
                c.sleep = (self.interval * i) if i < len(original_gpu_id) else 0

            result = p.map(self.f, params_list)

            self.config.gpu_id = original_gpu_id

        return result

def convert(x, to=int, delimiter=','):
    if type(x) == str:
        if delimiter in x:
            return list(map(to, x.split(delimiter)))
        else:
            return [to(x)]
    else:
        return [x]

def get_config(settings):
    import argparse

    p = argparse.ArgumentParser()

    for s in settings:
        assert type(s[0]) == str

        if type(s[2]) == bool:
            p.add_argument('--' + s[0], action='store_true', default=s[2])
        else:
            p.add_argument('--' + s[0], type=str, default=s[2])

    config = p.parse_args()

    d = vars(config)    
    for s in settings:
        d[s[0]] = convert(d[s[0]], to=s[1])

    return config
