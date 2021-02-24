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
class Reporter():

    def __init__(self):
        self.config_d = {}
        self.result_d = {}
        self.cnt = 0

    def add(self, config, result):
        assert type(config) == dict and type(result) == dict

        assert len(self.config_d) == 0 or len(self.config_d) == len(config)
        assert len(self.result_d) == 0 or len(self.result_d) == len(result)

        for k, v in config.items():
            if self.config_d.get(k) is not None:
                self.config_d[k] = self.config_d[k] + [v]
            else:
                self.config_d[k] = [v]

        for k, v in result.items():
            if self.result_d.get(k) is not None:
                self.result_d[k] = self.result_d[k] + [v]
            else:
                self.result_d[k] = [v]

        self.cnt += 1

    def export(self, fn, delimiter=','):
        head = delimiter.join(list(self.config_d.keys()) + list(self.result_d.keys()))
        rows = []

        for row_index in range(self.cnt):
            row = []
            for k in self.config_d.keys():
                row += [self.config_d[k][row_index]]
            for k in self.result_d.keys():
                row += [self.result_d[k][row_index]]

            rows += [delimiter.join(list(map(str, row)))]

        total = '\n'.join([head] + rows)
        
        with open(fn, 'w') as f:
            f.write(total)
