
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

def variational_info_bottleneck(forward_fn):
    def _reparameterize_normal(mu, sigma, k, stochastic_inference):
        import torch
        if torch.is_grad_enabled() or stochastic_inference:
            expanded_sigma = sigma.unsqueeze(0).expand(k, *sigma.size())
            z = torch.randn_like(expanded_sigma).mul(expanded_sigma) + mu
        else:
            return mu.unsqueeze(0).expand(k, *mu.size())
        return z

    def decorated_forward(self, x, distribution=None, k=1, stochastic_inference=True):
        output = forward_fn(self, x)
        if distribution is None:
            return output
        elif distribution == "normal":
            mu, logvar = output.split(output.size(-1) // 2, dim=-1)
            if k < 1:
                raise ValueError("k should be >= 1")
            z = _reparameterize_normal(mu, (logvar * .5).exp(), k, stochastic_inference)
            return {'z': z, 'mu': mu, 'logvar': logvar}
        else:
            raise NotImplementedError("Wrong distribution for information bottleneck: {}".format(distribution))

    return decorated_forward
