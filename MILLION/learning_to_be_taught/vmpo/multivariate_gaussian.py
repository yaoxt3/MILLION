import numpy as np
import torch
import math
from learning_to_be_taught.vmpo.utils import batched_quadratic_form

from rlpyt.distributions.base import Distribution
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import valid_mean

EPS = 1e-20

DistInfo = namedarraytuple("DistInfo", ["mean", 'covariance'])
# DistInfoStd = namedarraytuple("DistInfoStd", ["mean", "log_std"])


class MultivariateGaussian(Distribution):
    """Multivariate Gaussian with full covariance matrix
    Noise clipping or sample clipping optional during sampling, but not
    accounted for in formulas (e.g. entropy).
    Clipping of standard deviation optional and accounted in formulas.
    Squashing of samples to squash * tanh(sample) is optional and accounted for
    in log_likelihood formula but not entropy.
    """

    def __init__(
            self,
            dim,
            std=None,
            clip=None,
            noise_clip=None,
            min_std=None,
            max_std=None,
            squash=None,  # None or > 0
    ):
        """Saves input arguments."""
        self._dim = dim
        self.set_std(std)
        self.clip = clip
        self.noise_clip = noise_clip
        self.min_std = min_std
        self.max_std = max_std
        self.min_log_std = math.log(min_std) if min_std is not None else None
        self.max_log_std = math.log(max_std) if max_std is not None else None
        self.squash = squash
        assert (clip is None or squash is None), "Choose one."

    @property
    def dim(self):
        return self._dim

    def kl(self, old_dist_info, new_dist_info):
        return self.mu_kl(old_dist_info, new_dist_info) + self.sigma_kl(old_dist_info, new_dist_info)

    def mean_kl(self, old_dist_info, new_dist_info):
        old_mean = old_dist_info.mean
        new_mean = new_dist_info.mean
        mu_kl = 0.5 * batched_quadratic_form(new_mean - old_mean, old_dist_info.covariance.inverse())
        return mu_kl

    def covariance_kl(self, old_dist_info, new_dist_info):
        d = np.prod(old_dist_info.mean.shape[-1])
        old_sigma = old_dist_info.covariance
        new_sigma = new_dist_info.covariance

        trace = torch.sum(torch.diagonal(torch.bmm(new_sigma.inverse(), old_sigma), dim1=-2, dim2=-1), dim=-1)
        sigma_kl = 0.5 * (trace - d + torch.log(new_sigma.det().clamp_min(EPS) / old_sigma.det().clamp_min(EPS)))
        return sigma_kl

    def entropy(self, dist_info):
        """Uses ``self.std`` unless that is None, then will get log_std from dist_info.  Not
        implemented for squashing.
        """
        d = np.prod(dist_info.mean.shape[-1])
        return d / 2 + d / 2 * math.log(2 * math.pi) + 0.5 * dist_info.covariance.det().log()

    def perplexity(self, dist_info):
        return torch.exp(self.entropy(dist_info))

    def mean_entropy(self, dist_info, valid=None):
        return valid_mean(self.entropy(dist_info), valid)

    def mean_perplexity(self, dist_info, valid=None):
        return valid_mean(self.perplexity(dist_info), valid)

    def log_likelihood(self, x, dist_info):
        """
        Uses ``self.std`` unless that is None, then uses log_std from dist_info.
        When squashing: instead of numerically risky arctanh, assume param
        'x' is pre-squash action, see ``sample_loglikelihood()`` below.
        """
        d = torch.tensor(dist_info.mean.shape[-1], dtype=torch.float32)
        logli = -0.5 * batched_quadratic_form(x - dist_info.mean, dist_info.covariance) \
                - (d/2) * math.log(2 * math.pi) - 0.5 * torch.log(dist_info.covariance.det().clamp_min(EPS)) #JK + 1e-5)
        # print('logli ax:' + str(logli.max()) + str(dist_info.covariance.det()))
        return logli

    def likelihood_ratio(self, x, old_dist_info, new_dist_info):
        logli_old = self.log_likelihood(x, old_dist_info)
        logli_new = self.log_likelihood(x, new_dist_info)
        return torch.exp(logli_new - logli_old)

    def sample_loglikelihood(self, dist_info):
        """
        Special method for use with SAC algorithm, which returns a new sampled
        action and its log-likelihood for training use.  Temporarily turns OFF
        squashing, so that log_likelihood can be computed on non-squashed sample,
        and then restores squashing and applies it to the sample before output.
        """
        squash = self.squash
        self.squash = None  # Temporarily turn OFF, raw sample into log_likelihood.
        sample = self.sample(dist_info)
        self.squash = squash  # Turn it back ON, squash correction in log_likelihood.
        logli = self.log_likelihood(sample, dist_info)
        if squash is not None:
            sample = squash * torch.tanh(sample)
        return sample, logli

    def sample(self, dist_info):
        """
        Generate random samples using ``torch.normal``, from
        ``dist_info.mean``. Uses ``self.std`` unless it is ``None``, then uses
        ``dist_info.log_std``.
        """
        mean_shape = dist_info.mean.shape
        mean = dist_info.mean.view(-1, mean_shape[-1])
        covariance = dist_info.covariance.view(-1, mean.shape[-1], mean.shape[-1])
        sample = mean + torch.bmm(covariance, torch.normal(torch.zeros_like(mean).unsqueeze(-1), torch.ones_like(mean).unsqueeze(-1))).squeeze(-1)
        return sample.reshape(mean_shape)

    def set_std(self, std):
        """
        Input value, which can be same shape as action space, or else broadcastable
        up to that shape, or ``None`` to turn OFF and use ``dist_info.log_std`` in
        other methods.
        """
        if std is not None:
            if not isinstance(std, torch.Tensor):
                std = torch.tensor(std).float()  # Can be size == 1 or dim.
            # Used to have, but shape of std should broadcast everywhere needed:
            # if std.numel() == 1:
            #     std = std * torch.ones(self.dim).float()  # Make it size dim.
            assert std.numel() in (self.dim, 1)
        self.std = std
