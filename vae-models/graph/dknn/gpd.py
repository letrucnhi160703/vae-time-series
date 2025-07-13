import torch
from torch.distributions import Distribution, constraints

class GeneralizedPareto(Distribution):
    arg_constraints = {}
    # support = constraints.positive
    has_rsample = True

    def __init__(self, concentration, scale, loc, validate_args=None):
        self.concentration = concentration
        self.scale = scale
        # loc = torch.full_like(scale, loc)
        self.loc = loc

        batch_shape = torch.broadcast_shapes(
            concentration.shape, scale.shape, loc.shape
        )
        super().__init__(batch_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        u = torch.rand(sample_shape + self.batch_shape, device=self.scale.device)
        return self._inverse_cdf(u)

    def rsample(self, sample_shape=torch.Size()):
        u = torch.rand(sample_shape + self.batch_shape, device=self.scale.device)
        u = u.clamp(min=1e-6, max=1 - 1e-6)  # tránh log(0)
        return self._inverse_cdf(u)

    def _inverse_cdf(self, u):
        # Inverse CDF method
        xi = self.concentration
        sigma = self.scale
        mu = self.loc

        if torch.any(xi == 0):
            # GPD giảm về exponential
            return mu - sigma * torch.log(1 - u)
        else:
            return mu + sigma / xi * ((1 - u) ** (-xi) - 1)

    def log_prob(self, value):
        xi = self.concentration
        sigma = self.scale
        mu = self.loc
        z = (value - mu) / sigma
        inside = 1 + xi * z

        log_pdf = -torch.log(sigma) - (1/xi + 1) * torch.log(inside)
        log_pdf = torch.where(
            inside > 0,
            log_pdf,
            torch.full_like(log_pdf, float('-inf'))
        )
        return log_pdf