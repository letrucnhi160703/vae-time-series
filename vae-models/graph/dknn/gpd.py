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
        u = u.clamp(min=1e-4, max=1 - 1e-4)
        return self._inverse_cdf(u)

    def rsample(self, sample_shape=torch.Size()):
        u = torch.rand(sample_shape + self.batch_shape, device=self.scale.device)
        u = u.clamp(min=1e-4, max=1 - 1e-4)  # tránh log(0)
        return self._inverse_cdf(u)

    def _inverse_cdf(self, u):
        # Inverse CDF method
        xi = self.concentration
        sigma = self.scale
        mu = self.loc

        is_xi_zero = xi < 1e-4
        z_exp = mu - sigma * torch.log(1 - u)
        z_gpd = mu + sigma / xi * ((1 - u) ** (-xi) - 1)

        z = torch.where(is_xi_zero, z_exp, z_gpd)

        if torch.isinf(z).any():
            print("z has inf!!!!")

            # # Tìm vị trí inf đầu tiên
            # idx = torch.isinf(z).nonzero(as_tuple=False)[0]  # shape (5,) → [sample, time, layer, batch, node]

            # # Số bước lấy xung quanh (trước/sau)
            # pad = 2

            # # Extract slice indices
            # s, t, l, b, n = idx.tolist()

            # # Giới hạn vùng lấy (để không vượt chỉ số)
            # s_start, s_end = max(0, s - pad), min(z.shape[0], s + pad + 1)
            # t_start, t_end = max(0, t - pad), min(z.shape[1], t + pad + 1)
            # l_start, l_end = max(0, l - pad), min(z.shape[2], l + pad + 1)
            # b_start, b_end = max(0, b - pad), min(z.shape[3], b + pad + 1)
            # n_start, n_end = max(0, n - pad), min(z.shape[4], n + pad + 1)

            # # Trích vùng lân cận (subtensor nhỏ)
            # z_slice = z[s_start:s_end, t_start:t_end, l_start:l_end, b_start:b_end, n_start:n_end]
            # sigma_slice = sigma[t_start:t_end, l_start:l_end, b_start:b_end, n_start:n_end]
            # xi_slice = xi[t_start:t_end, l_start:l_end, b_start:b_end, n_start:n_end]
            # u_slice = u[s_start:s_end, t_start:t_end, l_start:l_end, b_start:b_end, n_start:n_end]
            # is_xi_zero_slice = is_xi_zero[t_start:t_end, l_start:l_end, b_start:b_end, n_start:n_end]

            # # Ghi ra file
            # def save_tensor(name, tensor):
            #     with open(f"{name}.txt", "w") as f:
            #         f.write(str(tensor.tolist()))

            # save_tensor("z", z_slice)
            # save_tensor("sigma", sigma_slice)
            # save_tensor("xi", xi_slice)
            # save_tensor("u", u_slice)
            # save_tensor("is_xi_zero", is_xi_zero_slice)

        return z

        # if torch.any(torch.abs(xi) < 1e-4):
        #     # GPD giảm về exponential
        #     z = mu - sigma * torch.log(1 - u)
        #     if torch.isinf(z).any():
        #         print("z has inf!!!!")
        #         with open("z.txt", "w") as f:
        #             f.write(str(z.tolist()))
        #         with open("sigma.txt", "w") as f:
        #             f.write(str(sigma.tolist()))
        #         with open("u.txt", "w") as f:
        #             f.write(str(u.tolist()))
                
        #     # z = torch.clamp(z, min=-1e4, max=1e4)
        #     return z
        # else:
        #     z = mu + sigma / xi * ((1 - u) ** (-xi) - 1)
        #     if torch.isinf(z).any():
        #         print("z has inf!!!!")
        #         with open("z.txt", "w") as f:
        #             f.write(str(z.tolist()))
        #         with open("sigma.txt", "w") as f:
        #             f.write(str(sigma.tolist()))
        #         with open("u.txt", "w") as f:
        #             f.write(str(u.tolist()))
        #         with open("xi.txt", "w") as f:
        #             f.write(str(xi.tolist()))
        #     # z = torch.clamp(z, min=-1e4, max=1e4)
        #     return z

    def log_prob2(self, value):
        xi = self.concentration
        sigma = self.scale
        mu = self.loc
        
        z = (value - mu) / sigma
        z = torch.clamp(z, min=-1e4, max=1e4)

        inside = 1 + xi * z

        log_pdf = -torch.log(sigma) - (1/xi + 1) * torch.log(inside)
        log_pdf = torch.where(
            inside > 0,
            log_pdf,
            torch.full_like(log_pdf, 0)
        )
        return log_pdf

    def log_prob(self, value):
        xi = self.concentration
        sigma = torch.clamp(self.scale, min=1e-6)
        mu = self.loc
        z = (value - mu) / sigma
        z = torch.clamp(z, min=-1e4, max=1e4)

        inside = 1 + xi * z
        valid = inside > 1e-3

        # GPD trở thành Exponential khi xi ≈ 0
        log_pdf_exp = -torch.log(sigma) - z

        if torch.isnan(log_pdf_exp).any():
            print("log_pdf_exp has nan!!!!")
        if torch.isinf(log_pdf_exp).any():
            print("log_pdf_exp has inf!!!!")

        # Trường hợp thường (xi ≠ 0)
        log_pdf_gpd = -torch.log(sigma) - (1 / xi + 1) * torch.log(inside)

        if torch.isnan(log_pdf_gpd).any():
            print("log_pdf_gpd has nan!!!!")
        if torch.isinf(log_pdf_gpd).any():
            print("log_pdf_gpd has inf!!!!")


        log_pdf = torch.where(torch.abs(xi) < 1e-3, log_pdf_exp, log_pdf_gpd)

        # Nếu invalid (inside <= 0), trả về 0
        log_pdf = torch.where(valid, log_pdf, torch.full_like(log_pdf, 0))

        if torch.isnan(log_pdf).any():
            print("log_pdf has nan!!!!")
        if torch.isinf(log_pdf).any():
            print("log_pdf has inf!!!!")

        return log_pdf



