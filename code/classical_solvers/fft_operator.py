"""Exact-grid convolution representation of a translation-invariant matrix."""
import torch


class GridConvolution:
    def __init__(self, matrix, channels=5, shape=(16, 32, 32), receiver_side=33):
        self.channels, self.shape = channels, tuple(shape)
        self.receiver_side = receiver_side
        nz, ny, nx = self.shape
        if ny != nx or receiver_side != nx + 1:
            raise ValueError("Expected an n-cell horizontal grid and n+1 receiver nodes")
        self.side = 2 * nx
        delta = torch.arange(-(nx - 1), receiver_side, device=matrix.device)
        dy, dx = torch.meshgrid(delta, delta, indexing="ij")
        receivers = dy.clamp_min(0) * receiver_side + dx.clamp_min(0)
        voxels = (-dy).clamp_min(0) * nx + (-dx).clamp_min(0)
        kernel = matrix.new_empty(channels, nz, self.side, self.side)
        for component in range(channels):
            for depth in range(nz):
                kernel[component, depth] = matrix[receivers * channels + component,
                                                  depth * ny * nx + voxels]
        self.kernel = torch.roll(kernel, shifts=(-(ny - 1), -(nx - 1)), dims=(-2, -1))
        self.spectrum = torch.fft.rfft2(self.kernel)

    def forward(self, density):
        count = density.shape[0]
        padded = density.new_zeros(count, self.shape[0], self.side, self.side)
        padded[:, :, :self.shape[1], :self.shape[2]] = density.reshape(count, *self.shape)
        spectrum = torch.fft.rfft2(padded)
        response = torch.fft.irfft2(torch.einsum("bzuv,czuv->bcuv", spectrum, self.spectrum),
                                     s=(self.side, self.side))
        response = response[:, :, :self.receiver_side, :self.receiver_side]
        return response.permute(0, 2, 3, 1).reshape(count, -1)

    def adjoint(self, response):
        count = response.shape[0]
        grid = response.reshape(count, self.receiver_side, self.receiver_side, self.channels).permute(0, 3, 1, 2)
        padded = response.new_zeros(count, self.channels, self.side, self.side)
        padded[:, :, :self.receiver_side, :self.receiver_side] = grid
        spectrum = torch.fft.rfft2(padded)
        density = torch.fft.irfft2(torch.einsum("bcuv,czuv->bzuv", spectrum, self.spectrum.conj()),
                                    s=(self.side, self.side))
        return density[:, :, :self.shape[1], :self.shape[2]].reshape(count, -1)

    @torch.no_grad()
    def audit(self, matrix):
        nz, ny, nx = self.shape
        y, x = torch.meshgrid(torch.arange(ny, device=matrix.device),
                              torch.arange(nx, device=matrix.device), indexing="ij")
        largest = matrix.new_zeros(())
        squared = matrix.new_zeros(())
        for receiver in range(self.receiver_side**2):
            ry, rx = divmod(receiver, self.receiver_side)
            reconstructed = self.kernel[:, :, (ry - y) % self.side, (rx - x) % self.side]
            expected = matrix[receiver * self.channels:(receiver + 1) * self.channels]
            error = reconstructed.reshape_as(expected) - expected
            largest = torch.maximum(largest, error.abs().max())
            squared += error.square().sum()
        generator = torch.Generator(device=matrix.device).manual_seed(202609071)
        model = torch.randn(4, matrix.shape[1], device=matrix.device, dtype=matrix.dtype, generator=generator)
        data = torch.randn(4, matrix.shape[0], device=matrix.device, dtype=matrix.dtype, generator=generator)
        forward = self.forward(model)
        adjoint = self.adjoint(data)
        direct = model @ matrix.T
        direct_adjoint = data @ matrix
        inner_l, inner_r = (forward * data).sum(), (model * adjoint).sum()
        result = {"full_matrix_max_abs_error": float(largest),
                  "full_matrix_relative_frobenius_error": float(squared.sqrt() / matrix.norm()),
                  "forward_relative_error": float((forward - direct).norm() / direct.norm()),
                  "adjoint_relative_error": float((adjoint - direct_adjoint).norm() / direct_adjoint.norm()),
                  "adjoint_inner_product_relative_error": float((inner_l - inner_r).abs() / torch.maximum(inner_l.abs(), inner_r.abs()))}
        if result["full_matrix_relative_frobenius_error"] > 1e-7 or max(
                result["forward_relative_error"], result["adjoint_relative_error"],
                result["adjoint_inner_product_relative_error"]) > 3e-6:
            raise RuntimeError(f"Matrix/convolution audit failed: {result}")
        return result


