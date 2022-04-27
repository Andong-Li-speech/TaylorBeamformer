import torch

from torch_complex.tensor import ComplexTensor


def complex_matrix2real_matrix(c: ComplexTensor) -> torch.Tensor:
    # NOTE(kamo):
    # Complex value can be expressed as follows
    #   a + bi => a * x + b y
    # where
    #   x = |1 0|  y = |0 -1|
    #       |0 1|,     |1  0|
    # A complex matrix can be also expressed as
    #   |A -B|
    #   |B  A|
    # and complex vector can be expressed as
    #   |A|
    #   |B|
    assert c.size(-2) == c.size(-1), c.size()
    # (∗, m, m) -> (*, 2m, 2m)
    return torch.cat(
        [torch.cat([c.real, -c.imag], dim=-1), torch.cat([c.imag, c.real], dim=-1)],
        dim=-2,
    )


def complex_vector2real_vector(c: ComplexTensor) -> torch.Tensor:
    # (∗, m, k) -> (*, 2m, k)
    return torch.cat([c.real, c.imag], dim=-2)


def real_matrix2complex_matrix(c: torch.Tensor) -> ComplexTensor:
    assert c.size(-2) == c.size(-1), c.size()
    # (∗, 2m, 2m) -> (*, m, m)
    n = c.size(-1)
    assert n % 2 == 0, n
    real = c[..., : n // 2, : n // 2]
    imag = c[..., n // 2 :, : n // 2]
    return ComplexTensor(real, imag)


def real_vector2complex_vector(c: torch.Tensor) -> ComplexTensor:
    # (∗, 2m, k) -> (*, m, k)
    n = c.size(-2)
    assert n % 2 == 0, n
    real = c[..., : n // 2, :]
    imag = c[..., n // 2 :, :]
    return ComplexTensor(real, imag)
