from distutils.version import LooseVersion
import functools
from typing import Sequence
from typing import Union

import torch
from torch.nn import functional as F

from torch_complex.tensor import ComplexTensor
from torch_complex.utils import complex_matrix2real_matrix
from torch_complex.utils import complex_vector2real_vector
from torch_complex.utils import real_matrix2complex_matrix
from torch_complex.utils import real_vector2complex_vector


__all__ = [
    "einsum",
    "cat",
    "stack",
    "pad",
    "squeeze",
    "reverse",
    "trace",
    "allclose",
    "matmul",
    "solve",
]


def _fcomplex(func, nthargs=0):
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Union[ComplexTensor, torch.Tensor]:
        signal = args[nthargs]
        if isinstance(signal, ComplexTensor):
            real_args = args[:nthargs] + (signal.real,) + args[nthargs + 1 :]
            imag_args = args[:nthargs] + (signal.imag,) + args[nthargs + 1 :]
            real = func(*real_args, **kwargs)
            imag = func(*imag_args, **kwargs)
            return ComplexTensor(real, imag)
        else:
            return func(*args, **kwargs)

    return wrapper


def einsum(equation, *operands):
    """Einsum

    >>> import numpy
    >>> def get(*shape):
    ...     real = numpy.random.rand(*shape)
    ...     imag = numpy.random.rand(*shape)
    ...     return real + 1j * imag
    >>> x = get(3, 4, 5)
    >>> y = get(3, 5, 6)
    >>> z = get(3, 6, 7)
    >>> test = einsum('aij,ajk,akl->ail',
    ...               [ComplexTensor(x), ComplexTensor(y), ComplexTensor(z)])
    >>> valid = numpy.einsum('aij,ajk,akl->ail', x, y, z)
    >>> numpy.testing.assert_allclose(test.numpy(), valid)
    >>> _ = einsum('aij->ai', ComplexTensor(x))
    >>> _ = einsum('aij->ai', [ComplexTensor(x)])

    """
    if len(operands) == 1 and isinstance(operands[0], (tuple, list)):
        operands = operands[0]

    x = operands[0]
    if isinstance(x, ComplexTensor):
        real_operands = [[x.real]]
        imag_operands = [[x.imag]]
    else:
        real_operands = [[x]]
        imag_operands = []

    for x in operands[1:]:
        if isinstance(x, ComplexTensor):
            real_operands, imag_operands = (
                [ops + [x.real] for ops in real_operands]
                + [ops + [-x.imag] for ops in imag_operands],
                [ops + [x.imag] for ops in real_operands]
                + [ops + [x.real] for ops in imag_operands],
            )
        else:
            real_operands = [ops + [x] for ops in real_operands]
            imag_operands = [ops + [x] for ops in imag_operands]

    real = sum([torch.einsum(equation, ops) for ops in real_operands])
    imag = sum([torch.einsum(equation, ops) for ops in imag_operands])
    return ComplexTensor(real, imag)


def cat(seq: Sequence[Union[ComplexTensor, torch.Tensor]], *args, **kwargs):
    """
    cat(seq, dim=0, *, out=None)
    cat(seq, axis=0, *, out=None)
    """
    reals = [v.real if isinstance(v, ComplexTensor) else v for v in seq]
    imags = [
        v.imag if isinstance(v, ComplexTensor) else torch.zeros_like(v.real)
        for v in seq
    ]
    out = kwargs.pop("out", None)
    if out is not None:
        out = out
        out_real = out.real
        out_imag = out.imag
    else:
        out_real = out_imag = None
    return ComplexTensor(
        torch.cat(reals, *args, out=out_real, **kwargs),
        torch.cat(imags, *args, out=out_imag, **kwargs),
    )


def stack(seq: Sequence[Union[ComplexTensor, torch.Tensor]], *args, **kwargs):
    """
    stack(tensors, dim=0, * out=None)
    stack(tensors, axis=0, * out=None)

    """
    reals = [v.real if isinstance(v, ComplexTensor) else v for v in seq]
    imags = [
        v.imag if isinstance(v, ComplexTensor) else torch.zeros_like(v.real)
        for v in seq
    ]

    out = kwargs.pop("out", None)
    if out is not None:
        out_real = out.real
        out_imag = out.imag
    else:
        out_real = out_imag = None
    return ComplexTensor(
        torch.stack(reals, *args, out=out_real, **kwargs),
        torch.stack(imags, *args, out=out_imag, **kwargs),
    )


pad = _fcomplex(F.pad)
squeeze = _fcomplex(torch.squeeze)


@_fcomplex
def reverse(tensor: torch.Tensor, dim=0) -> torch.Tensor:
    # https://discuss.pytorch.org/t/how-to-reverse-a-torch-tensor/382
    idx = [i for i in range(tensor.size(dim) - 1, -1, -1)]
    idx = torch.LongTensor(idx).to(tensor.device)
    inverted_tensor = tensor.index_select(dim, idx)
    return inverted_tensor


@_fcomplex
def signal_frame(
    signal: torch.Tensor, frame_length: int, frame_step: int, pad_value=0
) -> torch.Tensor:
    """Expands signal into frames of frame_length.

    Args:
        signal : (B * F, D, T)
    Returns:
        torch.Tensor: (B * F, D, T, W)
    """
    signal = F.pad(signal, (0, frame_length - 1), "constant", pad_value)
    indices = sum(
        [
            list(range(i, i + frame_length))
            for i in range(0, signal.size(-1) - frame_length + 1, frame_step)
        ],
        [],
    )

    signal = signal[..., indices].view(*signal.size()[:-1], -1, frame_length)
    return signal


def trace(a: ComplexTensor) -> ComplexTensor:
    if LooseVersion(torch.__version__) >= LooseVersion("1.3"):
        datatype = torch.bool
    else:
        datatype = torch.uint8
    E = torch.eye(a.shape[-1], dtype=datatype).expand(*a.size())
    if LooseVersion(torch.__version__) >= LooseVersion("1.1"):
        E = E.type(torch.bool)
    return a[E].view(*a.size()[:-1]).sum(-1)


def allclose(
    a: Union[ComplexTensor, torch.Tensor],
    b: Union[ComplexTensor, torch.Tensor],
    rtol=1e-05,
    atol=1e-08,
    equal_nan=False,
) -> bool:
    if isinstance(a, ComplexTensor) and isinstance(b, ComplexTensor):
        return torch.allclose(
            a.real, b.real, rtol=rtol, atol=atol, equal_nan=equal_nan
        ) and torch.allclose(a.imag, b.imag, rtol=rtol, atol=atol, equal_nan=equal_nan)
    elif not isinstance(a, ComplexTensor) and isinstance(b, ComplexTensor):
        return torch.allclose(
            a.real, b.real, rtol=rtol, atol=atol, equal_nan=equal_nan
        ) and torch.allclose(
            torch.zeros_like(b.imag), b.imag, rtol=rtol, atol=atol, equal_nan=equal_nan
        )
    elif isinstance(a, ComplexTensor) and not isinstance(b, ComplexTensor):
        return torch.allclose(
            a.real, b, rtol=rtol, atol=atol, equal_nan=equal_nan
        ) and torch.allclose(
            a.imag, torch.zeros_like(a.imag), rtol=rtol, atol=atol, equal_nan=equal_nan
        )
    else:
        return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def matmul(
    a: Union[ComplexTensor, torch.Tensor], b: Union[ComplexTensor, torch.Tensor]
) -> ComplexTensor:
    if isinstance(a, ComplexTensor) and isinstance(b, ComplexTensor):
        return a @ b
    elif not isinstance(a, ComplexTensor) and isinstance(b, ComplexTensor):
        o_real = torch.matmul(a, b.real)
        o_imag = torch.matmul(a, b.imag)
    elif isinstance(a, ComplexTensor) and not isinstance(b, ComplexTensor):
        return a @ b
    else:
        o_real = torch.matmul(a.real, b.real)
        o_imag = torch.zeros_like(o_real)
    return ComplexTensor(o_real, o_imag)


def solve(b: ComplexTensor, a: ComplexTensor) -> ComplexTensor:
    """Solve ax = b"""
    a = complex_matrix2real_matrix(a)
    b = complex_vector2real_vector(b)
    x, LU = torch.solve(b, a)
    return real_vector2complex_vector(x), real_matrix2complex_matrix(LU)
