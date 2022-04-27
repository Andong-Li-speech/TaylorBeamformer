import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from torch_complex.tensor import ComplexTensor
import torch_complex.functional as F
from utils.utils import complex_mul, complex_conj, NormSwitch

class GeneralizedMultichannelWienerFiter(nn.Module):
    def __init__(self,
                 k1: list,
                 k2: list,
                 c: int,
                 M: int,
                 fft_num: int,
                 hid_node: int,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 group_num: int,
                 is_gate: bool,
                 dilations: list,
                 is_causal: bool,
                 is_u2: bool,
                 rnn_type: str,
                 norm1d_type: str,
                 norm2d_type: str,
                 intra_connect: str,
                 inter_connect: str,
                 out_type: str,
                 ):
        super(GeneralizedMultichannelWienerFiter, self).__init__()
        self.k1 = tuple(k1)
        self.k2 = tuple(k2)
        self.c = c
        self.M = M
        self.fft_num = fft_num
        self.hid_node = hid_node
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.group_num = group_num
        self.is_gate = is_gate
        self.dilations = dilations
        self.is_causal = is_causal
        self.is_u2 = is_u2
        self.rnn_type = rnn_type
        self.norm1d_type = norm1d_type
        self.norm2d_type = norm2d_type
        self.intra_connect = intra_connect
        self.inter_connect = inter_connect
        self.out_type = out_type

        # Components
        # inv module
        self.inv_module = NeuralInvModule(M, hid_node, out_type, rnn_type)
        if is_u2:
            self.en = U2Net_Encoder(2*M, self.k1, self.k2, c, intra_connect, norm2d_type)
            self.de = U2Net_Decoder(c, self.k1, self.k2, fft_num, intra_connect, inter_connect, norm2d_type,
                                    out_type)
        else:
            self.en = UNet_Encoder(2*M, self.k1, c, norm2d_type)
            self.de = UNet_Decoder(c, self.k1, fft_num, inter_connect, norm2d_type, out_type)
        tcn_list = []
        for i in range(group_num):
            tcn_list.append(TCMGroup(kd1, cd1, d_feat, is_gate, dilations, is_causal, norm1d_type))
        self.tcns = nn.ModuleList(tcn_list)

    def forward(self, inpt):
        """
        inpt: (B,T,F,M,2)
        """
        inv_Phi_yy = self.inv_module(inpt)  # (B,T,F,M,M,2)
        b_size, seq_len, freq_num, M, _ = inpt.shape
        inpt1 = inpt.view(b_size, seq_len, freq_num, -1).permute(0,3,1,2).contiguous()
        en_x, en_list = self.en(inpt1)
        en_x = en_x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
        acc_x = Variable(torch.zeros_like(en_x), requires_grad=True).to(en_x.device)
        x = en_x
        for i in range(len(self.tcns)):
            x = self.tcns[i](x)
            acc_x = acc_x + x
        x = acc_x
        x = x.view(b_size, 64, 4, seq_len).transpose(-2, -1).contiguous()
        Vec_Ys = self.de(inpt, x, en_list)   # (B,T,F,M,2)

        # derive wiener filter
        inpt_complex = ComplexTensor(inpt[...,0], inpt[...,-1])  # (B,T,F,M)
        inv_Phi_yy_complex = ComplexTensor(inv_Phi_yy[...,0], inv_Phi_yy[...,-1])
        Vec_Ys_complex = ComplexTensor(Vec_Ys[...,0], Vec_Ys[...,-1])
        mcwf_bf_complex = F.einsum("...mn,...p->...m", [inv_Phi_yy_complex, Vec_Ys_complex])  # (B,T,F,M)
        bf_x_complex = F.einsum("...m,...n->...", [mcwf_bf_complex.conj(), inpt_complex])
        bf_x = torch.stack((bf_x_complex.real, bf_x_complex.imag), dim=-1)  # (B,T,F,2)
        return bf_x


class NeuralInvModule(nn.Module):
    def __init__(self,
                 M: int,
                 hid_node: int,
                 out_type: str,
                 rnn_type: str,
                 ):
        super(NeuralInvModule, self).__init__()
        self.M = M
        self.hid_node = hid_node
        self.out_type = out_type
        self.rnn_type = rnn_type

        # Components
        inpt_dim = 2*M*M
        self.norm = nn.LayerNorm([inpt_dim])
        self.rnn = getattr(nn, rnn_type)(input_size=inpt_dim, hidden_size=hid_node, num_layers=2)
        self.w_dnn = nn.Sequential(
            nn.Linear(hid_node, hid_node),
            nn.ReLU(True),
            nn.Linear(hid_node, inpt_dim))

    def forward(self, inpt):
        """
        inpt: (B,T,F,M,2)
        return: (B,T,F,M,M,2)
        """
        b_size, seq_len, freq_num, M, _ = inpt.shape
        inpt_complex = ComplexTensor(inpt[...,0], inpt[...,-1])  # (B,T,F,M)
        inpt_cov = F.einsum("...m,...n->...mn", [inpt_complex.conj(), inpt_complex])  # (B,T,F,M,M)
        inpt_cov = inpt_cov.view(b_size, seq_len, freq_num, -1)
        inpt_cov = torch.cat((inpt_cov.real, inpt_cov.imag), dim=-1)  # (B,T,F,2MM)
        inpt_cov = self.norm(inpt_cov)
        inpt_cov = inpt_cov.transpose(1,2).contiguous().view(b_size*freq_num, seq_len, -1)
        h, _ = self.rnn(inpt_cov)
        inv_cov = self.w_dnn(h)  # (BF,T,2MM)
        inv_cov = inv_cov.view(b_size, freq_num, seq_len, M, M, 2)
        return inv_cov.transpose(1, 2).contiguous()


class UNet_Encoder(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: tuple,
                 c: int,
                 norm2d_type: str,
                 ):
        super(UNet_Encoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.c = c
        self.norm2d_type = norm2d_type
        kernel_begin = (k1[0], 5)
        stride = (1, 2)
        c_final = 64
        unet = []
        unet.append(nn.Sequential(
            GateConv2d(cin, c, kernel_begin, stride, padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm2d_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm2d_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm2d_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm2d_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c_final, k1, (1,2), padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm2d_type, "2D", c_final),
            nn.PReLU(c_final)))
        self.unet_list = nn.ModuleList(unet)

    def forward(self, x: Tensor) -> tuple:
        en_list = []
        for i in range(len(self.unet_list)):
            x = self.unet_list[i](x)
            en_list.append(x)
        return x, en_list

class UNet_Decoder(nn.Module):
    def __init__(self,
                 c: int,
                 k1: tuple,
                 fft_num: int,
                 inter_connect: str,
                 norm2d_type: str,
                 out_type: str,
                 ):
        super(UNet_Decoder, self).__init__()
        self.k1 = k1
        self.c = c
        self.fft_num = fft_num
        self.inter_connect = inter_connect
        self.norm2d_type = norm2d_type
        self.out_type = out_type

        kernel_end = (k1[0], 5)
        stride = (1, 2)
        unet = []
        if inter_connect == "add":
            inter_c = c
            c_begin = 64
        elif inter_connect == "cat":
            inter_c = c*2
            c_begin = 64*2
        else:
            raise RuntimeError("Skip connections only support add or concatenate operation")

        unet.append(nn.Sequential(
            GateConvTranspose2d(c_begin, c, k1, stride),
            NormSwitch(norm2d_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConvTranspose2d(inter_c, c, k1, stride),
            NormSwitch(norm2d_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConvTranspose2d(inter_c, c, k1, stride),
            NormSwitch(norm2d_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConvTranspose2d(inter_c, c, k1, stride),
            NormSwitch(norm2d_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConvTranspose2d(inter_c, c, kernel_end, stride),
            NormSwitch(norm2d_type, "2D", c),
            nn.PReLU(c)))
        self.unet_list = nn.ModuleList(unet)
        self.out_r = nn.Sequential(
            nn.Conv2d(c, 1, (1,1), (1,1)),
            nn.Linear(fft_num//2+1, fft_num//2+1))
        self.out_i = nn.Sequential(
            nn.Conv2d(c, 1, (1,1), (1,1)),
            nn.Linear(fft_num//2+1, fft_num//2+1))

    def forward(self, inpt: Tensor, x: Tensor, en_list: list):
        """
        inpt: (B,T,F,M,2)
        return: (B,T,F,M,2)
        """
        b_size, seq_len, freq_num, _, _ = inpt.shape
        if self.inter_connect == "add":
            for i in range(len(self.unet_list)):
                tmp = x + en_list[-(i + 1)]
                x = self.unet_list[i](tmp)
        elif self.inter_connect == "cat":
            for i in range(len(self.unet_list)):
                tmp = torch.cat((x, en_list[-(i + 1)]), dim=1)
                x = self.unet_list[i](tmp)
        else:
            raise Exception("only add and cat are supported")
        # output
        if self.out_type == "mask":
            gain = torch.stack((self.out_r(x).squeeze(dim=1), self.out_i(x).squeeze(dim=1)), dim=-1)  # (B,T,F,2)
            ref_inpt = inpt[...,0,:]  # (B,T,F,2)
            Yy = complex_mul(inpt, complex_conj(ref_inpt[...,None,:]))  # (B,T,F,M,2)
            out = complex_mul(complex_conj(gain[...,None,:]), Yy)  # (B,T,F,M,2)

        elif self.out_type == "mapping":
            map = torch.stack((self.out_r(x).squeeze(dim=1), self.out_i(x).squeeze(dim=1)), dim=-1)  # (B,T,F,2)
            out = complex_mul(inpt, complex_conj(map[...,None,:]))  # (B,T,F,M,2)
        else:
            raise Exception("only mask and mapping are supported")
        return out

class U2Net_Encoder(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: tuple,
                 k2: tuple,
                 c: int,
                 intra_connect: str,
                 norm2d_type: str,
                 ):
        super(U2Net_Encoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.intra_connect = intra_connect
        self.norm2d_type = norm2d_type

        c_last = 64
        kernel_begin = (k1[0], 5)
        stride = (1, 2)
        meta_unet = []
        meta_unet.append(
            En_unet_module(cin, c, kernel_begin, k2, intra_connect, norm2d_type, scale=4, de_flag=False))
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm2d_type, scale=3, de_flag=False))
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm2d_type, scale=2, de_flag=False))
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm2d_type, scale=1, de_flag=False))
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.last_conv = nn.Sequential(
            GateConv2d(c, c_last, k1, stride, (0, 0, k1[0]-1, 0)),
            NormSwitch(norm2d_type, "2D", c_last),
            nn.PReLU(c_last)
        )

    def forward(self, x: Tensor) -> tuple:
        en_list = []
        for i in range(len(self.meta_unet_list)):
            x = self.meta_unet_list[i](x)
            en_list.append(x)
        x = self.last_conv(x)
        en_list.append(x)
        return x, en_list

class U2Net_Decoder(nn.Module):
    def __init__(self,
                 c: int,
                 k1: tuple,
                 k2: tuple,
                 fft_num: int,
                 intra_connect: str,
                 inter_connect: str,
                 norm2d_type: str,
                 out_type: str,
                 ):
        super(U2Net_Decoder, self).__init__()
        self.c = c
        self.k1 = k1
        self.k2 = k2
        self.fft_num = fft_num
        self.intra_connect = intra_connect
        self.inter_connect = inter_connect
        self.norm2d_type = norm2d_type
        self.out_type = out_type

        kernel_end = (k1[0], 5)
        stride = (1, 2)
        meta_unet = []
        if inter_connect == "add":
            inter_c = c
            c_begin = 64
        elif inter_connect == "cat":
            inter_c = c*2
            c_begin = 64*2
        else:
            raise Exception("Skip connections only support add or concatenate operation")
        meta_unet.append(
            En_unet_module(c_begin, c, k1, k2, intra_connect, norm2d_type, scale=1, de_flag=True))
        meta_unet.append(
            En_unet_module(inter_c, c, k1, k2, intra_connect, norm2d_type, scale=2, de_flag=True))
        meta_unet.append(
            En_unet_module(inter_c, c, k1, k2, intra_connect, norm2d_type, scale=3, de_flag=True))
        meta_unet.append(
            En_unet_module(inter_c, c, k1, k2, intra_connect, norm2d_type, scale=4, de_flag=True))
        meta_unet.append(nn.Sequential(
            GateConvTranspose2d(inter_c, c, kernel_end, stride),
            NormSwitch(norm2d_type, "2D", c),
            nn.PReLU(c)))
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.out_r = nn.Sequential(
            nn.Conv2d(c, 1, (1, 1), (1, 1)),
            nn.Linear(fft_num//2+1, fft_num//2+1))
        self.out_i = nn.Sequential(
            nn.Conv2d(c, 1, (1, 1), (1, 1)),
            nn.Linear(fft_num//2+1, fft_num//2+1))

    def forward(self, inpt: Tensor, x: Tensor, en_list: list):
        """
        inpt: (B,T,F,M,2)
        return: (B,T,F,M,2)
        """
        b_size, seq_len, freq_num, M, _ = inpt.shape
        if self.inter_connect == "add":
            for i in range(len(self.meta_unet_list)):
                tmp = x + en_list[-(i+1)]
                x = self.meta_unet_list[i](tmp)
        elif self.inter_connect == "cat":
            for i in range(len(self.meta_unet_list)):
                tmp = torch.cat((x, en_list[-(i+1)]), dim=1)
                x = self.meta_unet_list[i](tmp)
        else:
            raise Exception("only add and cat are supported")
        # output
        if self.out_type == "mask":
            gain = torch.stack((self.out_r(x).squeeze(dim=1), self.out_i(x).squeeze(dim=1)), dim=-1)  # (B,T,F,2)
            ref_inpt = inpt[..., 0, :]  # (B,T,F,2)
            Yy = complex_mul(inpt, complex_conj(ref_inpt[..., None, :]))  # (B,T,F,M,2)
            out = complex_mul(complex_conj(gain[..., None, :]), Yy)  # (B,T,F,M,2)

        elif self.out_type == "mapping":
            map = torch.stack((self.out_r(x).squeeze(dim=1), self.out_i(x).squeeze(dim=1)), dim=-1)  # (B,T,F,2)
            out = complex_mul(inpt, complex_conj(map[..., None, :]))  # (B,T,F,M,2)
        else:
            raise Exception("only mask and mapping are supported")
        return out

class En_unet_module(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 k1: tuple,
                 k2: tuple,
                 intra_connect: str,
                 norm2d_type: str,
                 scale: int,
                 de_flag: bool = False,
                 ):
        super(En_unet_module, self).__init__()
        self.cin = cin
        self.cout = cout
        self.k1 = k1
        self.k2 = k2
        self.intra_connect = intra_connect
        self.norm2d_type = norm2d_type
        self.scale = scale
        self.de_flag = de_flag

        in_conv_list = []
        if de_flag is False:
            in_conv_list.append(GateConv2d(cin, cout, k1, (1, 2), (0, 0, k1[0]-1, 0)))
        else:
            in_conv_list.append(GateConvTranspose2d(cin, cout, k1, (1, 2)))
        in_conv_list.append(NormSwitch(norm2d_type, "2D", cout))
        in_conv_list.append(nn.PReLU(cout))
        self.in_conv = nn.Sequential(*in_conv_list)

        enco_list, deco_list = [], []
        for _ in range(scale):
            enco_list.append(Conv2dunit(k2, cout, norm2d_type))
        for i in range(scale):
            if i == 0:
                deco_list.append(Deconv2dunit(k2, cout, "add", norm2d_type))
            else:
                deco_list.append(Deconv2dunit(k2, cout, intra_connect, norm2d_type))
        self.enco = nn.ModuleList(enco_list)
        self.deco = nn.ModuleList(deco_list)
        self.skip_connect = Skip_connect(intra_connect)


    def forward(self, inputs: Tensor) -> Tensor:
        x_resi = self.in_conv(inputs)
        x = x_resi
        x_list = []
        for i in range(len(self.enco)):
            x = self.enco[i](x)
            x_list.append(x)

        for i in range(len(self.deco)):
            if i == 0:
                x = self.deco[i](x)
            else:
                x_con = self.skip_connect(x, x_list[-(i+1)])
                x = self.deco[i](x_con)
        x_resi = x_resi + x
        del x_list
        return x_resi

class Conv2dunit(nn.Module):
    def __init__(self,
                 k: tuple,
                 c: int,
                 norm2d_type: str,
                 ):
        super(Conv2dunit, self).__init__()
        self.k, self.c = k, c
        self.norm2d_type = norm2d_type
        k_t = k[0]
        stride = (1, 2)
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConstantPad2d((0, 0, k_t-1, 0), value=0.),
                nn.Conv2d(c, c, k, stride),
                NormSwitch(norm2d_type, "2D", c),
                nn.PReLU(c)
                )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(c, c, k, stride),
                NormSwitch(norm2d_type, "2D", c),
                nn.PReLU(c)
            )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class Deconv2dunit(nn.Module):
    def __init__(self,
                 k: tuple,
                 c: int,
                 intra_connect: str,
                 norm2d_type: str,
                 ):
        super(Deconv2dunit, self).__init__()
        self.k, self.c = k, c
        self.intra_connect = intra_connect
        self.norm2d_type = norm2d_type
        k_t = k[0]
        stride = (1, 2)
        deconv_list = []
        if self.intra_connect == "add":
            if k_t > 1:
                deconv_list.append(nn.ConvTranspose2d(c, c, k, stride)),
                deconv_list.append(Chomp_T(k_t-1))
            else:
                deconv_list.append(nn.ConvTranspose2d(c, c, k, stride))
        elif self.intra_connect == "cat":
            if k_t > 1:
                deconv_list.append(nn.ConvTranspose2d(2*c, c, k, stride))
                deconv_list.append(Chomp_T(k_t-1))
            else:
                deconv_list.append(nn.ConvTranspose2d(2*c, c, k, stride))
        deconv_list.append(NormSwitch(norm2d_type, "2D", c))
        deconv_list.append(nn.PReLU(c))
        self.deconv = nn.Sequential(*deconv_list)

    def forward(self, inputs: Tensor) -> Tensor:
        assert inputs.dim() == 4
        return self.deconv(inputs)

class GateConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: tuple,
                 padding: tuple,
                 ):
        super(GateConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        k_t = kernel_size[0]
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConstantPad2d(padding, value=0.),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size, stride=stride))
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size,
                                  stride=stride)
    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(dim=1)
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()

class GateConvTranspose2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: tuple,
                 ):
        super(GateConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        k_t = kernel_size[0]
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size,
                                   stride=stride),
                Chomp_T(k_t-1))
        else:
            self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size,
                                           stride=stride)

    def forward(self, inputs: Tensor) -> Tensor:
        assert inputs.dim() == 4
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()

class Skip_connect(nn.Module):
    def __init__(self,
                 connect):
        super(Skip_connect, self).__init__()
        self.connect = connect

    def forward(self, x_main, x_aux):
        if self.connect == "add":
            x = x_main + x_aux
        elif self.connect == "cat":
            x = torch.cat((x_main, x_aux), dim=1)
        return x

class TCMGroup(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 is_gate: bool,
                 dilations: list,
                 is_causal: bool,
                 norm1d_type: str,
                 ):
        super(TCMGroup, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.is_gate = is_gate
        self.dilations = dilations
        self.is_causal = is_causal
        self.norm1d_type = norm1d_type

        tcm_list = []
        for i in range(len(dilations)):
            tcm_list.append(SqueezedTCM(kd1, cd1, dilation=dilations[i], d_feat=d_feat, is_gate=is_gate,
                                        is_causal=is_causal, norm1d_type=norm1d_type))
        self.tcm_list = nn.ModuleList(tcm_list)

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        for i in range(len(self.dilations)):
            x = self.tcm_list[i](x)
        return x

class SqueezedTCM(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 dilation: int,
                 d_feat: int,
                 is_gate: bool,
                 is_causal: bool,
                 norm1d_type: str,
                 ):
        super(SqueezedTCM, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.dilation = dilation
        self.d_feat = d_feat
        self.is_gate = is_gate
        self.is_causal = is_causal
        self.norm1d_type = norm1d_type

        self.in_conv = nn.Conv1d(d_feat, cd1, kernel_size=1, bias=False)
        if is_causal:
            pad = ((kd1-1)*dilation, 0)
        else:
            pad = ((kd1-1)*dilation//2, (kd1-1)*dilation//2)
        self.left_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm1d_type, "1D", cd1),
            nn.ConstantPad1d(pad, value=0.),
            nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False)
        )
        if is_gate:
            self.right_conv = nn.Sequential(
                nn.PReLU(cd1),
                NormSwitch(norm1d_type, "1D", cd1),
                nn.ConstantPad1d(pad, value=0.),
                nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False),
                nn.Sigmoid()
            )
        self.out_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm1d_type, "1D", cd1),
            nn.Conv1d(cd1, d_feat, kernel_size=1, bias=False)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        resi = inputs
        x = self.in_conv(inputs)
        if self.is_gate:
            x = self.left_conv(x) * self.right_conv(x)
        else:
            x = self.left_conv(x)
        x = self.out_conv(x)
        x = x + resi
        return x

class Chomp_T(nn.Module):
    def __init__(self,
                 t: int):
        super(Chomp_T, self).__init__()
        self.t = t

    def forward(self, x):
        return x[:, :, :-self.t, :]


if __name__ == "__main__":
    net = GeneralizedMultichannelWienerFiter(k1=[2,3],
                                             k2=[1,3],
                                             c=64,
                                             M=6,
                                             fft_num=320,
                                             hid_node=64,
                                             kd1=5,
                                             cd1=64,
                                             d_feat=256,
                                             group_num=2,
                                             is_gate=True,
                                             dilations=[1,2,5,9],
                                             is_causal=True,
                                             is_u2=True,
                                             rnn_type="LSTM",
                                             norm1d_type="BN",
                                             norm2d_type="BN",
                                             intra_connect="cat",
                                             inter_connect="cat",
                                             out_type="mask",
                                             ).cuda()
    from utils.utils import numParams
    print(f"The number of trainable parameters:{numParams(net)}")
    import ptflops
    flops, macs = ptflops.get_model_complexity_info(net, (101,161,6,2))
    x = torch.rand([2,51,161,6,2]).cuda()
    y = net(x)
    print(f"{x.shape}->{y.shape}")
