import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from torch_complex.tensor import ComplexTensor
import torch_complex.functional as F
from utils.utils import complex_mul, complex_conj, NormSwitch
import math

class TaylorBeamformer(nn.Module):
    def __init__(self,
                 k1: list,
                 k2: list,
                 ref_mic: int,
                 c: int,
                 embed_dim: int,
                 fft_num: int,
                 order_num: int,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 dilations: list,
                 group_num: int,
                 hid_node: int,
                 M: int,
                 rnn_type: str,
                 intra_connect: str,
                 inter_connect: str,
                 out_type: str,   # ["mask", "mapping"]
                 bf_type: str,  # ["embedding", "generalized", "mvdr"]
                 norm_type: str,  # ["BN", "LN"]
                 is_compress: bool,
                 is_total_separate: bool,  # whether the encoder in the spectral domain contains no spatial info
                 is_u2: bool,
                 is_1dgate: bool,
                 is_squeezed: bool,
                 is_causal: bool,
                 is_param_share: bool,
                 ):
        super(TaylorBeamformer, self).__init__()
        self.k1 = tuple(k1)
        self.k2 = tuple(k2)
        self.ref_mic = ref_mic
        self.c = c
        self.embed_dim = embed_dim
        self.fft_num = fft_num
        self.order_num = order_num
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.dilations = dilations
        self.group_num = group_num
        self.hid_node = hid_node
        self.M = M
        self.rnn_type = rnn_type
        self.intra_connect = intra_connect
        self.inter_connect = inter_connect
        self.out_type = out_type
        self.bf_type = bf_type
        self.norm_type = norm_type
        self.is_compress = is_compress
        self.is_total_separate = is_total_separate
        self.is_u2 = is_u2
        self.is_1dgate = is_1dgate
        self.is_squeezed = is_squeezed
        self.is_causal = is_causal
        self.is_param_share = is_param_share

        assert (out_type, bf_type) in [("mask", "mvdr"), ("mask", "generalized"), ("mapping", "embedding")]
        # Components
        self.zeroorderblock = ZeroOrderBlock(self.k1, self.k2, c, embed_dim, fft_num, kd1, cd1, d_feat, dilations,
                                             group_num, hid_node, M, rnn_type, intra_connect, inter_connect, out_type,
                                             bf_type, norm_type, is_u2, is_1dgate, is_causal)
        if order_num > 0:
            if not is_total_separate:
                if is_u2:
                    self.highorderen = U2Net_Encoder(2*M, self.k1, self.k2, c, intra_connect, norm_type)
                else:
                    self.highorderen = UNet_Encoder(2*M, self.k1, c, norm_type)
            else:
                if is_u2:
                    self.highorderen = U2Net_Encoder(2, self.k1, self.k2, c, intra_connect, norm_type)
                else:
                    self.highorderen = UNet_Encoder(2, self.k1, c, norm_type)

            highorderblock_list = []
            if is_param_share:
                highorderblock_list.append(HighOrderBlock(kd1, cd1, d_feat, dilations, group_num, fft_num, is_1dgate,
                                                          is_causal, is_squeezed, norm_type))
            else:
                for i in range(order_num):
                    highorderblock_list.append(HighOrderBlock(kd1, cd1, d_feat, dilations, group_num, fft_num, is_1dgate,
                                                              is_causal, is_squeezed, norm_type))
            self.highorderblock_list = nn.ModuleList(highorderblock_list)

    def forward(self, inpt):
        """
        inpt: (B,T,F,M,2)
        return: spatial_x_wo_sum: (B,T,F,M,2) and out_term: (B,T,F,2)
        """
        if inpt.ndim == 4:
            inpt = inpt.unsqueeze(dim=-2)
        b_size, seq_len, freq_num, _, _ = inpt.shape
        # zero order process
        spatial_x = self.zeroorderblock(inpt)  # (B,T,F,2)
        # taylor unfolding process
        if self.is_compress:
            inpt_mag, inpt_phase = torch.norm(inpt, dim=-1)**0.5, torch.atan2(inpt[...,-1], inpt[...,0])
            inpt = torch.stack((inpt_mag*torch.cos(inpt_phase), inpt_mag*torch.sin(inpt_phase)), dim=-1)
            spatial_mag, spatial_phase = (torch.norm(spatial_x, dim=-1)+1e-10)**0.5, \
                                         torch.atan2(spatial_x[...,-1], spatial_x[...,0])
            spatial_x = torch.stack((spatial_mag*torch.cos(spatial_phase), spatial_mag*torch.sin(spatial_phase)), dim=1)
        else:
            spatial_x = spatial_x.permute(0,3,1,2).contiguous()
        out_term, pre_term = spatial_x, spatial_x  # (B,2,T,F)
        # high order encoding
        if self.order_num > 0:
            if not self.is_total_separate:
                inpt = inpt.view(b_size, seq_len, freq_num, -1).permute(0,3,1,2).contiguous()  # (B,2M,T,F)
            else:
                inpt = inpt[...,self.ref_mic,:].permute(0,3,1,2).contiguous()   # (B,2,T,F)
            en_x, _ = self.highorderen(inpt)
            en_x = en_x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)

            for order_id in range(self.order_num):
                if self.is_param_share:
                    update_term = self.highorderblock_list[0](en_x, pre_term) + order_id * pre_term
                else:
                    update_term = self.highorderblock_list[order_id](en_x, pre_term) + order_id * pre_term
                pre_term = update_term
                out_term = out_term + update_term / math.factorial(order_id+1)
        return spatial_x.permute(0,2,3,1), out_term.permute(0,2,3,1)


class ZeroOrderBlock(nn.Module):
    def __init__(self,
                 k1: tuple,
                 k2: tuple,
                 c: int,
                 embed_dim: int,
                 fft_num: int,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 dilations: list,
                 group_num: int,
                 hid_node: int,
                 M: int,
                 rnn_type: str,
                 intra_connect: str,
                 inter_connect: str,
                 out_type: str,
                 bf_type: str,
                 norm_type: str,
                 is_u2: bool,
                 is_1dgate: bool,
                 is_causal: bool,
                 ):
        super(ZeroOrderBlock, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.embed_dim = embed_dim
        self.fft_num = fft_num
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.dilations = dilations
        self.group_num = group_num
        self.hid_node = hid_node
        self.M = M
        self.rnn_type = rnn_type
        self.intra_connect = intra_connect
        self.inter_connect = inter_connect
        self.out_type = out_type
        self.bf_type = bf_type
        self.norm_type = norm_type
        self.is_u2 = is_u2
        self.is_1dgate = is_1dgate
        self.is_causal = is_causal

        # Components
        if is_u2:
            self.en = U2Net_Encoder(2*M, k1, k2, c, intra_connect, norm_type)
            self.de = U2Net_Decoder(c, k1, k2, embed_dim, fft_num, intra_connect, inter_connect, out_type, norm_type)
        else:
            self.en = UNet_Encoder(2*M, k1, c, norm_type)
            self.de = UNet_Decoder(c, k1, embed_dim, fft_num, inter_connect, out_type, norm_type)
        tcns = []
        for i in range(group_num):
            tcns.append(TCMGroup(kd1, cd1, d_feat, is_1dgate, dilations, is_causal, norm_type))
        self.tcns = nn.ModuleList(tcns)
        self.bf_module = BeamformingModule(embed_dim, M, hid_node, out_type, bf_type, rnn_type)


    def forward(self, inpt):
        """
        inpt: (B,T,F,M,2)
        return: (B,T,F,M,2)
        """
        b_size, seq_len, freq_num, channel_num, _ = inpt.shape
        inpt_x = inpt.contiguous().view(b_size, seq_len, freq_num, -1).permute(0,3,1,2).contiguous()
        en_x, en_list = self.en(inpt_x)
        x = en_x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
        x_acc = Variable(torch.zeros(x.size()), requires_grad=True).to(x.device)
        for i in range(self.group_num):
            x = self.tcns[i](x)
            x_acc += x
        x = x_acc
        x = x.view(b_size, -1, 4, seq_len).transpose(-2, -1).contiguous() # 4 denotes the freq size of the last encoding layer

        if self.out_type == "mask":
            est_s, est_n = self.de(inpt, x, en_list)
            bf_weight = self.bf_module(est_s, est_n)
        else:
            embed_x = self.de(inpt, x, en_list)
            bf_weight = self.bf_module(embed_x)
        bf_x = torch.sum(complex_mul(complex_conj(bf_weight), inpt), dim=-2)
        return bf_x


class HighOrderBlock(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 dilations: list,
                 group_num: int,
                 fft_num: int,
                 is_1dgate: bool,
                 is_causal: bool,
                 is_squeezed: bool,
                 norm_type: str,
                 ):
        super(HighOrderBlock, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.dilations = dilations
        self.group_num = group_num
        self.fft_num = fft_num
        self.is_1dgate = is_1dgate
        self.is_causal = is_causal
        self.is_squeezed = is_squeezed
        self.norm_type = norm_type

        in_feat = (fft_num//2+1)*2 + d_feat
        self.in_conv = nn.Conv1d(in_feat, d_feat, 1)
        if not is_squeezed:
            tcm_r_list, tcm_i_list = [], []
            for i in range(group_num):
                tcm_r_list.append(TCMGroup(kd1, cd1, d_feat, is_1dgate, dilations, is_causal, norm_type))
                tcm_i_list.append(TCMGroup(kd1, cd1, d_feat, is_1dgate, dilations, is_causal, norm_type))
            self.tcms_r, self.tcms_i = nn.ModuleList(tcm_r_list), nn.ModuleList(tcm_i_list)
        else:
            tcm_list = []
            for i in range(group_num):
                tcm_list.append(TCMGroup(kd1, cd1, d_feat, is_1dgate, dilations, is_causal, norm_type))
            self.tcms = nn.ModuleList(tcm_list)
        self.real_resi, self.imag_resi = nn.Conv1d(d_feat, fft_num//2+1, 1), nn.Conv1d(d_feat, fft_num//2+1, 1)


    def forward(self, en_x: Tensor, pre_x: Tensor) -> Tensor:
        """
        :param en_x:  (B, C, T)
        :param pre_x: (B, 2, T, F)
        :return:  (B, 2, T, F)
        """
        assert en_x.ndim == 3 and pre_x.ndim == 4
        # fuse the features
        b_size, _, seq_len, freq_num = pre_x.shape
        x1 = pre_x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
        x = torch.cat((en_x, x1), dim=1)
        # in conv
        x = self.in_conv(x)
        # STCMs
        if not self.is_squeezed:
            x_r, x_i = x, x
            for i in range(self.group_num):
                x_r, x_i = self.tcms_r[i](x_r), self.tcms_i[i](x_i)
        else:
            for i in range(self.group_num):
                x = self.tcms[i](x)
            x_r, x_i = x, x
        # generate real and imaginary parts
        x_r, x_i = self.real_resi(x_r).transpose(-2, -1), self.imag_resi(x_i).transpose(-2, -1)
        return torch.stack((x_r, x_i), dim=1).contiguous()

class UNet_Encoder(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: tuple,
                 c: int,
                 norm_type: str,
                 ):
        super(UNet_Encoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.c = c
        self.norm_type = norm_type
        kernel_begin = (k1[0], 5)
        stride = (1, 2)
        c_final = 64
        unet = []
        unet.append(nn.Sequential(
            GateConv2d(cin, c, kernel_begin, stride, padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c_final, k1, (1,2), padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c_final),
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
                 embed_dim: int,
                 fft_num: int,
                 inter_connect: str,
                 out_type: str,
                 norm_type: str,
                 ):
        super(UNet_Decoder, self).__init__()
        self.k1 = k1
        self.c = c
        self.embed_dim = embed_dim
        self.fft_num = fft_num
        self.inter_connect = inter_connect
        self.out_type = out_type
        self.norm_type = norm_type

        kernel_end = (k1[0], 5)
        stride = (1, 2)
        unet = []
        if inter_connect == "add":
            inter_c = c
            c_begin = 64
        elif inter_connect == "cat":
            inter_c = c * 2
            c_begin = 64 * 2
        else:
            raise RuntimeError("Skip connections only support add or concatenate operation")

        unet.append(nn.Sequential(
            GateConvTranspose2d(c_begin, c, k1, stride),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConvTranspose2d(inter_c, c, k1, stride),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConvTranspose2d(inter_c, c, k1, stride),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConvTranspose2d(inter_c, c, k1, stride),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        self.unet_list = nn.ModuleList(unet)
        if out_type == "mask":
            self.conv = nn.Sequential(
                GateConvTranspose2d(inter_c, c, kernel_end, stride),
                NormSwitch(norm_type, "2D", c),
                nn.PReLU(c)
            )
            self.mask_s = nn.Sequential(
                nn.Conv2d(c, 2, (1, 1), (1, 1)),
                nn.Linear(fft_num//2+1, fft_num//2+1)
            )
            self.mask_n = nn.Sequential(
                nn.Conv2d(c, 2, (1, 1), (1, 1)),
                nn.Linear(fft_num//2+1, fft_num//2+1)
            )
        elif out_type == "mapping":
            self.embed = nn.Sequential(
                GateConvTranspose2d(inter_c, embed_dim, kernel_end, stride),
                nn.Linear(fft_num//2+1, fft_num//2+1)
            )

    def forward(self, inpt: Tensor, x: Tensor, en_list: list):
        """
        inpt: (B,T,F,M,2)
        return: (B,-1,T,F)
        """
        b_size, seq_len, freq_num, _, _ = inpt.shape
        if self.inter_connect == "add":
            for i in range(len(self.unet_list)):
                tmp = x + en_list[-(i + 1)]
                x = self.unet_list[i](tmp)
            x = x + en_list[0]
        elif self.inter_connect == "cat":
            for i in range(len(self.unet_list)):
                tmp = torch.cat((x, en_list[-(i + 1)]), dim=1)
                x = self.unet_list[i](tmp)
            x = torch.cat((x, en_list[0]), dim=1)
        else:
            raise RuntimeError("only add and cat are supported")
        # output
        if self.out_type == "mask":
            x = self.conv(x)
            mask_s, mask_n = self.mask_s(x).permute(0,2,3,1).contiguous().unsqueeze(dim=-2), \
                             self.mask_n(x).permute(0,2,3,1).contiguous().unsqueeze(dim=-2)
            est_s, est_n = complex_mul(inpt, mask_s), complex_mul(inpt, mask_n)
            return est_s, est_n
        elif self.out_type == "mapping":
            out_x = self.embed(x).permute(0,2,3,1).contiguous()
            return out_x
        else:
            raise RuntimeError("only mask and mapping are supported")


class U2Net_Encoder(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: tuple,
                 k2: tuple,
                 c: int,
                 intra_connect: str,
                 norm_type: str,
                 ):
        super(U2Net_Encoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        c_last = 64
        kernel_begin = (k1[0], 5)
        stride = (1, 2)
        meta_unet = []
        meta_unet.append(
            En_unet_module(cin, c, kernel_begin, k2, intra_connect, norm_type, scale=4, de_flag=False))
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=3, de_flag=False))
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=2, de_flag=False))
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=1, de_flag=False))
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.last_conv = nn.Sequential(
            GateConv2d(c, c_last, k1, stride, (0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c_last),
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
                 embed_dim: int,
                 fft_num: int,
                 intra_connect: str,
                 inter_connect: str,
                 out_type: str,
                 norm_type: str,
                 ):
        super(U2Net_Decoder, self).__init__()
        self.c = c
        self.k1 = k1
        self.k2 = k2
        self.embed_dim = embed_dim
        self.fft_num = fft_num
        self.intra_connect = intra_connect
        self.inter_connect = inter_connect
        self.out_type = out_type
        self.norm_type = norm_type
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
            raise RuntimeError("Skip connections only support add or concatenate operation")
        meta_unet.append(
            En_unet_module(c_begin, c, k1, k2, intra_connect, norm_type, scale=1, de_flag=True))
        meta_unet.append(
            En_unet_module(inter_c, c, k1, k2, intra_connect, norm_type, scale=2, de_flag=True))
        meta_unet.append(
            En_unet_module(inter_c, c, k1, k2, intra_connect, norm_type, scale=3, de_flag=True))
        meta_unet.append(
            En_unet_module(inter_c, c, k1, k2, intra_connect, norm_type, scale=4, de_flag=True))
        self.meta_unet_list = nn.ModuleList(meta_unet)
        if out_type == "mask":
            self.conv = nn.Sequential(
                GateConvTranspose2d(inter_c, c, kernel_end, stride),
                NormSwitch(norm_type, "2D", c),
                nn.PReLU(c)
            )
            self.mask_s = nn.Sequential(
                nn.Conv2d(c, 2, (1, 1), (1, 1)),
                nn.Linear(fft_num//2+1, fft_num//2+1)
            )
            self.mask_n = nn.Sequential(
                nn.Conv2d(c, 2, (1, 1), (1, 1)),
                nn.Linear(fft_num//2+1, fft_num//2+1)
            )
        elif out_type == "mapping":
            self.embed = nn.Sequential(
                GateConvTranspose2d(inter_c, embed_dim, kernel_end, stride),
                nn.Linear(fft_num//2+1, fft_num//2+1)
            )

    def forward(self, inpt: Tensor, x: Tensor, en_list: list):
        """
        inpt: (B,T,F,M,2)
        return: (B,T,F,M,2) or (B,T,F,K)
        """
        b_size, seq_len, freq_num, _, _ = inpt.shape
        if self.inter_connect == "add":
            for i in range(len(self.meta_unet_list)):
                tmp = x + en_list[-(i+1)]
                x = self.meta_unet_list[i](tmp)
            x = x + en_list[0]
        elif self.inter_connect == "cat":
            for i in range(len(self.meta_unet_list)):
                tmp = torch.cat((x, en_list[-(i+1)]), dim=1)
                x = self.meta_unet_list[i](tmp)
            x = torch.cat((x, en_list[0]), dim=1)
        else:
            raise RuntimeError("only add and cat are supported")
        # output
        if self.out_type == "mask":
            x = self.conv(x)
            mask_s, mask_n = self.mask_s(x).permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-2), \
                             self.mask_n(x).permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-2)
            est_s, est_n = complex_mul(inpt, mask_s), complex_mul(inpt, mask_n)
            return est_s, est_n
        elif self.out_type == "mapping":
            out_x = self.embed(x).permute(0, 2, 3, 1).contiguous()
            return out_x
        else:
            raise RuntimeError("only mask and mapping are supported")


class En_unet_module(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 k1: tuple,
                 k2: tuple,
                 intra_connect: str,
                 norm_type: str,
                 scale: int,
                 de_flag: bool = False,
                 ):
        super(En_unet_module, self).__init__()
        self.cin = cin
        self.cout = cout
        self.k1 = k1
        self.k2 = k2
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        self.scale = scale
        self.de_flag = de_flag

        in_conv_list = []
        if de_flag is False:
            in_conv_list.append(GateConv2d(cin, cout, k1, (1, 2), (0, 0, k1[0]-1, 0)))
        else:
            in_conv_list.append(GateConvTranspose2d(cin, cout, k1, (1, 2)))
        in_conv_list.append(NormSwitch(norm_type, "2D", cout))
        in_conv_list.append(nn.PReLU(cout))
        self.in_conv = nn.Sequential(*in_conv_list)

        enco_list, deco_list = [], []
        for _ in range(scale):
            enco_list.append(Conv2dunit(k2, cout, norm_type))
        for i in range(scale):
            if i == 0:
                deco_list.append(Deconv2dunit(k2, cout, "add", norm_type))
            else:
                deco_list.append(Deconv2dunit(k2, cout, intra_connect, norm_type))
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
                 norm_type: str,
                 ):
        super(Conv2dunit, self).__init__()
        self.k, self.c = k, c
        self.norm_type = norm_type
        k_t = k[0]
        stride = (1, 2)
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConstantPad2d((0, 0, k_t-1, 0), value=0.),
                nn.Conv2d(c, c, k, stride),
                NormSwitch(norm_type, "2D", c),
                nn.PReLU(c)
                )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(c, c, k, stride),
                NormSwitch(norm_type, "2D", c),
                nn.PReLU(c)
            )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class Deconv2dunit(nn.Module):
    def __init__(self,
                 k: tuple,
                 c: int,
                 intra_connect: str,
                 norm_type: str,
                 ):
        super(Deconv2dunit, self).__init__()
        self.k, self.c = k, c
        self.intra_connect = intra_connect
        self.norm_type = norm_type
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
        deconv_list.append(NormSwitch(norm_type, "2D", c))
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
                 norm_type: str,
                 ):
        super(TCMGroup, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.is_gate = is_gate
        self.dilations = dilations
        self.is_causal = is_causal
        self.norm_type = norm_type

        tcm_list = []
        for i in range(len(dilations)):
            tcm_list.append(SqueezedTCM(kd1, cd1, dilation=dilations[i], d_feat=d_feat, is_gate=is_gate,
                                        is_causal=is_causal, norm_type=norm_type))
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
                 norm_type: str,
                 ):
        super(SqueezedTCM, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.dilation = dilation
        self.d_feat = d_feat
        self.is_gate = is_gate
        self.is_causal = is_causal
        self.norm_type = norm_type

        self.in_conv = nn.Conv1d(d_feat, cd1, kernel_size=1, bias=False)
        if is_causal:
            pad = ((kd1-1)*dilation, 0)
        else:
            pad = ((kd1-1)*dilation//2, (kd1-1)*dilation//2)
        self.left_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            nn.ConstantPad1d(pad, value=0.),
            nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False)
        )
        if is_gate:
            self.right_conv = nn.Sequential(
                nn.PReLU(cd1),
                NormSwitch(norm_type, "1D", cd1),
                nn.ConstantPad1d(pad, value=0.),
                nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False),
                nn.Sigmoid()
            )
        self.out_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
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


class BeamformingModule(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 M: int,
                 hid_node: int,
                 out_type: str,
                 bf_type: str,
                 rnn_type: str,
                 ):
        super(BeamformingModule, self).__init__()
        self.embed_dim = embed_dim
        self.M = M
        self.hid_node = hid_node
        self.out_type = out_type
        self.bf_type = bf_type
        self.rnn_type = rnn_type
        assert out_type in ["mask", "mapping"]
        assert bf_type in ["embedding", "generalized", "mvdr"]
        if out_type == "mask":
            inpt_dim = 2*2*M*M
        elif out_type == "mapping":
            inpt_dim = embed_dim
        else:
            raise RuntimeError("only mask and mapping are supported")

        if bf_type in ["embedding", "generalized"]:
            #self.norm = nn.GroupNorm(1, inpt_dim)  # this type of layer-norm cannot guarantee causality in the inference
            self.norm = nn.LayerNorm([inpt_dim])
            self.rnn = getattr(nn, rnn_type)(input_size=inpt_dim, hidden_size=hid_node, num_layers=2)
            self.w_dnn = nn.Sequential(
                nn.Linear(hid_node, hid_node),
                nn.ReLU(True),
                nn.Linear(hid_node, 2*M)
            )
        elif bf_type == "mvdr":
            #self.norm1 = nn.GroupNorm(1, inpt_dim//2)  # this type of layer-norm cannot guarantee causality in the inference
            #self.norm2 = nn.GroupNorm(1, inpt_dim//2)  # this type of layer-norm cannot guarantee causality in the inference
            self.norm1 = nn.LayerNorm([inpt_dim//2])
            self.norm2 = nn.LayerNorm([inpt_dim//2])
            self.rnn1 = getattr(nn, rnn_type)(input_size=inpt_dim//2, hidden_size=hid_node, num_layers=2, batch_first=True)
            self.rnn2 = getattr(nn, rnn_type)(input_size=inpt_dim//2, hidden_size=hid_node, num_layers=2, batch_first=True)
            self.pca_dnn = nn.Sequential(
                nn.Linear(hid_node, hid_node),
                nn.ReLU(True),
                nn.Linear(hid_node, 2*M)
            )
            self.inverse_dnn = nn.Sequential(
                nn.Linear(hid_node, hid_node),
                nn.ReLU(True),
                nn.Linear(hid_node, 2*M*M)
            )

    def forward(self, inpt1, inpt2=None):
        if self.out_type == "mask":
            est_s, est_n = inpt1, inpt2
            complex_s = ComplexTensor(est_s[...,0], est_s[...,-1])  # (B,T,F,M)
            complex_n = ComplexTensor(est_n[...,0], est_n[...,-1])  # (B,T,F,M)
            cov_s = F.einsum("...m,...n->...mn", [complex_s.conj(), complex_s])  # (B,T,F,M,M)
            cov_n = F.einsum("...m,...n->...mn", [complex_n.conj(), complex_n])  # (B,T,F,M,M)
            b_size, seq_len, freq_num, M, M = cov_s.shape
            cov_s, cov_n = cov_s.view(b_size, seq_len, freq_num, -1), cov_n.view(b_size, seq_len, freq_num, -1)
            cov_ss = torch.cat((cov_s.real, cov_s.imag), dim=-1).permute(0,3,1,2)  # (B,2*M*M,T,F)
            cov_nn = torch.cat((cov_n.real, cov_n.imag), dim=-1).permute(0,3,1,2)  # (B,2*M*M,T,F)
        else:
            embed_x = inpt1.permute(0,3,1,2)  # (B,-1,T,F)
            b_size, _, seq_len, freq_num = embed_x.shape

        if self.bf_type == "mvdr":
            cov_ss, cov_nn = self.norm1(cov_ss.permute(0,3,2,1).contiguous()), \
                             self.norm2(cov_nn.permute(0,3,2,1).contiguous())
            cov_ss, cov_nn = cov_ss.view(b_size*freq_num, seq_len, -1), \
                             cov_nn.view(b_size*freq_num, seq_len, -1)
            # steer vestor
            h1, _ = self.rnn1(cov_ss)
            steer_vec = self.pca_dnn(h1)
            steer_vec = steer_vec.view(b_size, freq_num, seq_len, self.M, 2).transpose(1,2)  # (B,T,F,M,2)
            # inverse rnn
            h2, _ = self.rnn2(cov_nn)
            inverse_phi = self.inverse_dnn(h2)
            inverse_phi = inverse_phi.view(b_size, freq_num, seq_len, self.M, self.M, 2).transpose(1, 2) # (B,T,F,M,M,2)
            # mvdr
            complex_steer_vec = ComplexTensor(steer_vec[...,0], steer_vec[...,-1])  # (B,T,F,M)
            complex_inverse_phi = ComplexTensor(inverse_phi[...,0], inverse_phi[...,-1])  # (B,T,F,M,M)
            nomin = F.einsum("...mn,...n->...m", [complex_inverse_phi, complex_steer_vec])  # (B,T,F,M)
            denomin = F.einsum("...m,...m->...", [complex_steer_vec.conj(), nomin])   # (B,T,F)
            bf_weight = nomin / denomin.unsqueeze(dim=-1)
            bf_weight = torch.stack((bf_weight.real, bf_weight.imag), dim=-1)   # (B,T,F,M,2)
        elif self.bf_type == "generalized":
            x = self.norm(torch.cat((cov_ss, cov_nn), dim=1).permute(0,3,2,1).contiguous())
            x = x.view(b_size*freq_num, seq_len, -1)
            h, _ = self.rnn(x)
            bf_weight = self.w_dnn(h)
            bf_weight = bf_weight.view(b_size, freq_num, seq_len, self.M, 2).transpose(1, 2)
        elif self.bf_type == "embedding":
            x = self.norm(embed_x.permute(0,3,2,1).contiguous())
            x = x.view(b_size*freq_num, seq_len, -1)
            h, _ = self.rnn(x)
            bf_weight = self.w_dnn(h)
            bf_weight = bf_weight.view(b_size, freq_num, seq_len, self.M, 2).transpose(1, 2)
        else:
            raise RuntimeError("only mvdr, generalized, and embedding are supported")
        return bf_weight


class RNN_BF(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 M: int,
                 hid_node: int,
                 out_type: str,
                 rnn_type: str,
                 ):
        super(RNN_BF, self).__init__()
        self.embed_dim = embed_dim
        self.M = M
        self.hid_node = hid_node
        self.out_type = out_type
        if out_type == "mask":
            inpt_dim = 2*2*M*M
        elif out_type == "embed":
            inpt_dim = embed_dim
        # Components
        if rnn_type == "lstm":
            setattr(self, "rnn1", nn.LSTM(input_size=inpt_dim, hidden_size=hid_node))
            setattr(self, "rnn2", nn.LSTM(input_size=hid_node, hidden_size=hid_node))
        elif rnn_type == "gru":
            setattr(self, "rnn1", nn.GRU(input_size=inpt_dim, hidden_size=hid_node))
            setattr(self, "rnn2", nn.GRU(input_size=hid_node, hidden_size=hid_node))
        self.w_dnn = nn.Sequential(
            nn.Linear(hid_node, hid_node),
            nn.ReLU(True),
            nn.Linear(hid_node, 2*M)
        )
        #self.norm = nn.GroupNorm(1, inpt_dim)  # this type of layer-norm cannot guarantee causality in the inference
        self.norm = nn.LayerNorm([inpt_dim])

    def forward(self, embed_x: Tensor) -> Tensor:
        """
        formulate the bf operation
        :param embed_x: (B, C, T, F)
        :return: (B, T, F, M, 2)
        """
        # norm
        B, _, T, F = embed_x.shape
        embed_x = embed_x.permute(0,3,2,1).contiguous()  # (B,F,T,C)
        x = self.norm(embed_x)
        x = x.view(B*F, T, -1)
        x, _ = getattr(self, "rnn1")(x)
        x, _ = getattr(self, "rnn2")(x)
        x = x.view(B, F, T, -1).transpose(1, 2).contiguous()
        bf_w = self.w_dnn(x).view(B, T, F, self.M, 2)
        return bf_w


class Chomp_T(nn.Module):
    def __init__(self,
                 t: int):
        super(Chomp_T, self).__init__()
        self.t = t

    def forward(self, x):
        return x[:, :, :-self.t, :]



if __name__ == "__main__":
    net = TaylorBeamformer(
        k1=[1,3],
        k2=[2,3],
        ref_mic=0,
        c=64,
        embed_dim=64,
        fft_num=320,
        order_num=1,
        kd1=5,
        cd1=64,
        d_feat=256,
        dilations=[1,2,5,9],
        group_num=2,
        hid_node=64,
        M=6,
        rnn_type="LSTM",
        intra_connect="cat",
        inter_connect="cat",
        out_type="mapping",
        bf_type="embedding",
        norm_type="BN",
        is_compress=True,
        is_total_separate=False,
        is_u2=True,
        is_1dgate=True,
        is_squeezed=True,
        is_causal=True,
        is_param_share=False
    ).cuda()
    x = torch.rand([3,31,161,6,2]).cuda()
    from utils.utils import numParams
    from ptflops.flops_counter import get_model_complexity_info
    print("The number of parameters:{}".format(numParams(net)))
    get_model_complexity_info(net, (101, 161, 6, 2))
    y1, y2 = net(x)
    print("{}->{},{}".format(x.shape, y1.shape, y2.shape))
