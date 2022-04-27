import torch

class SpatialFilterLoss(object):
    def __init__(self, alpha, l_type):
        self.alpha = alpha
        self.l_type = l_type

    def __call__(self, resi, frame_list):
        """
        resi: (B,T,F,2), frame_list: list
        """
        b_size, seq_len, freq_num, _ = resi.shape
        mask_for_loss = []
        with torch.no_grad():
            for i in range(b_size):
                tmp_mask = torch.ones((frame_list[i], freq_num, 2), dtype=resi.dtype)
                mask_for_loss.append(tmp_mask)
            mask_for_loss = torch.nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(resi.device)
            mag_mask_for_loss = mask_for_loss[...,0]

        resi_mag = torch.norm(resi, dim=-1)
        if self.l_type == "L1" or self.l_type == "l1":
            loss_com = (torch.abs(resi) * mask_for_loss).sum() / mask_for_loss.sum()
            loss_mag = (torch.abs(resi_mag) * mag_mask_for_loss).sum() / mag_mask_for_loss.sum()
        elif self.l_type == "L2" or self.l_type == "l2":
            loss_com = (torch.square(resi) * mask_for_loss).sum() / mask_for_loss.sum()
            loss_mag = (torch.square(resi_mag) * mag_mask_for_loss).sum() / mag_mask_for_loss.sum()
        else:
            raise RuntimeError("only L1 and L2 are supported")
        return self.alpha * loss_com + (1 - self.alpha) * loss_mag


class ComMagEuclideanLoss(object):
    def __init__(self, alpha, l_type):
        self.alpha = alpha
        self.l_type = l_type

    def __call__(self, est, label, frame_list):
        """
            est: (B,T,F,2)
            label: (B,T,F,2)
            frame_list: list
            alpha: scalar
            l_type: str, L1 or L2
            """
        b_size, seq_len, freq_num, _ = est.shape
        mask_for_loss = []
        with torch.no_grad():
            for i in range(b_size):
                tmp_mask = torch.ones((frame_list[i], freq_num, 2), dtype=est.dtype)
                mask_for_loss.append(tmp_mask)
            mask_for_loss = torch.nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(est.device)
            mag_mask_for_loss = mask_for_loss[...,0]
        est_mag, label_mag = torch.norm(est, dim=-1), torch.norm(label, dim=-1)

        if self.l_type == "L1" or self.l_type == "l1":
            loss_com = (torch.abs(est - label) * mask_for_loss).sum() / mask_for_loss.sum()
            loss_mag = (torch.abs(est_mag - label_mag) * mag_mask_for_loss).sum() / mag_mask_for_loss.sum()
        elif self.l_type == "L2" or self.l_type == "l2":
            loss_com = (torch.square(est - label) * mask_for_loss).sum() / mask_for_loss.sum()
            loss_mag = (torch.square(est_mag - label_mag) * mag_mask_for_loss).sum() / mag_mask_for_loss.sum()
        else:
            raise RuntimeError("only L1 and L2 are supported!")
        return self.alpha * loss_com + (1 - self.alpha) * loss_mag
