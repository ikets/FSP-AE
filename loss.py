import torch as th
import torchaudio as ta


class LSD(th.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, dim=2, data_type="hrtf_mag"):
        '''
        Args:
            pred:   (B,2,L,S) complex (or float) tensor
            target: (B,2,L,S) complex (or float) tensor
        Returns:
            a scalar or (B,2,S) tensor
        '''
        if data_type == "hrtf":
            mag2db = ta.transforms.AmplitudeToDB(stype="magnitude")
            pred = mag2db(th.abs(pred))
            target = mag2db(th.abs(target))
        elif data_type == "hrtf_mag":
            pass
        else:
            raise NotImplementedError

        lsd = th.sqrt(th.mean((pred - target).pow(2), dim=dim))
        if self.reduction == "mean":
            lsd = th.mean(lsd)
        elif self.reduction == "sum":
            lsd = th.sum(lsd)
        elif self.reduction in ["none", None]:
            pass
        else:
            raise ValueError

        return lsd
