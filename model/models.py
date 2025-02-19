import torch as th
import torch.nn as nn

from .modules import HyperLinearBlock, FourierFeatureMapping


class FreqSrcPosCondAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Stats for standardization
        self.stats = {
            "none": {
                "hrtf_mag": {"mean": 0.0, "std": 1.0},
                "itd": {"mean": 0.0, "std": 1.0}
            }
        }

        # Fourier Feature Mapping
        self.ffm_srcpos = FourierFeatureMapping(
            num_features=config.fourier_feature_mapping.num_features.source_position,
            input_dim=3,
            trainable=config.fourier_feature_mapping.trainable)
        self.ffm_freq = FourierFeatureMapping(
            num_features=config.fourier_feature_mapping.num_features.frequency,
            input_dim=1,
            trainable=config.fourier_feature_mapping.trainable)
        self.radius_norm = config.radius_norm
        self.freq_norm = config.freq_norm
        self.num_mes_norm = config.num_mes_norm

        # Encoder
        dim_cond_vec_enc = config.fourier_feature_mapping.num_features.source_position * 2 + config.fourier_feature_mapping.num_features.frequency * 2 + 2
        modules = []
        for l_e in range(config.encoder.num_layers):
            in_dim = 1 if l_e == 0 else config.encoder.mid_dim
            out_dim = config.encoder.out_dim if l_e == config.encoder.num_layers - 1 else config.encoder.mid_dim
            post_prcs = l_e != (config.encoder.num_layers - 1)

            modules.extend([
                HyperLinearBlock(in_dim=in_dim,
                                 out_dim=out_dim,
                                 hidden_dim=config.weight_bias_generator.mid_dim,
                                 num_hidden=config.weight_bias_generator.num_layers,
                                 cond_dim=dim_cond_vec_enc, post_prcs=post_prcs),
            ])
        self.encoder = nn.Sequential(*modules)

        # Decoder
        dim_cond_vec_dec = config.fourier_feature_mapping.num_features.source_position * 2 + config.fourier_feature_mapping.num_features.frequency * 2 + 1
        modules = []
        for l_d in range(config.decoder.num_layers):
            in_dim = config.decoder.in_dim if l_d == 0 else config.decoder.mid_dim
            out_dim = 1 if l_d == config.decoder.num_layers - 1 else config.decoder.mid_dim
            post_prcs = l_d != (config.encoder.num_layers - 1)

            modules.extend([
                HyperLinearBlock(in_dim=in_dim,
                                 out_dim=out_dim,
                                 hidden_dim=config.weight_bias_generator.mid_dim,
                                 num_hidden=config.weight_bias_generator.num_layers,
                                 cond_dim=dim_cond_vec_dec, post_prcs=post_prcs),
            ])
        self.decoder = nn.Sequential(*modules)

    def set_stats(self, mean, std, dataset_name, data_type="hrtf_mag"):
        if dataset_name not in self.stats:
            self.stats[dataset_name] = {}
        self.stats[dataset_name][data_type] = {"mean": mean, "std": std}

    def standardize(self, input, dataset_name, data_type="hrtf_mag", reverse=False, device="cuda"):
        if not reverse:
            output = (input - self.stats[dataset_name][data_type]["mean"].to(device)) / self.stats[dataset_name][data_type]["std"].to(device)
        else:
            output = input * self.stats[dataset_name][data_type]["std"].to(device) + self.stats[dataset_name][data_type]["mean"].to(device)
        return output

    def switch_device(self, inputs, device="cuda"):
        outputs = []
        for input in inputs:
            outputs.append(input.to(device))
        return outputs

    def get_conditioning_vector(self, pos_cart, freq, use_num_pos=False, device="cuda"):
        S, B, _ = pos_cart.shape
        L = freq.shape[1]
        conditioning_vector = []

        pos_cart = pos_cart / self.radius_norm  # (S, B, 3)
        pos_cart_lr_flip = pos_cart * th.tensor([1, -1, 1], device=device)[None, None, :]  # (S, B, 3)

        pos_cart = self.ffm_srcpos(pos_cart)  # (S, B, 32)
        pos_cart = pos_cart.unsqueeze(2).tile(1, 1, L, 1)  # (S, B, L, 32)

        pos_cart_lr_flip = self.ffm_srcpos(pos_cart_lr_flip)  # (S, B, 32)
        pos_cart_lr_flip = pos_cart_lr_flip.unsqueeze(2).tile(1, 1, L, 1)  # (S, B, L, 32)

        pos_cart_all = th.cat((pos_cart, pos_cart_lr_flip, pos_cart[:, :, 0:1, :]), dim=2)  # (S, B, 2L+1, 32)
        conditioning_vector.append(pos_cart_all)

        freq = freq / self.freq_norm
        freq = self.ffm_freq(freq.unsqueeze(-1))  # (1, L, 16)
        freq = freq.reshape(1, 1, L, -1).tile(S, B, 2, 1)  # (S, B, 2L, 16)
        freq = th.cat((freq, th.zeros(S, B, 1, freq.shape[-1], device=device, dtype=th.float32)), dim=2)  # (S, B, 2L+1, 16)
        conditioning_vector.append(freq)

        if use_num_pos:
            num_pos = B / self.num_mes_norm * th.ones(S, B, 2 * L + 1, 1, device=device, dtype=th.float32)  # (S, B, 2L+1, 1)
            conditioning_vector.append(num_pos)

        delta = th.cat((th.zeros(2 * L, device=device, dtype=th.float32),
                        th.ones(1, device=device, dtype=th.float32)), dim=0)  # (2L + 1)
        delta = delta.reshape(1, 1, 2 * L + 1, 1).tile(S, B, 1, 1)  # (S, B, 2L+1, 1)
        conditioning_vector.append(delta)

        conditioning_vector = th.cat(conditioning_vector, dim=-1)  # (S, B, 2L+1, 50 or 49)
        return conditioning_vector.to(device)

    def forward(self, hrtf_mag, itd, freq, mes_pos_cart, tar_pos_cart, dataset_name="none", device="cuda"):
        '''
        Args:
            hrtf_mag:     (S ,B_m, 2, L)
            itd:          (S, B_m)
            freq:         (S, L)
            mes_pos_cart: (S, B_m, 3)
            tar_pos_cart: (S, B_t, 3)
            dataset_name: str
            device: str

        Returns:
            hrtf_mag_pred: (S, B_t, 2, L)
            itd_pred:      (S, B_t)
        '''
        _, B_m, _, L = hrtf_mag.shape
        assert hrtf_mag.shape[1] == itd.shape[1] == B_m
        B_t = tar_pos_cart.shape[1]

        hrtf_mag, itd, freq, mes_pos_cart, tar_pos_cart = self.switch_device([hrtf_mag, itd, freq, mes_pos_cart, tar_pos_cart], device=device)

        hrtf_mag = self.standardize(hrtf_mag, dataset_name, "hrtf_mag", device=device)  # (S, B_m, 2, L)
        itd = self.standardize(itd, dataset_name, "itd", device=device).unsqueeze(-1)  # (S, B_m, 1)

        hrtf_mag = th.cat((hrtf_mag[:, :, 0, :], hrtf_mag[:, :, 1, :]), dim=-1)  # (S, B_m, 2L)
        encoder_input = th.cat((hrtf_mag, itd), dim=-1).unsqueeze(-1)  # (S, B_m, 2L+1, 1)
        encoder_cond = self.get_conditioning_vector(mes_pos_cart, freq, use_num_pos=True, device=device)  # (S, B_m, 2L+1, 50)

        latent = self.encoder((encoder_input, encoder_cond))[0]  # (S, B_m, 2L+1, D)
        prototype = th.mean(latent, dim=1, keepdim=True)  # (S, 1, 2L+1, D)

        decoder_input = prototype.tile(1, B_t, 1, 1)  # (S, B_t, 2L+1, D)
        decoder_cond = self.get_conditioning_vector(tar_pos_cart, freq, use_num_pos=False, device=device)  # (S, B_t, 2L+1, 49)
        decoder_output = self.decoder((decoder_input, decoder_cond))[0]  # (S, B_t, 2L+1, 1)

        hrtf_mag_pred = th.cat((decoder_output[:, :, None, :L, 0], decoder_output[:, :, None, L:2 * L, 0]), dim=2)  # (S, B_t, 2, L)
        itd_pred = decoder_output[:, :, -1, 0]  # (S, B_t)

        hrtf_mag_pred = self.standardize(hrtf_mag_pred, dataset_name, "hrtf_mag", reverse=True, device=device)
        itd_pred = self.standardize(itd_pred, dataset_name, "itd", reverse=True, device=device)

        return hrtf_mag_pred, itd_pred
