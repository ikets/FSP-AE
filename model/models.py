import torch as th
import torch.nn as nn

from .modules import Net, HyperLinearBlock, FourierFeatureMapping


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
        return conditioning_vector

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

    def forward_old(self, data, mode='train'):
        '''
        :param: data: dict contains
            'SrcPos':       (S,B,3) torch.float32 [r,phi,theta]
            'SrcPos_Cart':  (S,B,3) torch.float32 [x,y,z]
            'HRTF':         (S,B,2,L,2) torch.float32 not implemented yet
            'HRTF_mag':     (S,B,2,L) torch.float32
            'HRIR':         (S,B,2,2L) torch.float32
            'ITD':          (S,B) torch.float32
            'db_name':      list of str
            'sub_idx':      (S) torch.int64

        :return: out:
            - HRTF produce as a (B, 4L, S) complex tensor
                4L ... [l_re, l_im, r_re, r_im]
            - HRTF's magnitude as a (B,2,L,S) float tensor
            - HRIR as a (B,2,L',S) float tensor
            - ITDs as a (B, S) float tensor
        '''
        if True:
            returns = {}
            db_name = data["db_name"][0]
            # for k in data:
            #     ic(k)
            #     if hasattr(data[k], 'shape'):
            #         ic(data[k].shape)
            #         ic(data[k].dtype)
            
            if 'HRTF' in self.config['data_kind_interp']:
                raise NotImplementedError

            for data_kind in self.config['data_kind_interp']:
                if data_kind in ['HRTF_mag','HRTF']:
                    LL = self.filter_length
                elif data_kind in ['HRIR']:
                    LL = self.filter_length * 2

            S,B,_ = data["SrcPos"].shape
            
            #===v sampling v====
            if self.config["pln_smp"]:
                idx_mes_pos = plane_sample(pts=data['SrcPos_Cart'][0], axes=self.config["pln_smp_axes"], thr=0.01)
            elif self.config["pln_smp_paral"]:
                idx_mes_pos = parallel_planes_sample(pts=data['SrcPos_Cart'][0], values=th.tensor([-.5,0.0,.5]).to(data['SrcPos'].device)*(data['SrcPos'][0,0,0]), axis=self.config['pln_smp_paral_axis'], thr=0.01)
            elif self.config["num_pts"] >= min(B,362):
                idx_mes_pos = range(0,B) 
            elif self.config["random_sample"] and mode == 'train':
                perm = th.randperm(B)
                idx_mes_pos = perm[:self.config["num_pts"]]
            # elif self.config["num_pts"] >= min(B,362):
            #     idx_mes_pos = range(0,B) 
            elif self.config["num_pts"] < 9:
                idx_mes_pos = aprox_reg_poly(pts=data['SrcPos_Cart'][0]/(data['SrcPos'][0,0,0]), num_pts=self.config["num_pts"], db_name=db_name)
            else:
                self.t = round(self.config["num_pts"]**0.5-1)
                idx_mes_pos = aprox_t_des(pts=data['SrcPos_Cart'][0]/(data['SrcPos'][0,0,0]), t=self.t, plot=False, db_name=db_name)
 
            returns["idx_mes_pos"] = idx_mes_pos
            B_mes = len(idx_mes_pos)
            data_mes = {}

            #=== Standardize ====
            for data_kind in self.config['data_kind_interp']:
                if data_kind in ['HRTF_mag','HRTF','HRIR','ITD']:
                    data[data_kind] = (data[data_kind] - self.config["Val_Standardize"][data_kind][db_name]["mean"]) / (self.config["Val_Standardize"][data_kind][db_name]["std"])
            data['SrcPos_Cart'] = data['SrcPos_Cart']/(self.config["SrcPos_Cart_norm"])
            data['Freq'] = (th.arange(self.filter_length+1)[1:]/self.filter_length).to(data['HRTF_mag'].device).unsqueeze(-1)
            # data['Freq'] = (th.arange(self.filter_length+1)[1:]/(self.filter_length/2)-1).to(data['HRTF_mag'].device).unsqueeze(-1)

            for k in data:
                if k in ['SrcPos','SrcPos_Cart','HRTF','HRTF_mag','HRIR','ITD']:
                    assert data[k].shape[1] == B
                    data_mes[k] = data[k][:,idx_mes_pos]
            #===^ sampling ^====
            

            B_mes_norm = self.config["Bp_norm"] if "Bp_norm" in self.config.keys() else B
            data['B_mes'] = th.tensor([B_mes/B_mes_norm]).to(data['HRTF_mag'].device).unsqueeze(-1)
            # data['B_mes'] = th.tensor([B_mes/B_mes_norm*2-1]).to(data['HRTF_mag'].device).unsqueeze(-1)
            
            if self.config["use_lr_aug"]:
                lr_flip_tensor = th.tensor([1,-1,1], device=data['SrcPos_Cart'].device)
                data['SrcPos_Cart_lr_flip'] = data['SrcPos_Cart'] * lr_flip_tensor[None,None,:]
                data_mes['SrcPos_Cart_lr_flip'] = data_mes['SrcPos_Cart'] * lr_flip_tensor[None,None,:]
                # (S,B,3)
            else:
                raise NotImplementedError
            
            for data_kind in self.config["data_kind_ffm"]:
                # ic.enable()
                # ic(data_kind)
                # ic(data[data_kind].device)

                for data_str in ['data','data_mes']:
                    if data_kind in eval(data_str):
                        # eval(data_str)[data_kind] = self.ffm[data_kind](eval(data_str)[data_kind])
                        exec(f'eval(data_str)[data_kind] = self.ffm_{data_kind}(eval(data_str)[data_kind])')
                        if data_kind == 'SrcPos_Cart':
                            # eval(data_str)['SrcPos_Cart_lr_flip'] = self.ffm[data_kind](eval(data_str)['SrcPos_Cart_lr_flip'])
                            exec(f"eval(data_str)['SrcPos_Cart_lr_flip'] = self.ffm_{data_kind}(eval(data_str)['SrcPos_Cart_lr_flip'])")

            device = data['HRTF_mag'].device
            hyper_en_x = th.zeros(S,B_mes,0,1, device=device, dtype=th.float32)
            for data_kind in self.config["data_kind_interp"]:
                if data_kind in ['ITD']:
                    hyper_en_x = th.cat((hyper_en_x, data_mes[data_kind][:,:,None,None]), dim=2) # (S,B_mes,1,1)
                elif data_kind in ['HRTF_mag','HRIR']:
                    hyper_en_x = th.cat((hyper_en_x, data_mes[data_kind][:,:,0,:,None], data_mes[data_kind][:,:,1,:,None]), dim=2) # (S,B_mes,2L,1)

            hyper_en_z = th.zeros(S,B_mes,2*LL+1,0, device=device, dtype=th.float32)
            for data_kind in self.config["data_kind_hyper_en"]:
                if data_kind in ['SrcPos_Cart']:
                    hyper_en_z = th.cat((hyper_en_z, th.cat((data_mes[data_kind].unsqueeze(2).tile(1,1,LL,1), data_mes[f'{data_kind}_lr_flip'].unsqueeze(2).tile(1,1,LL,1), data_mes[data_kind].unsqueeze(2)), dim=2)), dim=3) # (S,B_mes,2*LL+1,3 or num_ff)
                elif data_kind in ['Freq']:
                    freq_tensor = data[data_kind].tile(2,1)
                    freq_dammy  = th.zeros_like(freq_tensor)[0:1,:]
                    hyper_en_z = th.cat((hyper_en_z, th.cat((freq_tensor,freq_dammy), dim=0)[None,None,:,:].tile(S,B_mes,1,1)), dim=3)  # (S,B_mes,2*LL+1,1 or num_ff)
                elif data_kind in ['B_mes']:
                    hyper_en_z = th.cat((hyper_en_z, data[data_kind][None,None,:,:].tile(S,B_mes,2*LL+1,1)), dim=3) # (S,B_mes,2*LL+1,1 or num_ff)
                delta = th.cat((th.zeros(2*LL, device=device, dtype=th.float32), th.ones(1, device=device, dtype=th.float32)), dim=0)
                hyper_en_z = th.cat((hyper_en_z, delta[None,None,:,None].tile(S,B_mes,1,1)), dim=3)
            
            # ic(hyper_en_x.shape)
            # ic(hyper_en_z.shape)
            
            latents = self.encoder({
                "x": hyper_en_x, # (S,B_mes, 1 or 2LL or 2LL+1, 1)
                "z": hyper_en_z, # (S,B_mes, 1 or 2LL or 2LL+1, *)
            })["x"]
            # (S,B_mes, 1 or 2LL or 2LL+1, d)
            LLL = latents.shape[2]
            # ic(latents.shape)

            returns["z_bm"] = latents
            latents = th.mean(latents, dim=self.config["mid_mean_dim"], keepdim=True)  # (S, 1, 1 or 2LL or 2LL+1, d)
            returns["z"] = latents

            latents = latents.tile(1,B,1,1)


            # latents = latents.unsqueeze(1).tile(1,B,1,1) # (S, B, 1 or 2LL or 2LL+1, d)

            hyper_de_z = th.zeros(S,B,2*LL+1,0, device=device, dtype=th.float32)
            for data_kind in self.config["data_kind_hyper_en"]:
                if data_kind in ['SrcPos_Cart']:
                    hyper_de_z = th.cat((hyper_de_z, th.cat((data[data_kind].unsqueeze(2).tile(1,1,LL,1), data[f'{data_kind}_lr_flip'].unsqueeze(2).tile(1,1,LL,1), data[data_kind].unsqueeze(2)), dim=2)), dim=3) # (S,B,2*LL+1,3 or num_ff)
                elif data_kind in ['Freq']:
                    freq_tensor = data[data_kind].tile(2,1)
                    freq_dammy  = th.zeros_like(freq_tensor)[0:1,:]
                    hyper_de_z = th.cat((hyper_de_z, th.cat((freq_tensor,freq_dammy), dim=0)[None,None,:,:].tile(S,B,1,1)), dim=3)  # (S,B,2*LL+1,1 or num_ff)

            delta = th.cat((th.zeros(2*LL, device=device, dtype=th.float32), th.ones(1, device=device, dtype=th.float32)), dim=0)
            hyper_de_z = th.cat((hyper_de_z, delta[None,None,:,None].tile(S,B,1,1)), dim=3)
            
            # ic(latents.shape)
            # ic(hyper_de_z.shape)

            out_f = self.decoder({
                "x": latents,    # (S, B, 1 or 2LL or 2LL+1, d)
                "z": hyper_de_z, # (S, B, 1 or 2LL or 2LL+1, *)
            })["x"]

            # ic(out_f.shape)

            for data_kind in self.config["data_kind_interp"]:
                if data_kind == 'ITD':
                    returns['ITD'] = out_f[:,:,0,0]
                    out_f = out_f[:,:,0:,:]
                else:
                    returns[data_kind] = th.cat((out_f[:,:,None,:LL,0], out_f[:,:,None,LL:2*LL,0]), dim=2) # (S,B,2,LL)
                    out_f = out_f[:,:,2*LL:,:]
            
            for data_kind in self.config["data_kind_interp"]:
                returns[data_kind] = returns[data_kind] * self.config["Val_Standardize"][data_kind][db_name]["std"] + self.config["Val_Standardize"][data_kind][db_name]["mean"]
                # for dic in ['returns','data']:
                #     eval(dic)[data_kind] = eval(dic)[data_kind] * self.config["Val_Standardize"][data_kind][db_name]["std"] + self.config["Val_Standardize"][data_kind][db_name]["mean"]

        return returns

