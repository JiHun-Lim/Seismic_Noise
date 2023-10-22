import torch
import numpy as np
import pandas as pd
import os
import json
import random
import seisbench.data as sbd
import wandb
import tqdm
import h5py
from model import BBGenerator, BBDiscriminator
from config import config
from util import sbd_to_dataloader, noise, get_min_max_log_pga
from goodness import weighted_goodness_psd, goodness_fft
from ppsd import PPSD_Helper
from pathlib import Path
from matplotlib import pyplot as plt
from obspy import UTCDateTime

class bbgan:
    def __init__(self, train_config: config) -> None:
        self.train_config = train_config
        self.seed = train_config.seed
        self.epoch = train_config.epoch
        self.data_path = train_config.data_path
        self.design = train_config.design
        self.loss = train_config.loss
        self.device = train_config.device
        self.z_size = train_config.z_size
        self.batch_size = train_config.batch_size
        self.critic = train_config.critic_iter
        self.reg_lambda = train_config.reg_lambda
        self.goodness_weight = train_config.goodness_weight
        self.distance = train_config.distance
        self.dev_metric = train_config.dev_metric
        self.sta_codes = os.listdir(self.data_path)
        self.test = train_config.test
        self.path = "result/" + self.design + "_" + str(self.test)
        self.best_model_path = "/best_model/generator.pt"
        with open('catalogue_ref.json', 'r') as f:
            REF = json.load(f)
        self.REF = REF


    def train(self):

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        os.mkdir(self.path)

        for sta_code in self.sta_codes:

            # wandb.init(project="seismic_noise")
            # wandb.run.name = sta_code

            best_score = -1e10

            sta_code_path = self.path + "/" + sta_code
            
            os.mkdir(sta_code_path)
            os.mkdir(sta_code_path + "/best_model")
            os.mkdir(sta_code_path + "/ppsd_plot")
            os.mkdir(sta_code_path + "/fake_sample")
            os.mkdir(sta_code_path + "/real_sample")

            self.G = BBGenerator(self.z_size).to(self.device)
            self.D = BBDiscriminator().to(self.device)
            self.optimizer_g = torch.optim.NAdam(params = self.G.parameters(), lr = self.train_config.learning_rate, \
                                                betas=self.train_config.betas)
            self.optimizer_d = torch.optim.NAdam(params=self.D.parameters(), lr = self.train_config.learning_rate, \
                                                betas=self.train_config.betas)        
            # self.optimizer_g = torch.optim.Adam(params = self.G.parameters(), lr = self.train_config.learning_rate, \
            #                                     betas=self.train_config.betas)
            # self.optimizer_d = torch.optim.Adam(params=self.D.parameters(), lr = self.train_config.learning_rate, \
            #                                     betas=self.train_config.betas)        

            for epoch in range(self.epoch):
            
                data = sbd.WaveformDataset("data/" + sta_code)
                MIN_LOG_PGA, MAX_LOG_PGA = get_min_max_log_pga(data)
                self.min_log_pga = MIN_LOG_PGA
                self.max_log_pga = MAX_LOG_PGA

                self.dataloader, train_conditional_names, generate_conditional_names = \
                    sbd_to_dataloader(data, self.design, self.batch_size)

                self.G.train()
                self.D.train()
                size = len(self.dataloader.dataset)

                for batch_idx, batch in enumerate(self.dataloader):
                    
                    real_wfs = batch["X"].unsqueeze(3)
                    real_wfs = real_wfs.to(dtype=torch.float, device=self.device)
                    real_ln_cn = batch["ln_cn"].to(dtype=torch.float, device=self.device)
                    
                    vs = [batch[v_name].to(dtype=torch.float, device=self.device) \
                          for v_name in train_conditional_names]                
                    curr_batch_size = real_wfs.shape[0]

                    fake_waveforms = []
                    real_waveforms = []

                    for i_c in range(self.critic):
                        
                        self.optimizer_d.zero_grad()
                        batch_z = torch.tensor(noise(curr_batch_size, dim=100)).to(device=self.device)
                        fake_wfs, fake_ln_cn   = self.G(batch_z, self.design, *vs)
                        
                        alpha = torch.rand(size=(curr_batch_size, 1, 1, 1)).to(device=self.device)
                        alpha_cn = alpha.view(curr_batch_size, 1)
                        
                        Xwf_p = (alpha * real_wfs + ((1.0 - alpha) * fake_wfs)).requires_grad_(True)
                        Xcn_p = (alpha_cn * real_ln_cn + ((1.0 - alpha_cn) * fake_ln_cn)).requires_grad_(True)
                        
                        D_xp = self.D(Xwf_p, Xcn_p, self.design, *vs)

                        Xout_wf = torch.autograd.Variable(torch.Tensor(curr_batch_size, 1).fill_(1.0), requires_grad=False)
                        Xout_wf = torch.ones(size=(curr_batch_size, 1)).to(device=self.device)
                        Xout_wf = Xout_wf.requires_grad_(False)
                        grads_wf = torch.autograd.grad(
                            outputs=D_xp,
                            inputs=Xwf_p,
                            grad_outputs=Xout_wf,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )[0]
                        grads_wf = grads_wf.view(grads_wf.size(0), -1)
                        
                        Xout_cn = torch.ones(size=(curr_batch_size, 1)).to(device=self.device)
                        grads_cn = torch.autograd.grad(
                            outputs=D_xp,
                            inputs=Xcn_p,
                            grad_outputs=Xout_cn,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )[0]
                        
                        grads = torch.cat([grads_wf, grads_cn,], 1)
                        
                        d_gp_loss = self.reg_lambda*((grads.norm(2, dim=1) - 1) ** 2).mean()
                        d_w_loss = -torch.mean(self.D(real_wfs, real_ln_cn, self.design, *vs)) \
                            + torch.mean(self.D(fake_wfs, fake_ln_cn, self.design, *vs))
                        d_loss = d_w_loss + d_gp_loss

                        d_loss.backward()

                        self.optimizer_d.step()

                    #     fake_waveforms.append(10**self.recover_ln_cn(torch.tensor(fake_ln_cn)) * fake_wfs.squeeze())
                    #     real_waveforms.append(10**self.recover_ln_cn(torch.tensor(real_ln_cn)) * real_wfs.squeeze())

                    # fake_waveforms = torch.cat(fake_waveforms, dim=0)
                    # real_waveforms = torch.cat(real_waveforms, dim=0)
                                
                    # ppsd_helper = PPSD_Helper(
                    #     sta_code=sta_code,
                    #     sta_type="HH",
                    #     real_wfs=real_waveforms[:1000].detach().cpu().numpy(),
                    #     fake_wfs=fake_waveforms[:1000].detach().cpu().numpy(),
                    #     cpnt_order=data.component_order,
                    #     mode='train',
                    #     ppsd_pth= Path(sta_code_path) / "ppsd_plot",
                    #     epoch=epoch
                    # )

                    # score = ppsd_helper.get_ppsd_score(self.dev_metric)

                    # wandb.log({"ppsd_score": score})

                    wandb.log({"d_loss": d_loss})

                    self.optimizer_g.zero_grad()
                    
                    batch_z = torch.tensor(noise(curr_batch_size, dim=100)).to(device=self.device)
                    fake_wfs, fake_ln_cn   = self.G(batch_z, self.design, *vs)

                    g_loss = -torch.mean( self.D(fake_wfs, fake_ln_cn, self.design, *vs) )
                                        
                    recover_fake = 10**self.recover_ln_cn(fake_ln_cn)*torch.squeeze(fake_wfs, dim=-1)
                    recover_real = 10**self.recover_ln_cn(real_ln_cn)*torch.squeeze(real_wfs, dim=-1)
                    # g_loss += self.goodness_weight * weighted_goodness_psd(sta_code, data.component_order, recover_real, recover_fake, self.loss, loss=self.distance)
                    # g_loss += self.goodness_weight * goodness_fft(torch.squeeze(real_wfs, dim=-1), torch.squeeze(fake_wfs, dim=-1), loss=self.distance)
                    L1_loss = weighted_goodness_psd(sta_code, data.component_order, recover_real, recover_fake, "L1", loss=self.distance)
                    KL_loss = weighted_goodness_psd(sta_code, data.component_order, recover_real, recover_fake, "KL", loss=self.distance)
                    kl_loss = weighted_goodness_psd(sta_code, data.component_order, recover_real, recover_fake, "kl", loss=self.distance)
                    FFT_loss = goodness_fft(torch.squeeze(real_wfs, dim=-1), torch.squeeze(fake_wfs, dim=-1), loss=self.distance)

                    wandb.log({"L1_loss": L1_loss, "KL_loss": KL_loss, "FFT_loss": FFT_loss})

                    if self.test == 0:
                        g_loss += 10 * FFT_loss + 0.01 * KL_loss
                    elif self.test == 1:
                        g_loss += 10 * FFT_loss + 0.001 * KL_loss
                    elif self.test == 2:
                        g_loss += 10 * FFT_loss + 0.0001 * KL_loss
                    elif self.test == 3:
                        g_loss += 10 * L1_loss + 0.01 * KL_loss
                    elif self.test == 4:
                        if epoch <=10:
                            g_loss += 20 * L1_loss
                        else:
                            g_loss += 0.02 * KL_loss
                    elif self.test == 5:
                        if epoch <=10:
                            g_loss += 20 * L1_loss
                        else:
                            g_loss += 0.002 * KL_loss
                    elif self.test == 6:
                        g_loss += 20 * kl_loss                    
                        # g_loss += FFT_loss + KL_loss/256

                    wandb.log({"g_loss": g_loss})

                    g_loss.backward()
                    self.optimizer_g.step()

                    # if batch_idx % 5 ==0:
                    #     current = self.batch_size * batch_idx
                    #     print(f" [{current:>5d}/{size:>5d}]")

        
                self.G.eval()
                self.D.eval()
        
                fake_waveforms = []
                real_waveforms = []
        
                with torch.no_grad():
                    for batch_idx, batch in enumerate(self.dataloader):
                                                
                        real_wfs = batch["X"].to(dtype=torch.float, device=self.device)
                        real_ln_cn = batch["ln_cn"].to(dtype=torch.float, device=self.device)
                        curr_batch_size = real_wfs.shape[0]
                        vs = [batch[v_name].to(dtype=torch.float, device=self.device) for v_name in train_conditional_names]
                        batch_z = torch.tensor(noise(curr_batch_size, dim=100)).to(device=self.device)
                        fake_wfs, fake_ln_cn  = self.G(batch_z, self.design, *vs)
                        fake_wfs   = torch.squeeze(fake_wfs, dim=-1)
                        fake_ln_cn = fake_ln_cn.detach().cpu().numpy()
                        real_ln_cn = real_ln_cn.detach().cpu().numpy()
                                                                                        
                        fake_restored_signal = 10**self.recover_ln_cn(torch.tensor(fake_ln_cn))*fake_wfs.detach().cpu()
                        real_restored_signal = 10**self.recover_ln_cn(torch.tensor(real_ln_cn))*real_wfs.detach().cpu()
                        
                        fake_waveforms.append(fake_restored_signal)
                        real_waveforms.append(real_restored_signal)
                
                fake_waveforms = torch.cat(fake_waveforms, dim=0)
                real_waveforms = torch.cat(real_waveforms, dim=0)

                hist = fake_waveforms[0].reshape(-1)
                plt.hist(hist, bins=100)

                plt.savefig("fake.png")

                plt.close
                hist = real_waveforms[0].reshape(-1)
                plt.hist(hist, bins=100)

                plt.savefig("real.png")

                ppsd_helper = PPSD_Helper(
                    sta_code=sta_code,
                    sta_type="HH",
                    real_wfs=real_waveforms[:1000].detach().cpu().numpy(),
                    fake_wfs=fake_waveforms[:1000].detach().cpu().numpy(),
                    cpnt_order=data.component_order,
                    mode='dev',
                    ppsd_pth= Path(sta_code_path) / "ppsd_plot",
                    epoch=epoch
                )

                wandb.log({"fake_ppsd": [wandb.Image(sta_code_path + "/ppsd_plot/E_fake_epoch" + str(epoch) + ".png")]})
                wandb.log({"real_ppsd": [wandb.Image(sta_code_path + "/ppsd_plot/E_real_epoch" + str(epoch) + ".png")]})

                curr_score = ppsd_helper.get_ppsd_score(self.dev_metric)

                wandb.log({"score": curr_score})
                
                fake_sample = fake_waveforms[:4, :, :]
                real_sample = real_waveforms[:4, :, :]

                plt.figure(figsize=(30, 10))
                
                for j in range(3):
                    fig_idx = 1+j
                    plt.subplot(3, 1, fig_idx)
                    plt.plot(fake_sample[0][j])
                plt.savefig(sta_code_path + "/fake_sample/3.png")
                wandb.log({"fake_sample_3": [wandb.Image(sta_code_path + "/fake_sample/3.png")]})
                plt.close()

                plt.figure(figsize=(30, 10))
                
                for j in range(3):
                    fig_idx = 1+j
                    plt.subplot(3, 1, fig_idx)
                    plt.plot(real_sample[0][j])
                plt.savefig(sta_code_path + "/real_sample/3.png")
                wandb.log({"real_sample_3": [wandb.Image(sta_code_path + "/real_sample/3.png")]})
                plt.close()

                plt.figure(figsize=(30, 10))

                for i in range(4):
                    for j in range(3):
                        fig_idx = 1+i+4*j
                        plt.subplot(3, 4, fig_idx)
                        plt.plot(fake_sample[i][j])
                plt.savefig(sta_code_path + "/fake_sample/3_4.png")

                wandb.log({"fake_sample_3_4": [wandb.Image(sta_code_path + "/fake_sample/3_4.png")]})
                plt.close()

                plt.figure(figsize=(30, 10))
                
                for i in range(4):
                    for j in range(3):
                        fig_idx = 1+i+4*j
                        plt.subplot(3, 4, fig_idx)
                        plt.plot(real_sample[i][j])
                plt.savefig(sta_code_path + "/real_sample/3_4.png")
                wandb.log({"real_sample_3_4": [wandb.Image(sta_code_path + "/real_sample/3_4.png")]})
                plt.close()

                if curr_score > best_score:
                    
                    best_score = curr_score                    
                    torch.save(self.G.state_dict(), Path(sta_code_path) / 'best_model' / "generator.pt")

            wandb.finish()

                
    def recover_ln_cn(self, ln_cn:torch.Tensor):
        
        ln_cn = ln_cn.view(-1, 3, 1)
        ln_cn = (self.max_log_pga - self.min_log_pga)*(ln_cn + 1)/2 + self.min_log_pga
        return ln_cn

    def generate(self, model_path: str, after_train: bool):

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        for sta_code in self.sta_codes:

            if after_train:
                sta_code_path = self.path + "/" + sta_code                            
            else:
                sta_code_path = model_path + "/" + sta_code            
                os.mkdir(sta_code_path)
                os.mkdir(sta_code_path + "/fake_sample")
                os.mkdir(sta_code_path + "/real_sample")

            self.G = BBGenerator(self.z_size).to(self.device)
            self.D = BBDiscriminator().to(self.device)
            self.optimizer_g = torch.optim.Adam(params = self.G.parameters(), lr = self.train_config.learning_rate, \
                                                betas=self.train_config.betas)
            self.optimizer_d = torch.optim.Adam(params=self.D.parameters(), lr = self.train_config.learning_rate, \
                                                betas=self.train_config.betas)        

            data = sbd.WaveformDataset("data/" + sta_code)
            MIN_LOG_PGA, MAX_LOG_PGA = get_min_max_log_pga(data)
            self.min_log_pga = MIN_LOG_PGA
            self.max_log_pga = MAX_LOG_PGA

            self.dataloader, train_conditional_names, generate_conditional_names = \
                sbd_to_dataloader(data, self.design, self.batch_size)

            self.G.eval()

            if after_train:
                self.G.load_state_dict(torch.load(Path(sta_code_path + model_path)))
            else:
                self.G.load_state_dict(torch.load(Path(model_path) / "generator.pt"))


            fake_waveforms = []
            real_waveforms = []
    
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.dataloader):
                                            
                    real_wfs = batch["X"].to(dtype=torch.float, device=self.device)
                    real_ln_cn = batch["ln_cn"].to(dtype=torch.float, device=self.device)
                    curr_batch_size = real_wfs.shape[0]
                    vs = [batch[v_name].to(dtype=torch.float, device=self.device) for v_name in train_conditional_names]
                    batch_z = torch.tensor(noise(curr_batch_size, dim=100)).to(device=self.device)
                    fake_wfs, fake_ln_cn  = self.G(batch_z, self.design, *vs)
                    fake_wfs   = torch.squeeze(fake_wfs, dim=-1)
                    fake_ln_cn = fake_ln_cn.detach().cpu().numpy()
                    real_ln_cn = real_ln_cn.detach().cpu().numpy()
                                                                                    
                    fake_restored_signal = 10**self.recover_ln_cn(torch.tensor(fake_ln_cn))*fake_wfs.detach().cpu()
                    real_restored_signal = 10**self.recover_ln_cn(torch.tensor(real_ln_cn))*real_wfs.detach().cpu()
                    
                    fake_waveforms.append(fake_restored_signal)
                    real_waveforms.append(real_restored_signal)
            
            fake_waveforms = torch.cat(fake_waveforms, dim=0)
            real_waveforms = torch.cat(real_waveforms, dim=0)

            hist = fake_waveforms[0].reshape(-1)
            plt.hist(hist, bins=100)
            plt.savefig("fake.png")            
            plt.close()

            hist = real_waveforms[0].reshape(-1)
            plt.hist(hist, bins=100)
            plt.savefig("real.png")            
            plt.close()

            # data2 = real_waveforms

            # iffted_data = np.zeros(data2.shape)

            # for i in range(len(data2)):

            #     for j in range(3):

            #         waveform = data2[i][j]

            #         LOG_EPS = 1e-40
            #         real_fft = np.absolute(fft(waveform))
            #         real_power_spec = real_fft ** 2
            #         real_psd = 10*np.log10(real_power_spec + LOG_EPS)

            #         a = np.random.rand(6000)
            #         b = np.sqrt(1 - np.power(a,2))
            #         x = np.sqrt(np.power(10,real_psd/10))
            #         k = np.concatenate((np.fft.ifft(a*x + b*1j*x)[200:-200],
            #                             np.fft.ifft(a*x + b*1j*x)[200:600]))
            #         iffted_data[i][j] = 3 * k


            ppsd_helper = PPSD_Helper(
                sta_code=sta_code,
                sta_type="HH",
                real_wfs=real_waveforms[:1000].detach().cpu().numpy(),
                fake_wfs=fake_waveforms[:1000].detach().cpu().numpy(),
                cpnt_order=data.component_order,
                mode='dev',
                ppsd_pth= Path(sta_code_path),
                epoch=5
            )
                
            curr_score = ppsd_helper.get_ppsd_score(self.dev_metric)
            os.mkdir(sta_code_path + "/fake_sample" + f"/curr-score_{curr_score}")

            fake_sample = fake_waveforms[:4, :, :]
            real_sample = real_waveforms[:4, :, :]
            # iffted_sample = iffted_data[:4, :, :]

            plt.figure(figsize=(30, 10))

            for i in range(4):
                for j in range(3):
                    plt.acorr(fake_sample[i][j], usevlines=True, normed=True, maxlags = 3000, lw=0.1)
        
            plt.savefig(sta_code_path + "/acorrr_fake.png")
            plt.close()

            plt.figure(figsize=(30, 10))

            for i in range(4):
                for j in range(3):
                    plt.acorr(real_sample[i][j], usevlines=True, normed=True, maxlags = 3000, lw=0.1)
        
            plt.savefig(sta_code_path + "/acorrr_real.png")
            plt.close()

            # plt.figure(figsize=(30, 10))

            # for i in range(4):
            #     for j in range(3):
            #         plt.acorr(np.random.randn(len(real_sample[i][j])), usevlines=True, normed=True, maxlags = 3000, lw=0.1)

            # plt.savefig(sta_code_path + "/acorrr_wn.png")
            # plt.close()

            # plt.figure(figsize=(30, 10))

            # for i in range(4):
            #     for j in range(3):
            #         plt.acorr(iffted_sample[i][j], usevlines=True, normed=True, maxlags = 3000, lw=0.1)

            # plt.savefig(sta_code_path + "/acorrr_iffted.png")
            # plt.close()

            plt.figure(figsize=(30, 10))

            for i in range(4):
                for j in range(3):
                    fig_idx = 1+i+4*j
                    plt.subplot(3, 4, fig_idx)
                    plt.plot(fake_sample[i][j])
            plt.savefig(sta_code_path + "/fake_sample/3_4.png")
            plt.close()

            plt.figure(figsize=(30, 10))
            
            for j in range(3):
                fig_idx = 1+j
                plt.subplot(3, 1, fig_idx)
                plt.plot(fake_sample[3][j])
            plt.savefig(sta_code_path + "/fake_sample/3.png")
            plt.close()
            
            plt.figure(figsize=(30, 10))
            
            for i in range(4):
                for j in range(3):
                    fig_idx = 1+i+4*j
                    plt.subplot(3, 4, fig_idx)
                    plt.plot(real_sample[i][j])
            plt.savefig(sta_code_path + "/real_sample/3_4.png")
            plt.close()

            plt.figure(figsize=(30, 10))
            
            for j in range(3):
                fig_idx = 1+j
                plt.subplot(3, 1, fig_idx)
                plt.plot(real_sample[0][j])
            plt.savefig(sta_code_path + "/real_sample/3.png")
            plt.close()

            # plt.figure(figsize=(30, 10))

            # for i in range(4):
            #     for j in range(3):
            #         fig_idx = 1+i+4*j
            #         plt.subplot(3, 4, fig_idx)
            #         plt.plot(iffted_data[i][j])
            # plt.savefig(sta_code_path + "/fake_sample/iffted_3_4.png")
            # plt.close()

            # plt.figure(figsize=(30, 10))
            
            # for j in range(3):
            #     fig_idx = 1+j
            #     plt.subplot(3, 1, fig_idx)
            #     plt.plot(iffted_data[3][j])
            # plt.savefig(sta_code_path + "/fake_sample/iffted_3.png")
            # plt.close()

            # data = fake_sample

            # iffted_data = np.zeros(data.shape)

            # for i in range(len(data)):

            #     for j in range(3):

            #         waveform = data[i][j]

            #         LOG_EPS = 1e-40
            #         real_fft = np.absolute(fft(waveform))
            #         real_power_spec = real_fft ** 2
            #         real_psd = 10*np.log10(real_power_spec + LOG_EPS)

            #         a = np.random.rand(6000)
            #         b = np.sqrt(1 - np.power(a,2))
            #         x = np.sqrt(np.power(10,real_psd/10))
            #         k = np.concatenate((np.fft.ifft(a*x + b*1j*x)[200:-200], np.fft.ifft(a*x + b*1j*x)[200:600]))
            #         iffted_data[i][j] = k

            # plt.figure(figsize=(30, 10))

            # for i in range(4):
            #     for j in range(3):
            #         fig_idx = 1+i+4*j
            #         plt.subplot(3, 4, fig_idx)
            #         plt.plot(iffted_data[i][j])
            # plt.savefig(sta_code_path + "/fake_sample/iffted_concat_3_4.png")
            # plt.close()

            # plt.figure(figsize=(30, 10))
            
            # for j in range(3):
            #     fig_idx = 1+j
            #     plt.subplot(3, 1, fig_idx)
            #     plt.plot(iffted_data[1][j])
            # plt.savefig(sta_code_path + "/fake_sample/iffted_concat_3.png")
            # plt.close()

    def generate_h5py(self, model_path: str):
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        for sta_code in self.sta_codes:

            self.G = BBGenerator(self.z_size).to(self.device)
            self.optimizer_g = torch.optim.Adam(params = self.G.parameters(), lr = self.train_config.learning_rate, \
                                                betas=self.train_config.betas)

            data = sbd.WaveformDataset("data/" + sta_code)
            MIN_LOG_PGA, MAX_LOG_PGA = get_min_max_log_pga(data)
            self.min_log_pga = MIN_LOG_PGA
            self.max_log_pga = MAX_LOG_PGA

            df = pd.DataFrame()
            meta = data.metadata
                        
            batch_size = 1
            dataloader, train_conditional_names, generate_conditional_names = \
                sbd_to_dataloader(data, self.design, batch_size)

            self.G.eval()            
            self.G.load_state_dict(torch.load(Path(model_path) / "generator.pt"))

            self.fake_dir = sta_code

            os.makedirs(self.fake_dir, exist_ok=True)
        
            pbar = tqdm.tqdm(desc=f"Processing {sta_code}-HH", total=len(dataloader))

            fake_waveforms = []
        
            with h5py.File( self.fake_dir  +  f"/raw_waveforms_{sta_code}_HH.hdf5", "w") as ofh:
                grp = ofh.require_group(name="data")
            
                with torch.no_grad():
                    for batch_id, batch in enumerate(dataloader):

                        vs = [batch[v_name].to(dtype=torch.float, device=self.device) \
                              for v_name in train_conditional_names]
                        
                        batch_z =  torch.tensor(noise(batch_size, dim=100)).to(device=self.device)

                        fake_wfs, fake_ln_cn  = self.G(batch_z, self.design, *vs)
                        fake_wfs   = torch.squeeze(fake_wfs, dim=-1)
                        fake_ln_cn = fake_ln_cn.detach().cpu().numpy()
                                                                                        
                        fake_restored_signal = 10**self.recover_ln_cn(torch.tensor(fake_ln_cn))*fake_wfs.detach().cpu().numpy()
                        
                        network = meta["station_network_code"].unique()[0]
                        ## we are not conditioning the time variable for now.
                        vs_gen = [batch[v_name].to(dtype=torch.float, device=self.device) for v_name in generate_conditional_names]
                        
                        yyyy,mm,dd,index = vs_gen
                        yyyy=int(yyyy.item())
                        mm=int(mm.item())
                        dd=int(dd.item())
                        index=int(index.item())
                        info=batch_id
                        
                        'KS.CHC2.HH.KS201610:03d11:03d_61998_NO'
                        hkey = f'{network}.{sta_code}.HH.{network}{yyyy:04d}{mm:02d}{dd:02d}_{index}_{info}_NO'

                        fake_waveforms = np.squeeze(fake_restored_signal)
                        fake_waveforms = fake_waveforms.T
                        grp.create_dataset(name=hkey, data=fake_waveforms)
                        
                        pbar.update()


            waveform_name = f'raw_waveforms_{sta_code}_HH.hdf5'
            with h5py.File(self.fake_dir + "/" + waveform_name, 'r') as f:

                raw_data = f["data"]
                keys = raw_data.keys()

                split = np.random.choice(a=['train', 'dev', 'test'], size=len(keys) ,replace=True, p=[0.7, 0.2, 0.1])
                df["split"] = split

                def process(trace_name:str):
                    return trace_name.split('.')
                
                def index_to_utc(year:int, julday:int, index:int):

                    index = index // 100
                    hour = index // 3600
                    minute  = (index % 3600) // 60
                    second = index - (3600*hour + 60*minute)

                    return UTCDateTime(year=year, julday=julday, hour=hour, minute=minute, second=second)

                mp_pbar=tqdm.tqdm(desc=f'processing {sta_code} || HH', total=len(keys))

                trace_arrival_times = []

                for key in keys:
                    x = key.split('.')
                    trace_name = x[-1]
                    index = int(trace_name.split('_')[1])
                    year  = int(trace_name[2:6])
                    month = int(trace_name[6:8])
                    day   = int(trace_name[8:10])
                    julday = UTCDateTime(year=year, month=month, day=day, hour=0, minute=0).julday

                    trace_arrival_time = index_to_utc(year=year, julday=julday, index=index)

                    trace_arrival_times.append(trace_arrival_time)

                df["trace_start_time"] = trace_arrival_times
                df["trace_name"]       = keys

                df["receiver_network"] = 'KS'
                df["receiver_type"] = "HH"
                df["receiver_code"] = sta_code

                df["receiver_elevation_m"] = self.REF[sta_code]["HH"]["station_elevation_m"]
                df["receiver_latitude"]  = self.REF[sta_code]["HH"]["station_latitude_deg"]
                df["receiver_longitude"] = self.REF[sta_code]["HH"]["station_lingitude_deg"]

                df.to_csv(self.fake_dir +  f"/raw_metadata_{sta_code}_HH.csv")
                
            # pass

            src_path = tgt_path = self.fake_dir
            metadata_path = src_path +  f"/raw_metadata_{sta_code}_HH.csv"
            waveforms_path = src_path +  f"/raw_waveforms_{sta_code}_HH.hdf5"
            # seisbench formatted data output path


            # seis_base_path = Path("/app/src/dataset_seis/")
            seis_meta_path = tgt_path + f"/metadata_{sta_code}_HH.csv"
            seis_wave_path = tgt_path + f"/waveforms_{sta_code}_HH.hdf5"

            # read in csv metadata and filter only p,s,mag rows
            df = pd.read_csv(metadata_path)
            # df = df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1)
            df_psmag=df
            with sbd.WaveformDataWriter(seis_meta_path, seis_wave_path) as ofh, h5py.File(waveforms_path, 'r') as ifh:
                # once closed, no way to access?

                ofh.data_format = {
                    # 6000, 3 raw data > should be T'ed (Waveform Channel format is default in pytorch)
                    "dimension_order": "CW",
                    # TODO: verify this. I am guessing it as ZNE based on https://github.com/AIML-K/earthquake-seisbench/issues/23
                    "component_order": "ENZ",
                    "measurement": "velocity/acceleration",
                    "sampling_rate": 100,   # TODO: verify
                    "unit": "unknown",        # TODO: verify
                    "instrument_response": "unknown"
                }

                # reads in KR2 data
                data_raw = ifh['data']  # returns a group (775563 members)>

                for row in df_psmag.itertuples():
                    network, station, channel, event_name = row.trace_name.split('.')

                    trace_param = {
                        'trace_name': row.trace_name,
                        'trace_channel': channel,
                        'trace_sampling_rate_hz': ofh.data_format['sampling_rate'],
                        'trace_component_order': ofh.data_format['component_order'],
                        'trace_start_time' : row.trace_start_time
                    } ## added this line

                    station_param = {
                        'station_code': station,
                        'station_network_code': network,
                        'station_latitude_deg': row.receiver_latitude,
                        'station_longitude_deg': row.receiver_longitude,
                        'station_elevation_m'  : row.receiver_elevation_m
                    }

                    event_param = {'split': row.split}

                    # note, originally 6000,3 format, now 3x6000 format
                    waveform_cw = data_raw[trace_param['trace_name']][:].T

                    params = {**trace_param, **station_param, **event_param}

                    ofh.add_trace(params, waveform_cw)

