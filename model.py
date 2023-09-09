import torch
import torch.nn as nn
import torch.nn.functional as F

def embed(in_chan, out_chan):

    layers = nn.Sequential(
        nn.Linear(in_chan, 32), torch.nn.GELU(),
        nn.Linear(32, 64), torch.nn.GELU(),
        nn.Linear(64, 256), torch.nn.GELU(),
        nn.Linear(256, 512), torch.nn.GELU(),
        nn.Linear(512, 1024), torch.nn.GELU(),
        nn.Linear(1024, 2048), torch.nn.GELU(),
        nn.Linear(2048, out_chan), torch.nn.GELU()
    )

    return layers

def FCNC(n_vs=150, hidden_1=256, hidden_2=512):

    layers = nn.Sequential(
        nn.Linear(n_vs, hidden_1), torch.nn.GELU(),
        nn.Linear(hidden_1, hidden_2), torch.nn.GELU(),
        nn.Linear(hidden_2, hidden_2), torch.nn.GELU(),
        nn.Linear(hidden_2, hidden_1), torch.nn.GELU(),
        nn.Linear(hidden_1, 3),
        torch.nn.Tanh()
    )

    return layers

class BBGenerator(nn.Module):

    def __init__(self, z_size):
        super(BBGenerator, self).__init__()

        self.preprocessing = nn.Sequential(
            nn.Linear(z_size, 150, bias = False), nn.BatchNorm1d(3)
        )
        self.embed_time = embed(2,150)
        self.model_step1 = nn.Sequential(
            nn.Conv2d(4, 6, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), ), nn.BatchNorm2d(6), nn.GELU(),
            nn.Conv2d(6, 6, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), ), nn.BatchNorm2d(6), nn.GELU(),
            nn.Conv2d(6, 3, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), ), nn.BatchNorm2d(3), nn.GELU()
        )
        self.model_step2 = nn.Sequential(
            nn.Linear(150, 300, bias = False),  nn.BatchNorm1d(3), nn.GELU()
        )
        self.model_step3 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1), mode='nearest', ), 
            nn.Conv2d(6, 16, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), ), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), ), nn.BatchNorm2d(16), nn.GELU(),
            nn.Upsample(scale_factor=(2, 1), mode='nearest', ),
            nn.Conv2d(16, 32, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), ), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), ), nn.BatchNorm2d(32), nn.GELU(),
            nn.Upsample(scale_factor=(2, 1), mode='nearest', ),
            nn.Conv2d(32, 64, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), ), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), ), nn.BatchNorm2d(64), nn.GELU(),
            nn.Upsample(scale_factor=(2, 1), mode='nearest', ),
            nn.Conv2d(64, 128, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), ), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), ), nn.BatchNorm2d(128), nn.GELU(),
            nn.Upsample(scale_factor=(3, 1), mode='nearest', ),
            nn.Conv2d(128, 64, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), ), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), ), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), ), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), ), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), ), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), ), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 3, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), ),
            nn.Tanh()
        )
        self.model_lncn = FCNC(n_vs=150, hidden_1=256, hidden_2=512)

    def forward(self, z:torch.Tensor, design:str, v_time:torch.Tensor):

        x = torch.unsqueeze(self.preprocessing(z), 3)
        if design == "time":
            v_time =  self.embed_time(v_time)
            v_time  = v_time.reshape(-1, 1, 150, 1)
            x = torch.cat([x, v_time], 1)
        x_step1 = self.model_step1(x)
        x_step1 = torch.squeeze(x_step1, 3)
        x_step2 = self.model_step2(x_step1)
        x_step3 = x_step2[:, :, :250].reshape(-1, 6, 125, 1)
        xcn = x_step2[:, :, 250:].reshape(-1, 150)
        x_out = self.model_step3(x_step3)
        xcn_out = self.model_lncn(xcn)

        return (x_out, xcn_out)

class BBDiscriminator(nn.Module):

    def __init__(self):
        super(BBDiscriminator, self).__init__()

        self.embed_time  = embed(2, 6000)
        self.embed_type = embed(3, 6000)
        self.nn_cnorm = embed(3, 6000)

        self.embed_lncn  = nn.Sequential(
            nn.Linear(3, 1000), nn.BatchNorm1d(1000), nn.GELU()
            )
        self.model_step1 = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=(32, 1), stride=(2, 1), padding=(15, 0), ), nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=(31, 1), stride=(1, 1), padding=(15, 0), ), nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(32, 1), stride=(2, 1), padding=(15, 0), ), nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(31, 1), stride=(1, 1), padding=(15, 0), ), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(32, 1), stride=(2, 1), padding=(15, 0), ), nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=(31, 1), stride=(1, 1), padding=(15, 0), ), nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(32, 1), stride=(2, 1), padding=(15, 0), ), nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=(31, 1), stride=(1, 1), padding=(15, 0), ), nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=(32, 1), stride=(2, 1), padding=(15, 0), ), nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=(31, 1), stride=(1, 1), padding=(15, 0), ), nn.GELU()
        )
        self.model_step2 = nn.Sequential(
            nn.Linear(187, 110), nn.GELU(), 
            nn.Linear(110, 128), nn.GELU(),
            nn.Linear(128, 100), nn.GELU()
        )
        self.model_step3 = nn.Linear(256 * 100, 1)

    def forward(self, x:torch.Tensor, ln_cn:torch.Tensor, design:str, v_time:torch.Tensor):

        ln_cn = self.nn_cnorm(ln_cn).reshape(-1,1,6000,1)

        if design == "time":
            v_time =  self.embed_time(v_time).reshape(-1, 1, 6000, 1)
            x = torch.cat([x, ln_cn, v_time], dim=1)

        x_step1 = torch.squeeze(self.model_step1(x), 3)
        x_step2 = self.model_step2(x_step1).reshape(-1, 25600)
        out = self.model_step3(x_step2)
        return out
