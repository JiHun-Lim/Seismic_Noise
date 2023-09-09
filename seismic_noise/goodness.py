import torch
import torch.nn as nn
from torch.fft import rfft
from ppsd import PPSD_Helper

def normalize_waveform(wfs:torch.Tensor):
    '''
        expects (B, 3, 6000) shaped tensor.
        outputs same-shaped tensor in [-1,1] scale
    '''
    DIV_EPS = 1e-20
    wnorm = torch.max(torch.abs(wfs), dim=-1)[0] + DIV_EPS
    wnorm = torch.unsqueeze(wnorm, dim=-1)
    wfs   = wfs / wnorm
    
    return wfs

def weighted_goodness_psd(sta_code:str, cpnt_order:str, real_wfs:torch.Tensor, fake_wfs:torch.Tensor, metric_type:str='JS', loss=nn.SmoothL1Loss()):
    LOG_EPS = 1e-40
    batch_size = real_wfs.shape[0]
        
    real_fft = rfft(normalize_waveform(real_wfs)).absolute()[:, :, 1:]
    fake_fft = rfft(normalize_waveform(fake_wfs)).absolute()[:, :, 1:]
    
    real_power_spec = real_fft ** 2
    fake_power_spec = fake_fft ** 2
    
    real_psd = 10*torch.log10(real_power_spec + LOG_EPS)
    fake_psd = 10*torch.log10(fake_power_spec + LOG_EPS)
            
    period_left_bin = torch.ones(92)

    for i in range(len(period_left_bin)):
        period_left_bin[i] = 0.02 * 2 ** (i/8 - 1/2)

    period_right_bin = 2 * period_left_bin
    # period_center_bin = np.sqrt(2)*period_left_bin

    freq_left_bin = 1 / period_left_bin.flip(0)
    freq_right_bin = 1 / period_right_bin.flip(0)

    freq_in_loss = torch.linspace(0, 50, 3001)[1:]

    real_psd_92 = torch.ones(batch_size,3,92)
    fake_psd_92 = torch.ones(batch_size,3,92)

    for k in range(92):

        mask = (freq_right_bin[k] < freq_in_loss) & (freq_in_loss <= freq_left_bin[k])
        real_psd_92[:,:,k] = real_psd[:,:,mask].mean()
        fake_psd_92[:,:,k] = fake_psd[:,:,mask].mean()

    real_wfs_numpy = real_wfs.detach().cpu().numpy()
    fake_wfs_numpy = fake_wfs.detach().cpu().numpy()

    histogram_helper = PPSD_Helper(
            sta_code=sta_code, 
            sta_type= "HH", 
            real_wfs=real_wfs_numpy, 
            fake_wfs=fake_wfs_numpy, 
            cpnt_order='ZNE',
            mode='train'
        )

    difference_map = torch.tensor(histogram_helper.get_difference_map(metric_type)).flip(-1).unsqueeze(0)

    return loss( difference_map * real_psd_92, difference_map * fake_psd_92 )

def goodness_fft(real_wfs:torch.Tensor, fake_wfs:torch.Tensor, loss=nn.MSELoss()):
    real_wfs = normalize_waveform(real_wfs)
    fake_wfs = normalize_waveform(fake_wfs)
    
    real_fft = rfft(real_wfs, ).absolute()
    fake_fft = rfft(fake_wfs).absolute()
    return loss(real_fft, fake_fft)
