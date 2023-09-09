import torch
import numpy as np
import matplotlib.pyplot as plt
from obspy.signal import PPSD
from obspy.imaging.cm import pqlx
from pathlib import Path
from obspy import read_inventory, UTCDateTime, Trace, Stream
from obspy.signal.util import _npts2nfft
from obspy.signal.invsim import cosine_taper, invert_spectrum
from scipy.spatial.distance import jensenshannon as js
from scipy.special import kl_div

def get_invs(sta_code:str, sta_type:str, cpnt_order:str)->dict:
    invs = {}

    for channel in cpnt_order:
    
        inv_pth = f'sample_responses/{sta_code}/RESP.KS.{sta_code}..{sta_type}{channel}'
        inv = read_inventory(inv_pth, format='RESP')
        invs[channel] = inv
    return invs

def restore_response(rr_waveform:Trace, inventory,water_level=0, output:str="VEL"):
    response = rr_waveform._get_response(inventory)
    npts = len(rr_waveform.data)
    nfft = _npts2nfft(npts)
    data_freq=np.fft.rfft(rr_waveform.data, n=nfft)
    freq_response, freqs = \
    response.get_evalresp_response(rr_waveform.stats.delta, nfft,output=output)
    invert_spectrum(freq_response,water_level)
    freq_response[0]=1
    data_freq/=freq_response
    ret = np.fft.irfft(data_freq)[0:npts]
    taper=cosine_taper(npts, 0.05,sactaper=True, halfcosine=False)
    taper=[x if x>1e-8 else 1 for x in taper]
    ret/=taper
    return ret


def batch_to_st(sta_code:str, sta_type:str, invs:dict, wfs:torch.Tensor, cpnt_order:str):
    '''
       TODO : start-time adjustment
    '''
    def type_to_output(sta_type:str):
        if sta_type in ['HH', 'EL']:
            return 'VEL'
        elif sta_type == 'HG':
            return 'ACC'
        else:
            raise NotImplementedError
    output = type_to_output(sta_type=sta_type)
    
    stream_dict = {}
        
    for cpnt in cpnt_order:
        idx_channel = cpnt_order.index(cpnt_order)
        traces = []
        start_time = UTCDateTime(year=2017, julday=1, hour=0, minute=0)
        inv = invs[cpnt]
        for idx_wfs in range(wfs.shape[0]):
    
            tr = Trace(data=wfs[idx_wfs][idx_channel])
            tr.stats.sampling_rate = 100
            tr.stats.channel = sta_type + cpnt
            tr.stats.network = 'KS'
            tr.stats.station = sta_code
            tr.stats.starttime = start_time
            # tr.data = restore_response(tr, inv, output=output)
            start_time = start_time + 120
            
            traces.append(tr)
        st = Stream(traces=traces)
        stream_dict[cpnt] = st
    return stream_dict

class PPSD_Helper(object):
    '''
        An integrated ppsd-related object which offers the following features.
        - ppsd plotting every epoch
        - ppsd loss for training
        - ppsd metric score used in the dev-loop ( for the purpose of best-model tracking)
        
        When initializing, please input the pth / epoch when you're using the dev-mode
    '''
    def __init__(self, sta_code:str, sta_type:str, real_wfs:np.ndarray, fake_wfs:np.ndarray, cpnt_order:str, mode:str, ppsd_pth:Path=None, epoch:int=None):
        '''
            depending on mode, they work in a different way.
            But in common, the I am trying to make the call of PPSD object as little as possible casue it costs so much of a time.
        '''
        self.invs = get_invs(sta_code, sta_type, cpnt_order)
        self.sta_type = sta_type
        self.sta_code = sta_code
        self.cpnt_order = cpnt_order
        if mode == 'train':
            self._get_histograms(sta_code, sta_type, real_wfs, fake_wfs, cpnt_order)
        elif mode == 'dev':
            self._plot_and_histogram(real_wfs, fake_wfs, ppsd_pth, epoch)
        
        self.batch_size = real_wfs.shape[0]
    
    def _get_histograms(self, sta_code:str, sta_type:str, real_wfs:np.ndarray, fake_wfs:np.ndarray, cpnt_order:str)->None:
        '''
            calculating histogram takes so much of the time :') 
            it'd be better to store such information as a attribute of a class.
        '''
        real_histograms, fake_histograms = [], []
        
        real_st_dict = batch_to_st(sta_code, sta_type, self.invs, real_wfs, cpnt_order)
        fake_st_dict = batch_to_st(sta_code, sta_type, self.invs, fake_wfs, cpnt_order)

        for channel in cpnt_order:
        
            inv = self.invs[channel]
            
            size = real_wfs.shape[0]
            indices = list(range(size))
            np.random.shuffle(indices)
        
            real_st = real_st_dict[channel]
            fake_st = fake_st_dict[channel]

            real_ppsd = PPSD(real_st.traces[0].stats, metadata=inv, skip_on_gaps=True, ppsd_length=60, period_limits=[0.02, 100])
            real_ppsd.add(real_st)

            fake_ppsd = PPSD(fake_st.traces[0].stats, metadata=inv, skip_on_gaps=True, ppsd_length=60, period_limits=[0.02, 100])
            fake_ppsd.add(fake_st)
            
            real_histogram = real_ppsd.current_histogram
            fake_histogram = fake_ppsd.current_histogram
            
            real_histograms.append(real_histogram)
            fake_histograms.append(fake_histogram)
        
        self.real_histograms = np.stack(real_histograms)
        self.fake_histograms = np.stack(fake_histograms)
        
        self.period_bin_centers = real_ppsd.period_bin_centers
        self.period_bin_left_edges = real_ppsd.period_bin_left_edges
        self.period_bin_right_edges = real_ppsd.period_bin_right_edges
    
    def _plot_and_histogram(self, real_wfs:torch.Tensor, fake_wfs:torch.Tensor, ppsd_pth:Path, epoch:int):
        '''
            will be used inside of dev-loop
        '''
        invs  = self.invs
        cpnt_order = self.cpnt_order
        real_st_dict = batch_to_st(self.sta_code, self.sta_type, invs, real_wfs, cpnt_order)
        fake_st_dict = batch_to_st(self.sta_code, self.sta_type, invs, fake_wfs, cpnt_order)
        
        real_histograms = []
        fake_histograms = []

        for channel in cpnt_order:
        
            real_file_name = ppsd_pth / f'{channel}_real_epoch{epoch}.png'
            fake_file_name = ppsd_pth / f'{channel}_fake_epoch{epoch}.png'
            
            inv = invs[channel]
            
            
            size = real_wfs.shape[0]
            indices = list(range(size))
            np.random.shuffle(indices)
        
            real_st = real_st_dict[channel]
            fake_st = fake_st_dict[channel]

            real_ppsd = PPSD(real_st.traces[0].stats, metadata=inv, skip_on_gaps=True, ppsd_length=60, period_limits=[0.02, 100])
            real_ppsd.add(real_st)

            fake_ppsd = PPSD(fake_st.traces[0].stats, metadata=inv, skip_on_gaps=True, ppsd_length=60, period_limits=[0.02, 100])
            fake_ppsd.add(fake_st)
            real_ppsd.plot(real_file_name, cmap=pqlx)
            plt.close()
            fake_ppsd.plot(fake_file_name, cmap=pqlx)
            plt.close()
            
            real_histogram = real_ppsd.current_histogram
            fake_histogram = fake_ppsd.current_histogram
            
            real_histograms.append(real_histogram)
            fake_histograms.append(fake_histogram)
        
        self.real_histograms = np.stack(real_histograms)
        self.fake_histograms = np.stack(fake_histograms)
        
        self.period_bin_centers = real_ppsd.period_bin_centers
        self.period_bin_left_edges = real_ppsd.period_bin_left_edges
        self.period_bin_right_edges = real_ppsd.period_bin_right_edges
    
    def get_difference_map(self, metric_type:str):
        '''
            (n_period_bins, n_power_bins) -> (n_period_bins, )
        '''

        if metric_type == 'JS':
            ret = []
            for i in range(3):
                ret.append( js(self.real_histograms[i], self.fake_histograms[i], axis=1)[::-1] ) # need a inversion (period->freq)
            ret = np.stack(ret)
            return ret
        
        elif metric_type == 'L1':
            ret = []
            for i in range(3):
                ret.append( np.sum(np.abs(self.real_histograms[i].astype(np.int64) - self.fake_histograms[i].astype(np.int64)), axis=1) / self.batch_size)
            ret = np.stack(ret)
            return ret

        elif metric_type == 'KL':
            ret = []
            for i in range(3):
                bit = []
                for j in range(92):
                    kl = kl_div(self.real_histograms[i][j], self.fake_histograms[i][j] + 1e-3)
                    kl[np.isinf(kl)] = 0
                    bit.append(np.sum(kl))
                ret.append(bit)   
            ret = np.stack(ret)
            return ret

        elif metric_type == 'L1_KL':
            ret = []
            for i in range(3):
                bit = []
                for j in range(92):

                    if j <30:
                         
                        kl = kl_div(self.real_histograms[i][j], self.fake_histograms[i][j] + 1e-3)
                        bit.append(np.sum(kl))

                    else:

                        l1 = np.abs(self.real_histograms[i][j] - self.fake_histograms[i])
                        bit.append(np.sum(l1))
                         

                ret.append(bit)   
            ret = np.stack(ret)
            return ret

        elif metric_type == 'kl':
            ret = []
            for i in range(3):
                bit = []
                for j in range(92):
                    sigma_real = np.std(self.real_histograms[i][j])
                    sigma_fake = np.std(self.fake_histograms[i][j])
                    mu_real    = np.mean(self.real_histograms[i][j])
                    mu_fake    = np.mean(self.fake_histograms[i][j])
                    kl = np.log10(sigma_real / sigma_fake) + ( sigma_fake**2 + (mu_fake - mu_real)**2 ) / (2 * sigma_real**2) - 0.5
                    bit.append( kl)
                ret.append(bit)
            ret = np.stack(ret)
            return ret

        else:
            raise NotImplementedError

    def get_ppsd_score(self, metric_type:str):

        difference_map = self.get_difference_map(metric_type=metric_type)
        
        # TODO : channelwise tracking is possible theoretically. But do we really need this info?
        # TODO : For each frequency component allot a weight according to the situation -- weight 
        
        difference_score = - np.sum(difference_map) # the sign should be reversed so that the higher the better concept still works.
        
        return difference_score
    
    def plot_whole_score(self, metric_type:str, imap_type:str, d_map_pth:Path, epoch:int):
        # whole_score_freq = self.get_weight(metric_type=metric_type, imap_type=imap_type)[0]
        whole_score_freq = self.get_difference_map(metric_type=metric_type)
        freq = np.arange(0,92)
        # freq = np.linspace(0, 50, 3001)[1:]
        plt.figure(figsize=(30, 10))
        
        filepth = d_map_pth / f'{metric_type}-{epoch}.png'
        
        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.plot(freq, whole_score_freq[i])
        plt.savefig(fname=filepth)
        plt.close()
        
if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    import torch
    from pathlib import Path

    import seisbench
    import seisbench.data as sbd
    
    start = time.time()
    
    sta_code = 'CHC2'
    sta_type = 'HH'
    cpnt_order = 'ZNE'
    
    meta = sbd.WaveformDataset(seisbench.cache_root / 'datasets' / 'sample_noise' / 'processed_real' / sta_code)
    mask = meta['trace_channel'] == sta_type
    real_set = sbd.WaveformDataset(seisbench.cache_root / 'datasets' / 'sample_noise' / 'processed_real' / sta_code).filter(mask=mask, inplace=False)
    fake_set = sbd.WaveformDataset(seisbench.cache_root / 'datasets' / 'sample_noise' / 'fake' / 'bbgan' / 'unconditional' / 'design4' / sta_code / sta_type)
    
    real_wfs = real_set.get_waveforms(idx=list(range(1000)))
    fake_wfs = fake_set.get_waveforms(idx=list(range(1000)))
    
    histogram_helper = PPSD_Helper(
        sta_code=sta_code,
        sta_type=sta_type,
        real_wfs=real_wfs,
        fake_wfs=fake_wfs,
        cpnt_order=real_set.component_order,
        mode='train'
    )
    
    difference_map = histogram_helper.get_difference_map('JS')
    ppsd_score = histogram_helper.get_ppsd_score()
    
    print(time.time()-start)    