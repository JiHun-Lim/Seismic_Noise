import numpy as np
import datetime as dt
import seisbench.data as sbd
import seisbench.generate as sbg
from torch.utils.data import DataLoader
from obspy import UTCDateTime

eps = 1e-16

def sbd_to_dataloader(ori_data:sbd.WaveformDataset, design:int, batch_size:int):
    
    data        = ori_data.train()
    generator   = sbg.GenericGenerator(data)
    generate_conditional_names = ['v_year', 'v_month', 'v_day', 'v_index']

    @generator.augmentation
    def generate_conditional(x_dict:dict):
        _, metadatum = x_dict['X']
        utc = UTCDateTime(metadatum['trace_start_time'])
        yyyy = int(utc.year)
        mm   = int(utc.month)
        dd   = int(utc.day)
        
        origin =  UTCDateTime(yyyy,mm,dd)
        index  = int(utc - origin)
        
        x_dict['v_year'] = np.array([[yyyy]]).astype(np.int32)
        x_dict['v_month']= np.array([[mm]]).astype(np.int32)
        x_dict['v_day']  = np.array([[dd]]).astype(np.int32)
        x_dict['v_index']= np.array([[index]]).astype(np.int32)

    if design == "time":
        MIN_LOG_PGA, MAX_LOG_PGA = get_min_max_log_pga(ori_data)
        @generator.augmentation
        def get_get_lncn(x_dict:dict):
            raw_wfs, _ = x_dict['X']
            ln_cn = np.max(np.abs(raw_wfs), axis=-1)
            ln_cn = np.log10(ln_cn + eps)
            
            ln_cn = (ln_cn - MIN_LOG_PGA)/(MAX_LOG_PGA-MIN_LOG_PGA + eps)
            ln_cn = 2*ln_cn - 1
            ln_cn = np.array([ln_cn])
            x_dict['ln_cn'] = ln_cn
        
        @generator.augmentation
        def conditional(x_dict:dict):
            
            def type_to_onehot(type:str):
                TYPE_TO_INDEX = {
                    'HH' : 0,
                    'HG' : 1,
                    'EL' : 2
                }
                index = TYPE_TO_INDEX[type]
                
                return np.eye(len(TYPE_TO_INDEX.keys()))[index].reshape(1, -1)
            
            _, metadatum = x_dict['X']
            sta_time  = metadatum['trace_start_time']
            sta_type = metadatum['trace_channel']
            
            sta_time_org = dt.datetime.strptime(sta_time, '%Y-%m-%dT%H:%M:%S.%fZ')
            sta_time_0 = dt.datetime.strptime(sta_time[:10], '%Y-%m-%d')
            diff = sta_time_org - sta_time_0

            hour_6period = sta_time_org.hour - (sta_time_org.hour % 6)
            month_4period = sta_time_org.month - (sta_time_org.month % 6)

            # time_cos = np.cos(2*np.pi*diff.seconds/86400)
            # time_sin = np.sin(2*np.pi*diff.seconds/86400)
            # time_cos = np.cos(2*np.pi*sta_time_org.hour/24)
            # time_sin = np.sin(2*np.pi*sta_time_org.hour/24)
            # time_cos = np.cos(2*np.pi*hour_6period/24)
            # time_sin = np.sin(2*np.pi*hour_6period/24)
            time_cos = np.cos(2*np.pi*sta_time_org.month/12)
            time_sin = np.sin(2*np.pi*sta_time_org.month/12)
            # time_cos = np.cos(2*np.pi*month_4period/12)
            # time_sin = np.sin(2*np.pi*month_4period/12)
            
            v_time = np.array([[time_sin, time_cos]])
            # v_type = type_to_onehot(sta_type)
            
            x_dict['v_time'] = v_time
            # x_dict['v_type'] = v_type
            
            
        @generator.augmentation
        def original(x_dict:dict):
            
            DIVIDE_EPS = 1e-10
            
            waveform, metadatum = x_dict['X']
            mu = np.mean(waveform, axis=-1, keepdims=True)
            waveform = waveform - mu
            norm     = np.max(np.abs(waveform), axis=-1, keepdims=True) + DIVIDE_EPS
            waveform = waveform / norm
            
            x_dict['X'] = (waveform, metadatum)

        # train_conditional_names = ['v_time', 'v_type']
        train_conditional_names = ['v_time']
        
    return DataLoader(generator, batch_size=batch_size, shuffle=True), train_conditional_names, \
        generate_conditional_names

def get_min_max_log_pga(data:sbd.WaveformDataset):
    
    dataset_size = len(data)
    min_log_pga = 9999999
    max_log_pga = -9999999
    batch_size = 512
    
    for i in range(0, dataset_size, batch_size):
        if i+batch_size > dataset_size:
            wfs = data.get_waveforms(list(range(i, dataset_size)))
        else:
            wfs = data.get_waveforms(list(range(i, i+batch_size)))
        
        curr_c_norms = np.max(np.abs(wfs), axis=2)
        curr_vns_max = np.max(curr_c_norms, axis=1)
        curr_vns_min = np.min(curr_c_norms, axis=1)
        
        curr_vns_max = curr_vns_max[curr_vns_max != 0]
        curr_vns_min = curr_vns_min[curr_vns_min != 0]
        
        curr_pga_max = curr_vns_max.max()
        curr_pga_min = curr_vns_min.min()
        
        if curr_pga_min == 0:
            curr_log_pga_min = np.log10(curr_pga_min + eps)
        else:
            curr_log_pga_min = np.log10(curr_pga_min)
        curr_log_pga_max = np.log10(curr_pga_max + eps)
        
        min_log_pga = min(min_log_pga, curr_log_pga_min)
        max_log_pga = max(max_log_pga, curr_log_pga_max)
    return min_log_pga, max_log_pga

def noise(Nbatch, dim):
    # Generate noise from a uniform distribution
    m = 3
    return np.random.normal(size=[Nbatch, m, dim], scale = 2.0).astype(dtype=np.float32)
