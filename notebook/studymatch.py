# %%
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import numpy as np
import pandas as pd
import h5py

import pycbc.conversions, pycbc.distributions, pycbc.waveform, pycbc.filter, pycbc.types, pycbc.psd, pycbc.fft

from tqdm import tqdm
import multiprocessing

from pyseobnr.generate_waveform import GenerateWaveform as eob_td_wf
from matplotlib import pyplot as plt
    
import matplotlib.pyplot
matplotlib.use('Agg')  # or 'TkAgg'
import os

if not os.path.exists('figure'):
    os.makedirs('figure')

# %%
class GenWaveform(object):
    '''Waveform Generator
    '''
    def __init__(self, buffer_length, sample_rate, f_lower):
        self.f_lower = f_lower
        self.delta_f = 1.0 / buffer_length
        tlen = int(buffer_length * sample_rate) # buffer length x sample_rate
        self.flen = tlen // 2 + 1

        #psd is hard coded to O3 psd
        psd = pycbc.psd.read.from_txt('/work/yifanwang/ecc/templatebank/o3psd.txt', 
            self.flen, self.delta_f, self.f_lower, is_asd_file = False)
        
        self.kmin = int(f_lower * buffer_length)
        self.w = ((1.0 / psd[self.kmin:-1]) ** 0.5).astype(np.float32)
        
        qtilde = pycbc.types.zeros(tlen, np.complex64) # correlation in Fourier domain
        q = pycbc.types.zeros(tlen, np.complex64) # correlation
        self.qtilde_view = qtilde[self.kmin:self.flen - 1]
        self.ifft = pycbc.fft.IFFT(qtilde, q)
        
        # the maximum is around 0
        self.md = q._data[-100:]
        self.md2 = q._data[0:100] 

    def generate(self, **kwds):
        '''Return normalized hp
        '''
        if kwds['approximant'] in pycbc.waveform.fd_approximants():  
            hp, _ = pycbc.waveform.get_fd_waveform(delta_f = self.delta_f, **kwds)
        else:
            dt = 1.0 / self.sample_rate
            hp = pycbc.waveform.get_waveform_filter(
                        pycbc.types.zeros(self.flen, dtype=np.complex64),
                        delta_f=self.delta_f,
                        delta_t=dt,
                        f_lower=self.f_lower,
                        **kwds)
        
        hp.resize(self.flen)
        hp = hp.astype(np.complex64)
        
        hp[self.kmin:-1] *= self.w
        s = pycbc.filter.sigmasq(hp, low_frequency_cutoff=self.f_lower)
        hp /= s**0.5 
        
        hp.params = kwds
        hp.s = s

        return hp

    def match(self, hp, hc):
        hp.view = hp[self.kmin:-1]
        hc.view = hc[self.kmin:-1]
        pycbc.filter.correlate(hp.view, hc.view, self.qtilde_view)
        self.ifft.execute()
        m = max(abs(self.md).max(), abs(self.md2).max())
        return m * 4.0 * self.delta_f

    def overlap(self, hp, hc):
        o = hp.inner(hc)
        return o * 4.0 * self.delta_f
    
def wf_wrapper(p):
    index = p['index']
    try:
        hp = gen.generate(**p)
        return index, hp
    except Exception as e:
        print(e)
        return index, None

def match_wrapper(p):
    '''A wrapper function to compute match
    '''
    h1 =pycbc.types.FrequencySeries(initial_array=p['h1_data'], delta_f=p['h1_delta_f'])
    h2 =pycbc.types.FrequencySeries(initial_array=p['h2_data'], delta_f=p['h2_delta_f'])
    return p['bank_index'], gen.match(h1, h2)

def gen_injections():
    mass_lim = (5, 10)
    spin_lim = (-0.5, 0.5)
    ecc_lim = (0, 0.3)
    ano_lim = (0, 2*np.pi)

    uniform_prior = pycbc.distributions.Uniform(
                            mass1=mass_lim,
                            mass2=mass_lim,
                            spin1z=spin_lim,
                            spin2z=spin_lim,
                            eccentricity=ecc_lim,
                            rel_anomaly=ano_lim)

    def _q_lt_8(params):
        return pycbc.conversions.q_from_mass1_mass2(params["mass1"],params["mass2"]) < 8

    return pycbc.distributions.JointDistribution(["mass1",
                                "mass2",
                                "spin1z",
                                "spin2z",
                                "eccentricity",
                                "rel_anomaly"],
                                uniform_prior,
                                constraints=[_q_lt_8])

def tdwf_wrapper(p):
    p2 = {"approximant": "SEOBNRv5EHM",
          "ModeArray": [(2,2)],
          "f22_start": 20,
          "lmax_nyquist": 1
         }
    p.update(p2)
    index = p['index']
    try:
        wf = eob_td_wf(p)
        hp, _ = wf.generate_td_polarizations()
        return index, abs(float(hp.epoch))
    except Exception as e:
        print(e)
        return index, None


# %%
gen = GenWaveform(buffer_length = 32, sample_rate = 2048, f_lower = 20)

# %% [markdown]
# Read in template bank

# %%
with h5py.File('/work/yifanwang/ecc/src/searchtools/scripts/sucbankprodduration.hdf') as f:
        df_bank = pd.DataFrame(
           {'mass1': f['mass1'][:],
            'mass2': f['mass2'][:],
            'tau0': pycbc.conversions.tau0_from_mass1_mass2(f['mass1'][:],f['mass2'][:],15),
            'eccentricity': f['eccentricity'][:],
            'rel_anomaly': f['rel_anomaly'][:],
            'spin1z': f['spin1z'][:],
            'spin2z': f['spin2z'][:],
            'approximant': f['approximant'][:].astype('str'),
            'f_lower': f['f_lower'][:],
            'template_duration': f['template_duration'][:],}
        )
df_bank['index'] = df_bank.index

# %%
inj = gen_injections()

df_inj = pd.DataFrame(inj.rvs(20))

df_inj['tau0'] = pycbc.conversions.tau0_from_mass1_mass2(df_inj['mass1'],df_inj['mass2'],15)
df_inj['index'] = df_inj.index
df_inj['approximant'] = df_bank['approximant'][0]
df_inj['f_lower'] = df_bank['f_lower'][0]
df_inj['s'] = np.zeros(df_inj.shape[0])


inj_waveform = {}
parlist = ['index', 'approximant', 'f_lower', 'mass1', 'mass2', 'spin1z', 'spin2z', 'eccentricity', 'rel_anomaly']

with multiprocessing.Pool(128) as pool:
    for return_i, return_hp in pool.imap_unordered(
        wf_wrapper,
        ({k: df_inj.loc[idx,k] for k in parlist} for idx in tqdm(df_inj.index))
    ):
        inj_waveform[return_i] = return_hp
        try:
            df_inj.loc[return_i, 's'] = return_hp.s
        except AttributeError:
            df_inj.loc[return_i, 's'] = np.nan

df_inj.to_csv('figure/injections.csv', index=False)

# %%
def study_match(hpinj, hpinj_tau0, df_bank, tau0_tol=0.5, nproc=128):
    if hpinj == None:
        print("Failed waveform generation in injection")
        return None
    
    neighbor = df_bank[abs(df_bank['tau0'] - hpinj_tau0) < tau0_tol].index
    print("Number of FF jobs =", len(neighbor)) 

    all_fitting_factors = []
    parlist = ['index', 'approximant', 'f_lower', 'mass1', 'mass2', 'spin1z', 'spin2z', 'eccentricity', 'rel_anomaly']
    with multiprocessing.Pool(nproc) as pool:
        for return_i, return_hp in pool.imap_unordered(
            wf_wrapper,
            ({k: df_bank.loc[idx, k] for k in parlist} for idx in tqdm(neighbor))
        ):
            match = gen.match(hpinj, return_hp)
            dict_current = {'bank_row': return_i, 
                            'fittingfactor': match, 
                            'bank_s': return_hp.s,
                            }
            all_fitting_factors += [dict_current]
        
    return all_fitting_factors

# %%
for ii in range(20):
    hpinj = inj_waveform[ii]
    _, hpinj_duration = tdwf_wrapper({k: df_inj.loc[ii,k] for k in parlist})
    hpinj_tau0 = df_inj.loc[ii, 'tau0']
    if hpinj == None:
        print("Failed waveform generation in injections for #", ii)
        continue
    
    allff = study_match(hpinj, hpinj_tau0, df_bank, tau0_tol = 0.5, nproc = 128)
    df_ff = pd.DataFrame(allff)

    df_ff_sorted = df_ff.sort_values('fittingfactor', ascending=False)
    max_index = df_ff_sorted['fittingfactor'].idxmax()
    df_merged = df_ff_sorted.merge(df_bank, left_on='bank_row', right_index=True)
    df_merged.to_csv('figure/FF_'+str(ii)+'.csv',index=False)
    
    plt.figure()
    plt.scatter(df_merged['template_duration']-hpinj_duration, 1 - df_merged['fittingfactor'], 
            c=df_merged['tau0'] - hpinj_tau0, s=1)
    plt.colorbar(label='Template Tau0 - Injection Tau0')
    plt.axhline(0.05, color='r', linestyle='--')
    plt.ylabel('1 - Fitting Factor')
    plt.yscale('log')
    #plt.ylim(upp=0.9)
    plt.xlabel('Template Duration - Injection Duration')
    plt.title('Injection duration = '+str(hpinj_duration)+', Injection Tau0 = '+str(hpinj_tau0))
    plt.savefig('figure/FF_'+str(ii)+'.png',dpi=100)

    plt.figure()
    plt.scatter(df_merged['bank_s'] / hpinj.s, 1 - df_merged['fittingfactor'], 
            c=df_merged['template_duration']-hpinj_duration, s=1)
    plt.colorbar(label='Template Duration - Injection Duration')
    plt.axhline(0.05, color='r', linestyle='--')
    plt.ylabel('1 - Fitting Factor')
    plt.yscale('log')
    #plt.ylim(upp=0.9)
    plt.xlabel('Template sigma / Injection sigma')
    plt.title('Injection sigma = '+str(hpinj.s))
    plt.savefig('figure/FF_'+str(ii)+'_sigma.png',dpi=100)



