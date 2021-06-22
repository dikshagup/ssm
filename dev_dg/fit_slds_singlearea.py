import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

import ssm
import sys
sys.path.append(r"/Users/dikshagupta/python_transfer/pbups_phys/")
from phys_helpers import *
from physdata_preprocessing import *

from scipy.ndimage import gaussian_filter1d


    
ratnames = ["X046", "X059", "X062", "X085"]
regions = ["M2", "DMS"]

for ratname in ratnames:
    
    p = dict()
    p['ratname'] = ratname
    
    p['fr_thresh'] = 1.
    p['align_to'] = 'cpoke_in'
    p['post_mask'] = None
    p['window'] = [0, 1500]
    p['split_by'] = None
    p['filter_type'] = None
    p['filter_w'] = None
    p['binsize'] = 25
    p['nfolds'] = 10
    
    files = get_sortCells_for_rat(p['ratname'])
    for f in range(len(files)):
        p['filename'] = f
        
        for reg in regions:
            p['region'] = region
                
                for K in range(4):
                    p['K'] = K   # number of discrete states
                    
                    for D in range(8):
                        p['D'] = D   # number of latent dimensions
                        
                        fit_slds(p)
    
    
def format_spikes(FR, ntrials):
    datas = []
    for i in range(ntrials):
        spikes_cur = np.squeeze(FR[:,:,i])

        # remove columns which contain NaNs
        idx_keep = np.invert(np.any(np.isnan(spikes_cur), axis=0))
        spikes_cur = spikes_cur[:,idx_keep]

        # Transpose the data for passing to SSM fit function
        # To use the Poisson observation model, we must also
        # convert our arrays to be integer types.
        datas.append(np.asarray(spikes_cur.T, dtype=int))
    return datas



def fit_slds(p):
    
    df_trial, df_cell, _ = load_phys_data_from_Cell(p['filename'])
    y, inputs = load_data_for_slds(df_trial, df_cell, p)
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = p['nfolds'])
    train_scores = np.empty(p['nfolds'])
    test_scores = np.empty(p['nfolds'])
    
    for train, test in kf.split(range(len(df_trial))):
        
        y_train = [y[train_id] for train_id in train]
        inputs_train = [inputs[train_id] for train_id in train]
     
        print("Fitting SLDS with Laplace-EM")
        slds = ssm.SLDS(len(df_cell), p['K'], p['D'], M = inputs[0].shape[1], 
                        emissions="poisson_orthog", 
                        emission_kwargs={"bin_size":p['binsize']/1000, "link":"log"})
        slds.initialize(y_train, inputs = inputs_train)
        q_elbos, q = slds.fit(y_train, inputs = inputs_train, method="laplace_em",
                                      variational_posterior="structured_meanfield",
                                      num_iters=50, 
                                      initialize=True,
                                      num_init_restarts = 5,
                                      alpha=0.5)
        
        choice = split_trials(df_trial.loc[list(train)], 'pokedR')
        train_scores[r] = compute_fit_score(q, y_train, inputs_train, choice, p)

        y_test = [y[test_id] for test_id in test]
        inputs_test = [inputs[test_id] for test_id in test]
        choice = split_trials(df_trial.loc[list(test)], 'pokedR')
        test_scores[r] = compute_fit_score(q, y_test, inputs_test, choice, p)

        

def compute_fit_score(q, ytrue, inp, p):

    ypred = []
    for tr in range(len(ytrue)):
        q_x = q.mean_continuous_states[tr]
        ypred.append(slds.smooth(q_x, ytrue[tr], input = inp[tr]))
        
        
    
    # get number of neurons
    assert len(true_psths) == len(sim_psths)
    N = len(true_psths)

    r2 = np.zeros(N)

    for i in range(N):

        true_psth = true_psths[i]
        sim_psth = sim_psths[i]
        true_psth_mean = [true_psth[coh] for coh in range(len(true_psth)) if true_psth[coh].shape[0] > 0]
        mean_PSTH = np.nanmean(np.vstack(true_psth_mean))

        r2_num = 0.0
        r2_den = 0.0

        # number of coherences, loop over
        NC = len(true_psth)
        for j in range(NC):
            T = true_psth[j].shape[0]
            if T > 0:
                r2_num += np.nansum( (true_psth[j] - sim_psth[j][:T])**2)
                r2_den += np.nansum( (mean_PSTH - true_psth[j])**2)

        r2[i] = 1 - r2_num / r2_den

    return r2

    
    



    
    
    
    
    
    
def load_data_for_slds(df_trial, df_cell, p):
    
    ntrials = len(df_trial)

    # process df_cell
    df_cell = df_cell[df_cell['stim_fr'] >= p['fr_thresh']].reset_index()
    df_cell = df_cell[df_cell['region'].str.match(p['region'])].reset_index(drop = True)


    # get rasters for each neuron and concatenate
    y = []
    for cellnum in range(len(df_cell)):
        PSTH = make_psth(df_cell.loc[cellnum, 'spiketime_s'], 
                            df_trial,
                            split_by = p['split_by'],
                            align_to = p['align_to'], 
                            post_mask = p['post_mask'],
                            window = p['window'], 
                            filter_type = p['filter_type'],
                            filter_w = p['filter_w'],
                            binsize = p['binsize'],
                            plot = False)

        y.append(PSTH['trial_fr'][0].T)
    y = format_spikes(np.array(y), ntrials)
    
        # get click info:
    window = adjust_window(p['window'], p['binsize'])
    edges = np.arange(p['window'][0], p['window'][1] + p['binsize'], p['binsize'])*0.001 # convert to s 

    inputs = []
    for tr in range(ntrials):
        counts_L, _ = np.histogram(df_trial['leftBups'][tr] - df_trial['cpoke_in'][tr], edges)
        counts_R, _ = np.histogram(df_trial['rightBups'][tr] - df_trial['cpoke_in'][tr], edges)
        stimstart, _ = np.histogram(df_trial['stereo_click'][tr] - df_trial['cpoke_in'][tr], edges)
        inputs.append(np.asarray(np.vstack((counts_L, counts_R, stimstart)).T, dtype=int))
        
    return y, inputs

