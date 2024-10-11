
"""
# © Sinan Rasiya Koya, 2024. All rights reserved.

This code is released under the MIT License. If you use this code, please cite the following journal article:

**Rasiya Koya, S., & Roy, T. (2024). Temporal Fusion Transformers for streamflow Prediction: Value of combining attention with recurrence. 
Journal of Hydrology, 637, 131301.**

For any queries or issues, feel free to reach out at ssinanrk2@huskers.unl.edu.
"""
#%%
import os
import glob
import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths
print("\nNumber of CPUs = "+str(os.cpu_count())+"\n")

from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, RMSE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.metrics import MultiHorizonMetric

import hydroeval

#%% GLOBALS
# Setting the seed
pl.seed_everything(1729)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

#%% FUNCTIONS

# The following class is a custom loss function/metric for Kling-Gupta Efficiency (KGE). This is not used in the current implementation. 
# However, it can be used as a reference for custom metrics.
class kgeloss(MultiHorizonMetric):
    def loss(self, simulations: torch.Tensor, evaluation: torch.Tensor)-> torch.Tensor:
        # calculate error in timing and dynamics r
        # (Pearson's correlation coefficient)
        sim_mean = torch.mean(simulations)
        obs_mean = torch.mean(evaluation)

        r_num = torch.sum((simulations - sim_mean) * (evaluation - obs_mean))
        r_den = torch.sqrt(torch.sum((simulations - sim_mean) ** 2)
                        * torch.sum((evaluation - obs_mean) ** 2))
        r = r_num / r_den
        # calculate error in spread of flow alpha
        alpha = torch.std(simulations) / torch.std(evaluation)
        # calculate error in volume beta (bias of mean discharge)
        beta = (torch.sum(simulations)
                / torch.sum(evaluation))
        # calculate the Kling-Gupta Efficiency KGE
        kge_ = torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        return kge_

def train_TFT(f, out_dir, station_id, region, exptag):
    """This function trains the Temporal Fusion Transformer model on the dataset of a single basin.
    Args:
        f (str): Path to the dataset file.
        out_dir (str): Path to the output directory.
        station_id (str): ID of the station.
        region (str): Region of the station.
        exptag (str): Experiment tag. (a string to identify the experiment or version of run)"""
    print("\nExp details: "+exptag+"\n")

    start_time = time.perf_counter() # to track the time taken for training
    
    data=pd.read_csv(f,
                     parse_dates=['date'], 
                     index_col=['date'],
                     infer_datetime_format=True,
                     low_memory=False)

    prediction_length = 1 # 1 day ahead prediction
    lookback = 365 # Past input sequence length

    # train test split
    train_frac=0.7
    val_frac = 0.1
    n_train = int(data.shape[0] * train_frac)
    n_val = int(data.shape[0] * (train_frac+val_frac))
    
    data_train = data.iloc[:n_train,:]
    data_val = data.iloc[n_train-lookback:n_val,:]
    data_test = data.iloc[n_val-lookback:,:]
    
    # identify static and time-varying variables
    StatRealsColumns = [col for col in data_train.drop(['basin'], axis=1).columns if len(data_train[col].unique()) == 1]
    VaryUnkRealsColumns = [col for col in data_train.drop(['basin'], axis=1).columns if len(data_train[col].unique()) > 1]

    # adding time_idx
    data_train['time_idx'] = (data_train.index-data_train.index.min()).days+1
    data_val['time_idx'] = (data_val.index-data_val.index.min()).days+1
    data_test['time_idx'] = (data_test.index-data_test.index.min()).days+1
    training_cutoff = data_train["time_idx"].max() - prediction_length

    # defining PyTorch-Forecasting dataset
    training = TimeSeriesDataSet(
        data_train,
        time_idx="time_idx",
        target="streamflow",
        group_ids=["basin"],
        min_encoder_length=lookback,  
        # pytorch-forecasting allow variable input sequence lengths, 
        # but we use a fixed length here by setting min_encoder_length and max_encoder_length to the same value
        max_encoder_length=lookback,
        min_prediction_length=1,
        max_prediction_length=prediction_length,
        static_reals= StatRealsColumns,
        time_varying_unknown_reals=VaryUnkRealsColumns,
        target_normalizer=None,
        add_relative_time_idx=True,
    )
    
    # create validation and test set 
    validation = TimeSeriesDataSet.from_dataset(training, data_val, stop_randomization=True)
    testing = TimeSeriesDataSet.from_dataset(training, data_test, stop_randomization=True)
    
    # create dataloaders for model
    numcpu = os.cpu_count()
    batch_size = 32  
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=numcpu, pin_memory=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=numcpu, pin_memory=True)
    test_dataloader = testing.to_dataloader(train=False, batch_size=batch_size, num_workers=numcpu, pin_memory=True)

    # create model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,  
        attention_head_size=4,
        dropout=0.1,  
        hidden_continuous_size=8, 
        output_size=7,  # no.of quantiles for quantile loss
        loss=QuantileLoss(), 
        reduce_on_plateau_patience=4, # reduce learning rate if no improvement in validation loss after 4 epochs
        # default optimizer is ranger
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
    print("\nMODEL LOADED\n")

    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", 
                                            min_delta=1e-4, 
                                            patience=10, 
                                            verbose=False, 
                                            mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    log_path = out_dir+"lightning_logs/"+station_id
    Path(log_path).mkdir(parents=True, exist_ok=True)
    logger = TensorBoardLogger(log_path)  # logging results to a tensorboard
    ckpts = ModelCheckpoint(dirpath=log_path, 
                            monitor="val_loss",
                            save_top_k=1)
    
    # define pytorch lightning trainer
    trainer = pl.Trainer(max_epochs=100,
                         accelerator='gpu', 
                         gpus=-1,
                         gradient_clip_val=0.1,
                         callbacks=[lr_logger, early_stop_callback, ckpts],
                         logger=logger,
                         default_root_dir=log_path,
                         strategy='ddp_find_unused_parameters_false',
                         )
    
    #fit model
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    print("\nDONE TRAINING\n")
    finish_time = time.perf_counter()
    print(f"Trained:::: in {finish_time-start_time} seconds")

    # load best model
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    best_tft.eval().to('cuda')
    print("\nBEST MODEL LOADED\n")
    
    # define dataframes to store variable importance and attention weights
    var_imp_df = pd.DataFrame(columns = ['STATION_ID','TT_FLAG']+best_tft.static_variables+best_tft.encoder_variables)
    att_time_df = pd.DataFrame(columns = ['STATION_ID','TT_FLAG']+[str(i) for i in range(-lookback,0)])

    # predict and interpret TRAINING dataset. interpret_output() returns the attention weights and variable importance
    predictions_train, x_train = best_tft.predict(train_dataloader, return_x = True, mode='raw')
    interpretation_train = best_tft.interpret_output(predictions_train, reduction="sum")
    print("\nTRAIN INTERPRETATION DONE\n")
    finish_time = time.perf_counter()
    print(f"Train interpreted::: in {finish_time-start_time} seconds")

    # store variable importance and attention weights
    var_append = dict(zip(['STATION_ID','TT_FLAG']+best_tft.static_variables+best_tft.encoder_variables,
                          [station_id, 1]+
                          interpretation_train['static_variables'].cpu().numpy().tolist()+
                          interpretation_train['encoder_variables'].cpu().numpy().tolist()
                          ))
    var_imp_df = var_imp_df.append(var_append,
                                   ignore_index=True) 
    
    att_append = dict(zip(['STATION_ID','TT_FLAG']+[str(i) for i in range(-lookback,0)],
                          [station_id, 1]+
                          interpretation_train['attention'].cpu().numpy().tolist()
                          ))
    att_time_df = att_time_df.append(att_append,
                                     ignore_index=True)
    
    # store observed and simulated values
    predictions_train = predictions_train.prediction[...,int(best_tft.hparams.output_size/2)]
    trained_df = pd.DataFrame(columns = ['OBS', 'SIM', 'TT_FLAG'],
                              index = x_train['decoder_time_idx'].cpu().numpy()[:,0])
    trained_df['OBS'] = x_train['decoder_target'].cpu().numpy()
    trained_df['SIM'] = predictions_train.cpu().numpy()
    trained_df['TT_FLAG'] = np.ones((len(predictions_train),1))
    trained_df = trained_df.sort_index()
    trained_df = trained_df.loc[trained_df.index.unique().values]
    trained_df = trained_df[~trained_df.index.duplicated(keep='first')]
    trained_df.index = data_train.loc[data_train['time_idx'].isin(trained_df.index)].index
    

    # predict and interpret VALIDATION dataset.
    predictions_val, x_val = best_tft.predict(val_dataloader, return_x = True, mode='raw')

    # store observed and simulated values
    predictions_val = predictions_val.prediction[...,int(best_tft.hparams.output_size/2)]
    valed_df = pd.DataFrame(columns = ['OBS', 'SIM', 'TT_FLAG'],
                              index = x_val['decoder_time_idx'].cpu().numpy()[:,0])
    valed_df['OBS'] = x_val['decoder_target'].cpu().numpy()
    valed_df['SIM'] = predictions_val.cpu().numpy()
    valed_df['TT_FLAG'] = np.ones((len(predictions_val),1))+1
    valed_df = valed_df.sort_index()
    valed_df = valed_df.loc[valed_df.index.unique().values]
    valed_df = valed_df[~valed_df.index.duplicated(keep='first')]
    valed_df.index = data_val.loc[data_val['time_idx'].isin(valed_df.index)].index

    # predict and interpret TESTING dataset.
    predictions_test, x_test = best_tft.predict(test_dataloader, return_x=True, mode='raw') 
    interpretation_test = best_tft.interpret_output(predictions_test, reduction="sum")
    print("\nTEST INTERPRETATION DONE\n")
    finish_time = time.perf_counter()
    print(f"Test interpreted::: in {finish_time-start_time} seconds")
    
    # store variable importance and attention weights
    var_append = dict(zip(['STATION_ID','TT_FLAG']+best_tft.static_variables+best_tft.encoder_variables,
                          [station_id, 0]+
                          interpretation_test['static_variables'].cpu().numpy().tolist()+
                          interpretation_test['encoder_variables'].cpu().numpy().tolist()
                          ))
    var_imp_df = var_imp_df.append(var_append,
                                   ignore_index=True) 
    
    att_append = dict(zip(['STATION_ID','TT_FLAG']+[str(i) for i in range(-lookback,0)],
                          [station_id, 0]+
                          interpretation_test['attention'].cpu().numpy().tolist()
                          ))
    att_time_df = att_time_df.append(att_append,
                                     ignore_index=True)
    
    # store observed and simulated values
    predictions_test = predictions_test.prediction[...,int(best_tft.hparams.output_size/2)]
    tested_df = pd.DataFrame(columns = ['OBS', 'SIM', 'TT_FLAG'],
                              index = x_test['decoder_time_idx'].cpu().numpy()[:,0])
    tested_df['OBS'] = x_test['decoder_target'].cpu().numpy()
    tested_df['SIM'] = predictions_test.cpu().numpy()
    tested_df['TT_FLAG'] = np.ones((len(predictions_test),1))+2
    tested_df = tested_df.sort_index()
    tested_df = tested_df.loc[tested_df.index.unique().values]
    tested_df = tested_df[~tested_df.index.duplicated(keep='first')]
    tested_df.index = data_test.loc[data_test['time_idx'].isin(tested_df.index)].index

    # combine training, validation and testing dataframes
    ret_df = pd.concat([trained_df, valed_df, tested_df], axis = 0)
    
    # calculate performance scores
    train_KGE_all = hydroeval.kge(trained_df['OBS'].values,trained_df['SIM'].values)
    train_NSE = hydroeval.nse(trained_df['OBS'].values,trained_df['SIM'].values)
    
    test_KGE_all = hydroeval.kge(tested_df['OBS'].values,tested_df['SIM'].values)
    test_NSE = hydroeval.nse(tested_df['OBS'].values,tested_df['SIM'].values)
    
    perf_scores_df = pd.DataFrame([{ 'STATION_ID':station_id,
                                    'train_KGE':train_KGE_all[0,0],
                                    'train_KGE_r':train_KGE_all[1,0],
                                    'train_KGE_a':train_KGE_all[2,0],
                                    'train_KGE_b':train_KGE_all[3,0],
                                    'train_NSE':train_NSE,
                                    'test_KGE':test_KGE_all[0,0],
                                    'test_KGE_r':test_KGE_all[1,0],
                                    'test_KGE_a':test_KGE_all[2,0],
                                    'test_KGE_b':test_KGE_all[3,0],
                                    'test_NSE':test_NSE}])

    perf_scores_df.to_csv(out_dir+'Station_NSE_KGE_bestModel_'+region+'_'+exptag+'.csv',
                          mode='a',
                          header=not os.path.exists(out_dir+'Station_NSE_KGE_bestModel_'+region+'_'+exptag+'.csv'))
    
    var_imp_df.to_csv(out_dir+'Station_VarImportance_'+region+'_'+exptag+'.csv', 
                      mode='a', 
                      header=not os.path.exists(out_dir+'Station_VarImportance_'+region+'_'+exptag+'.csv'))
    
    att_time_df.to_csv(out_dir+'Station_Attention_'+region+'_'+exptag+'.csv', 
                       mode='a', 
                       header=not os.path.exists(out_dir+'Station_Attention_'+region+'_'+exptag+'.csv'))

    print("RESULTS SAVED to "+out_dir+"\n")
    finish_time = time.perf_counter()
    print(f"Done for {station_id} in {finish_time-start_time} seconds") 
    
    # although perfomance scores, variable importance, and attention scores are stored in file, a copy is returned and stored in main function
    return ret_df, var_imp_df, att_time_df, perf_scores_df
    
#%% MAIN
def main():
    exptag = "run1" # Experiment tag to identify the run
    in_dir = '/path/to/dataset/' # Path to the caravan dataset. Make sure the basic preprocessing discussed in Rasiya Koya et al. (2024) is done.
    region = 'camels_gb' # Region of the dataset. This is used to identify the output files.
    files = glob.glob(in_dir+region+'/*.csv')
    out_dir = '/path/to/output/' # Path to the output directory. Make sure the directory exists. Or create one.

    perf_scores_list = []
    var_imp_list = []
    att_time_list = []
    
    # Filter basins that are not trained. This is useful when the training is interrupted and the code is run again.
    files_filter=[]
    for f0 in files:
        fn0 = os.path.basename(f0)
        fn0 = out_dir+'Outputs/'+fn0
        if not os.path.exists(fn0):
            files_filter.append(f0)
    print("\n Length of files : "+str(len(files)))
    print("\n Length of filtered files: "+str(len(files_filter)))

    i = 0; toti = str(len(files_filter)); start_time = time.perf_counter()
    for f in files_filter: 
        fn = os.path.basename(f)
        station_id = fn.split('.')[0]
        i+=1; print("\n>>>>>>>\n>>>>>>> \n\n"+str(i)+'/'+toti+"\t"+station_id+"\n")
        
        # train the model
        obsim_df, var_imp, att_time, perf_scores = train_TFT(f, out_dir, station_id, region, exptag)
        
        obsim_df.to_csv(out_dir+"Outputs/"+fn) # make sure the Outputs/ directory exists inside the out_dir. Or create one.
        
        perf_scores_list.append(perf_scores)
        var_imp_list.append(var_imp)
        att_time_list.append(att_time)

        print("\n>>>>>>>\n>>>>>>>")
        
    perf_scores_df = pd.concat(perf_scores_list, axis=0, ignore_index=True)
    perf_scores_df.to_csv(out_dir+"Station_NSE_NSE_bestModel_"+region+"_"+exptag+"_copy.csv", index=False)

    var_imptcance_df = pd.concat(var_imp_list, axis=0, ignore_index=True)
    var_imptcance_df.to_csv(out_dir+"Station_VarImportance_"+region+"_"+exptag+"_copy.csv", index=False)

    attention_df = pd.concat(att_time_list, axis=0, ignore_index=True)
    attention_df.to_csv(out_dir+"Station_Attention_"+region+"_"+exptag+"_copy.csv", index=False)

    finish_time = time.perf_counter()
    print(f"All grids finished in {finish_time-start_time} seconds")

if __name__=='__main__':
    main()

