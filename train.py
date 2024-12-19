"""
COVID-19 Spread Prediction Using Graph Attention Networks and SIR Model Integration
"""

import torch
import os
import pickle
import numpy as np
import pandas as pd
from epiweeks import Week
from utils import date_today, gravity_law_commute_dist
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch import nn
import torch.nn.functional as F
from model import STAN
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import logging

logging.basicConfig(
    filename='experiment.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.info('Experiment Started')

os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print("CUDA is available. GPU will be used for computations.")
    os.system('nvidia-smi')
    device = torch.device('cuda')
else:
    print("CUDA is not available. CPU will be used for computations.")
    device = torch.device('cpu')
logging.info(f"Using device: {device}")

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()
logging.info('Set random seeds for reproducibility.')

# Ensure 'state_covid_data.pickle' and 'uszips.csv' are present
raw_data = pickle.load(open('./data/state_covid_data.pickle', 'rb'))
logging.info('Loaded COVID-19 data from pickle.')

pop_data = pd.read_csv('./uszips.csv')
pop_data = pop_data.groupby('state_name').agg({
    'population': 'sum',
    'density': 'mean',
    'lat': 'mean',
    'lng': 'mean'
}).reset_index()
logging.info('Aggregated population data.')

raw_data = pd.merge(raw_data, pop_data, how='inner', left_on='state', right_on='state_name')
logging.info('Merged COVID-19 data with population data.')

loc_list = list(raw_data['state'].unique())
loc_dist_map = {}

for each_loc in loc_list:
    loc_dist_map[each_loc] = {}
    for each_loc2 in loc_list:
        lat1 = raw_data[raw_data['state'] == each_loc]['lat'].unique()[0]
        lng1 = raw_data[raw_data['state'] == each_loc]['lng'].unique()[0]
        pop1 = raw_data[raw_data['state'] == each_loc]['population'].unique()[0]
        lat2 = raw_data[raw_data['state'] == each_loc2]['lat'].unique()[0]
        lng2 = raw_data[raw_data['state'] == each_loc2]['lng'].unique()[0]
        pop2 = raw_data[raw_data['state'] == each_loc2]['population'].unique()[0]
        loc_dist_map[each_loc][each_loc2] = gravity_law_commute_dist(lat1, lng1, pop1, lat2, lng2, pop2, r=0.5)

logging.info('Generated location similarity map.')
num_locations = len(loc_list)
print(f"Number of unique locations: {num_locations}")
logging.info(f"Number of unique locations: {num_locations}")

dist_threshold = 18
for each_loc in loc_dist_map:
    loc_dist_map[each_loc] = {k: v for k, v in sorted(loc_dist_map[each_loc].items(), key=lambda item: item[1], reverse=True)}

adj_map = {}
for each_loc in loc_dist_map:
    adj_map[each_loc] = []
    for i, each_loc2 in enumerate(loc_dist_map[each_loc]):
        if loc_dist_map[each_loc][each_loc2] > dist_threshold:
            if i < 4:
                adj_map[each_loc].append(each_loc2)
        else:
            if i < 2:
                adj_map[each_loc].append(each_loc2)

logging.info('Generated adjacency map based on similarity threshold.')

invalid_indices = []
for each_loc in adj_map:
    for each_loc2 in adj_map[each_loc]:
        try:
            idx = loc_list.index(each_loc2)
            if idx >= num_locations:
                invalid_indices.append((each_loc, each_loc2, idx))
        except ValueError:
            invalid_indices.append((each_loc, each_loc2, 'Not Found'))

if invalid_indices:
    print("Invalid indices found in adj_map:")
    for loc, loc2, idx in invalid_indices:
        print(f"{loc} -> {loc2}, Index: {idx}")
    logging.error("Invalid indices found in adj_map.")
    raise ValueError("Invalid indices found in adj_map.")
else:
    logging.info("All adjacency indices are valid.")

rows = []
cols = []
for each_loc in adj_map:
    for each_loc2 in adj_map[each_loc]:
        src_idx = loc_list.index(each_loc)
        dst_idx = loc_list.index(each_loc2)
        rows.append(src_idx)
        cols.append(dst_idx)

edge_index = torch.tensor([rows, cols], dtype=torch.long).to(device)
logging.info(f"Edge index created with {edge_index.size(1)} edges.")
max_index = edge_index.max().item()
min_index = edge_index.min().item()
print(f"edge_index min: {min_index}, max: {max_index}, num_nodes: {num_locations}")
logging.info(f"edge_index min: {min_index}, max: {max_index}, num_nodes: {num_locations}")
if max_index >= num_locations or min_index < 0:
    logging.error("edge_index contains invalid node indices.")
    raise ValueError("edge_index contains invalid node indices.")

g = Data(edge_index=edge_index, num_nodes=num_locations)
g = g.to(device)
logging.info('Created PyG Data object and moved to device.')

active_cases = []
confirmed_cases = []
new_cases = []
death_cases = []
static_feat = []

for each_loc in loc_list:
    active_cases.append(raw_data[raw_data['state'] == each_loc]['active'].values)
    confirmed_cases.append(raw_data[raw_data['state'] == each_loc]['confirmed'].values)
    new_cases.append(raw_data[raw_data['state'] == each_loc]['new_cases'].values)
    death_cases.append(raw_data[raw_data['state'] == each_loc]['deaths'].values)
    static_feat.append(raw_data[raw_data['state'] == each_loc][['population','density','lng','lat']].values[0])

active_cases = np.array(active_cases)
confirmed_cases = np.array(confirmed_cases)
death_cases = np.array(death_cases)
new_cases = np.array(new_cases)
static_feat = np.array(static_feat)
recovered_cases = confirmed_cases - active_cases - death_cases
susceptible_cases = static_feat[:, 0].reshape(-1, 1) - active_cases - recovered_cases

dI = np.concatenate((np.zeros((active_cases.shape[0], 1), dtype=np.float32), np.diff(active_cases)), axis=-1)
dR = np.concatenate((np.zeros((recovered_cases.shape[0], 1), dtype=np.float32), np.diff(recovered_cases)), axis=-1)
dS = np.concatenate((np.zeros((susceptible_cases.shape[0], 1), dtype=np.float32), np.diff(susceptible_cases)), axis=-1)

logging.info('Preprocessed dynamic features (dI, dR, dS).')

normalizer = {'S':{},'I':{},'R':{},'dS':{},'dI':{},'dR':{}}
for i, each_loc in enumerate(loc_list):
    normalizer['S'][each_loc]=(np.mean(susceptible_cases[i]),np.std(susceptible_cases[i]))
    normalizer['I'][each_loc]=(np.mean(active_cases[i]),np.std(active_cases[i]))
    normalizer['R'][each_loc]=(np.mean(recovered_cases[i]),np.std(recovered_cases[i]))
    normalizer['dI'][each_loc]=(np.mean(dI[i]),np.std(dI[i]))
    normalizer['dR'][each_loc]=(np.mean(dR[i]),np.std(dR[i]))
    normalizer['dS'][each_loc]=(np.mean(dS[i]),np.std(dS[i]))

logging.info('Built normalizer for features.')

def prepare_data(data, sum_I, sum_R, history_window=5, pred_window=15, slide_step=5):
    n_loc, timestep, n_feat = data.shape
    x = []
    y_I = []
    y_R = []
    last_I = []
    last_R = []
    concat_I = []
    concat_R = []

    for i in range(0, timestep - history_window - pred_window + 1, slide_step):
        x.append(data[:, i:i+history_window, :].reshape(n_loc, history_window*n_feat))
        concat_I.append(data[:, i+history_window-1, 0])
        concat_R.append(data[:, i+history_window-1, 1])
        last_I.append(sum_I[:, i+history_window-1])
        last_R.append(sum_R[:, i+history_window-1])
        y_I.append(data[:, i+history_window:i+history_window+pred_window,0])
        y_R.append(data[:, i+history_window:i+history_window+pred_window,1])

    x=np.array(x,dtype=np.float32)
    last_I=np.array(last_I,dtype=np.float32)
    last_R=np.array(last_R,dtype=np.float32)
    concat_I=np.array(concat_I,dtype=np.float32)
    concat_R=np.array(concat_R,dtype=np.float32)
    y_I=np.array(y_I,dtype=np.float32)
    y_R=np.array(y_R,dtype=np.float32)

    return x,last_I,last_R,concat_I,concat_R,y_I,y_R

logging.info('Defined data preparation function.')

valid_window=25
test_window=25
history_window=6
pred_window=15
slide_step=5
normalize=True

dynamic_feat = np.concatenate((np.expand_dims(dI, axis=-1),
                               np.expand_dims(dR, axis=-1),
                               np.expand_dims(dS, axis=-1)), axis=-1)

if normalize:
    for i, each_loc in enumerate(loc_list):
        mean_dI,std_dI=normalizer['dI'][each_loc]
        mean_dR,std_dR=normalizer['dR'][each_loc]
        mean_dS,std_dS=normalizer['dS'][each_loc]
        dynamic_feat[i,:,0]=(dynamic_feat[i,:,0]-mean_dI)/(std_dI if std_dI!=0 else 1)
        dynamic_feat[i,:,1]=(dynamic_feat[i,:,1]-mean_dR)/(std_dR if std_dR!=0 else 1)
        dynamic_feat[i,:,2]=(dynamic_feat[i,:,2]-mean_dS)/(std_dS if std_dS!=0 else 1)

logging.info('Normalized dynamic features.')

dI_mean = torch.tensor([normalizer['dI'][loc][0] for loc in loc_list],dtype=torch.float32).to(device)
dI_std  = torch.tensor([normalizer['dI'][loc][1] for loc in loc_list],dtype=torch.float32).to(device)
dR_mean = torch.tensor([normalizer['dR'][loc][0] for loc in loc_list],dtype=torch.float32).to(device)
dR_std  = torch.tensor([normalizer['dR'][loc][1] for loc in loc_list],dtype=torch.float32).to(device)

dI_mean=dI_mean.reshape(1,dI_mean.shape[0],1)
dI_std=dI_std.reshape(1,dI_std.shape[0],1)
dR_mean=dR_mean.reshape(1,dR_mean.shape[0],1)
dR_std=dR_std.reshape(1,dR_std.shape[0],1)

train_feat=dynamic_feat[:,:-valid_window-test_window,:]
val_feat=dynamic_feat[:,-valid_window-test_window:-test_window,:]
test_feat=dynamic_feat[:,-test_window:,:]

train_x,train_I,train_R,train_cI,train_cR,train_yI,train_yR=prepare_data(
    train_feat,active_cases[:,:-valid_window-test_window],recovered_cases[:,:-valid_window-test_window],
    history_window,pred_window,slide_step
)

val_x,val_I,val_R,val_cI,val_cR,val_yI,val_yR=prepare_data(
    val_feat,active_cases[:,-valid_window-test_window:-test_window],
    recovered_cases[:,-valid_window-test_window:-test_window],
    history_window,pred_window,slide_step
)

test_x,test_I,test_R,test_cI,test_cR,test_yI,test_yR=prepare_data(
    test_feat,active_cases[:,-test_window:],recovered_cases[:,-test_window:],
    history_window,pred_window,slide_step
)

logging.info('Prepared training, validation, and test datasets.')

print(f"train_x shape: {train_x.shape}")
print(f"train_cI shape: {train_cI.shape}")
print(f"train_cR shape: {train_cR.shape}")
print(f"train_I shape: {train_I.shape}")
print(f"train_R shape: {train_R.shape}")
print(f"N shape: {static_feat[:, 0].shape}")

logging.info(f"train_x shape: {train_x.shape}")
logging.info(f"train_cI shape: {train_cI.shape}")
logging.info(f"train_cR shape: {train_cR.shape}")
logging.info(f"train_I shape: {train_I.shape}")
logging.info(f"train_R shape: {train_R.shape}")
logging.info(f"N shape: {static_feat[:, 0].shape}")

in_dim=3*history_window
hidden_dim1=32
hidden_dim2=32
gru_dim=32
num_heads=1
pred_window=15

model=STAN(g,in_dim,hidden_dim1,hidden_dim2,gru_dim,num_heads,pred_window,device).to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-2)
criterion=nn.MSELoss()

logging.info('Initialized STAN model, optimizer, and loss function.')

train_x=torch.tensor(train_x).to(device)
train_I=torch.tensor(train_I).unsqueeze(2).to(device)
train_R=torch.tensor(train_R).unsqueeze(2).to(device)
train_cI=torch.tensor(train_cI).unsqueeze(2).to(device)
train_cR=torch.tensor(train_cR).unsqueeze(2).to(device)
train_yI=torch.tensor(train_yI).to(device)
train_yR=torch.tensor(train_yR).to(device)

val_x=torch.tensor(val_x).to(device)
val_I=torch.tensor(val_I).unsqueeze(2).to(device)
val_R=torch.tensor(val_R).unsqueeze(2).to(device)
val_cI=torch.tensor(val_cI).unsqueeze(2).to(device)
val_cR=torch.tensor(val_cR).unsqueeze(2).to(device)
val_yI=torch.tensor(val_yI).to(device)
val_yR=torch.tensor(val_yR).to(device)

test_x=torch.tensor(test_x).to(device)
test_I=torch.tensor(test_I).unsqueeze(2).to(device)
test_R=torch.tensor(test_R).unsqueeze(2).to(device)
test_cI=torch.tensor(test_cI).unsqueeze(2).to(device)
test_cR=torch.tensor(test_cR).unsqueeze(2).to(device)
test_yI=torch.tensor(test_yI).to(device)
test_yR=torch.tensor(test_yR).to(device)

dI_mean=dI_mean.to(device)
dI_std=dI_std.to(device)
dR_mean=dR_mean.to(device)
dR_std=dR_std.to(device)

N=torch.tensor(static_feat[:,0],dtype=torch.float32).to(device).unsqueeze(1)

logging.info('Converted all data to tensors and moved to device.')

all_loss=[]
val_loss_history=[]
file_name='./save/stan.pth'
min_val_loss=float('inf')
patience=10
counter=0
scale=0.1
max_epochs=100

os.makedirs('./save', exist_ok=True)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=5,verbose=True)

logging.info('Starting training loop.')
progress_bar=tqdm(range(max_epochs),desc="Training Epochs")

for epoch_num in progress_bar:
    model.train()
    optimizer.zero_grad()
    try:
        active_pred,recovered_pred,phy_active,phy_recover,h=model(train_x,train_cI,train_cR,N,train_I,train_R)
    except Exception as e:
        logging.error("Error during forward pass.", exc_info=True)
        raise e

    if normalize:
        phy_active=(phy_active - dI_mean)/dI_std
        phy_recover=(phy_recover - dR_mean)/dR_std

    # All have shape [time,batch,pred_window]
    loss = (criterion(active_pred, train_yI) +
            criterion(recovered_pred, train_yR) +
            scale*criterion(phy_active, train_yI) +
            scale*criterion(phy_recover, train_yR))

    loss.backward()
    optimizer.step()
    all_loss.append(loss.item())

    model.eval()
    with torch.no_grad():
        try:
            val_active_pred,val_recovered_pred,val_phy_active,val_phy_recover,_=model(val_x,val_cI,val_cR,N,val_I,val_R)
            if normalize:
                val_phy_active=(val_phy_active - dI_mean)/dI_std
                val_phy_recover=(val_phy_recover - dR_mean)/dR_std

            val_loss = (criterion(val_active_pred,val_yI) +
                        criterion(val_recovered_pred,val_yR) +
                        scale*criterion(val_phy_active,val_yI) +
                        scale*criterion(val_phy_recover,val_yR))
        except Exception as e:
            logging.error("Error during validation forward pass.", exc_info=True)
            raise e

    val_loss_history.append(val_loss.item())
    logging.info(f"Epoch {epoch_num+1}/{max_epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")
    progress_bar.set_postfix({'Train Loss': f"{loss.item():.4f}", 'Val Loss': f"{val_loss.item():.4f}"})

    if val_loss.item()<min_val_loss:
        torch.save({
            'epoch':epoch_num+1,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':loss.item(),
        },file_name)
        logging.info(f"Validation loss improved to {val_loss.item():.4f}. Model saved.")
        min_val_loss=val_loss.item()
        counter=0
    else:
        counter+=1
        if counter>=patience:
            logging.info("Early stopping triggered.")
            print("Early stopping triggered.")
            break

    scheduler.step(val_loss)

logging.info('Training loop completed.')

checkpoint=torch.load(file_name,map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()
logging.info('Loaded best model for evaluation.')

with torch.no_grad():
    try:
        test_active_pred,test_recovered_pred,test_phy_active,test_phy_recover,_=model(test_x,test_cI,test_cR,N,test_I,test_R)
        if normalize:
            test_active_pred=test_active_pred*dI_std+dI_mean
            test_recovered_pred=test_recovered_pred*dR_std+dR_mean
            test_phy_active=test_phy_active*dI_std+dI_mean
            test_phy_recover=test_phy_recover*dR_std+dR_mean
    except Exception as e:
        logging.error("Error during test forward pass.", exc_info=True)
        raise e

logging.info('Performed forward pass on test data.')

def get_real_y(data,history_window=5,pred_window=15,slide_step=5):
    n_loc,timestep=data.shape
    y=[]
    for i in range(0,timestep-history_window-pred_window+1,slide_step):
        y.append(data[:,i+history_window:i+history_window+pred_window])
    return np.array(y,dtype=np.float32)

logging.info('Defined function to extract real Y values.')

I_true=get_real_y(active_cases,history_window,pred_window,slide_step)
logging.info('Extracted real Y values for test set.')

plt.figure(figsize=(10,5))
plt.plot(all_loss,label='Training Loss')
plt.plot(val_loss_history,label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
logging.info('Plotted training and validation loss curves.')

states_to_plot=['California','New York','Texas']
for state in states_to_plot:
    if state not in loc_list:
        logging.warning(f"State {state} not found in loc_list.")
        continue
    loc=loc_list.index(state)
    plt.figure(figsize=(10,5))
    plt.plot(I_true[-1,loc,:],c='r',label='Ground Truth')
    plt.plot(test_active_pred[-1,loc,:].cpu().numpy(),c='b',label='Prediction')
    plt.title(f"COVID-19 Active Cases Prediction for {state}")
    plt.xlabel('Time Steps')
    plt.ylabel('Active Cases')
    plt.legend()
    plt.grid(True)
    plt.show()
    logging.info(f'Plotted predictions for {state}.')

def calculate_metrics(y_true,y_pred):
    mae=mean_absolute_error(y_true,y_pred)
    mse=mean_squared_error(y_true,y_pred)
    rmse=np.sqrt(mse)
    ss_res=np.sum((y_true-y_pred)**2)
    ss_tot=np.sum((y_true-np.mean(y_true))**2)
    r2=1-ss_res/ss_tot
    return {'MAE':mae,'MSE':mse,'RMSE':rmse,'R2':r2}

california_idx=loc_list.index('California')
y_true=I_true[:,california_idx,:].reshape(-1)
y_pred=test_active_pred[:,california_idx,:].cpu().numpy().reshape(-1)

metrics=calculate_metrics(y_true,y_pred)
print(f"Evaluation Metrics for California:")
for metric,value in metrics.items():
    print(f"{metric}: {value:.4f}")
    logging.info(f"{metric} for California: {value:.4f}")

residuals=y_true-y_pred
plt.figure(figsize=(10,5))
plt.hist(residuals,bins=50,color='purple',edgecolor='black')
plt.title('Residuals Distribution for California')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
logging.info('Performed residual analysis for California.')

torch.save({
    'epoch':epoch_num+1,
    'model_state_dict':model.state_dict(),
    'optimizer_state_dict':optimizer.state_dict(),
    'loss':loss.item(),
},'./save/stan_final.pth')
logging.info('Saved final model.')
logging.info('Experiment Completed Successfully.')
