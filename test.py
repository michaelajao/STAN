import math
import pickle
import pandas as pd
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def gravity_law_commute_dist(lat1, lng1, pop1, lat2, lng2, pop2, r=0.5):
    """
    Calculate the gravity law-based commute distance between two locations.
    """
    # Calculate Euclidean distance between two points
    distance = math.sqrt((lat1 - lat2)**2 + (lng1 - lng2)**2)
    # Apply gravity law formula
    gravity_dist = (pop1 * pop2) / (distance ** r + 1e-5)  # Added epsilon to avoid division by zero
    return gravity_dist

# Load COVID-19 data
raw_data = pickle.load(open('./data/state_covid_data.pickle', 'rb'))

# Load population data
pop_data = pd.read_csv('./uszips.csv')

# Aggregate population data by state
pop_data = pop_data.groupby('state_name').agg({
    'population': 'sum',
    'density': 'mean',
    'lat': 'mean',
    'lng': 'mean'
}).reset_index()

# Merge COVID-19 data with population data
raw_data = pd.merge(raw_data, pop_data, how='inner', left_on='state', right_on='state_name')

# Generate location similarity based on gravity law
loc_list = raw_data['state'].unique()
loc_list = sorted(loc_list)
num_states = len(loc_list)
state_to_index = {state: idx for idx, state in enumerate(loc_list)}
loc_dist_map = {}

for each_loc in loc_list:
    loc_dist_map[each_loc] = {}
    # Extract features for each_loc
    loc_data = raw_data[raw_data['state'] == each_loc].iloc[0]
    lat1 = loc_data['lat']
    lng1 = loc_data['lng']
    pop1 = loc_data['population']
    
    for each_loc2 in loc_list:
        # Extract features for each_loc2
        loc2_data = raw_data[raw_data['state'] == each_loc2].iloc[0]
        lat2 = loc2_data['lat']
        lng2 = loc2_data['lng']
        pop2 = loc2_data['population']
        
        # Calculate gravity-based commute distance
        gravity_dist = gravity_law_commute_dist(lat1, lng1, pop1, lat2, lng2, pop2, r=0.5)
        loc_dist_map[each_loc][each_loc2] = gravity_dist

print("Location Similarity Map Generated.")

# Define distance threshold
dist_threshold = 52 # Adjust based on data distribution

# Sort each state's neighbors based on similarity scores in descending order
for each_loc in loc_dist_map:
    # Sort neighbors by similarity score (descending)
    sorted_neighbors = sorted(loc_dist_map[each_loc].items(), key=lambda item: item[1], reverse=True)
    loc_dist_map[each_loc] = sorted_neighbors

# Create adjacency map: for each state, select top neighbors based on threshold
adj_map = {}
for each_loc in loc_dist_map:
    adj_map[each_loc] = []
    for i, (each_loc2, score) in enumerate(loc_dist_map[each_loc]):
        if score > dist_threshold:
            if i < 3:  # Select top 3 neighbors above threshold
                adj_map[each_loc].append(each_loc2)
        else:
            if i < 1:  # Select top 1 neighbor below threshold
                adj_map[each_loc].append(each_loc2)
            else:
                break  # Stop if score is below threshold and limit reached

# Display adjacency map
print("Adjacency Map:")
for state, neighbors in adj_map.items():
    print(f"{state}: {neighbors}")

# Initialize adjacency matrix
adj_matrix = np.zeros((num_states, num_states), dtype=np.float32)

# Populate adjacency matrix based on adj_map
for each_loc in adj_map:
    i = state_to_index[each_loc]
    for each_loc2 in adj_map[each_loc]:
        j = state_to_index[each_loc2]
        adj_matrix[i][j] = 1  # Binary adjacency; set to 1 if connected

# Add self-loops to the adjacency matrix
adj_matrix += np.eye(num_states, dtype=np.float32)

# Normalize adjacency matrix
adj_matrix = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)

# Convert adjacency matrix to tensor
adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32).to(device)

print(f"Adjacency Matrix Shape: {adj_matrix.shape}")

# Select relevant columns
dynamic_feature_names = ['confirmed', 'deaths', 'recovered', 'active', 'hospitalization', 'new_cases']
static_feature_names = ['population', 'density', 'lat', 'lng']

# Sort data by state and date
raw_data = raw_data.sort_values(['state', 'date_today']).reset_index(drop=True)

# Get unique states and dates
unique_states = sorted(raw_data['state'].unique())
num_states = len(unique_states)
state_to_index = {state: idx for idx, state in enumerate(unique_states)}

unique_dates = sorted(raw_data['date_today'].unique())
num_dates = len(unique_dates)
date_to_index = {date: idx for idx, date in enumerate(unique_dates)}

# Initialize dynamic_tensor and static_tensor
num_dynamic_features = len(dynamic_feature_names)
num_static_features = len(static_feature_names)

dynamic_tensor = np.zeros((num_states, num_dates, num_dynamic_features))
static_tensor = np.zeros((num_states, num_static_features))

# Fill in dynamic_tensor
for idx, row in raw_data.iterrows():
    state_idx = state_to_index[row['state']]
    date_idx = date_to_index[row['date_today']]
    dynamic_tensor[state_idx, date_idx, :] = row[dynamic_feature_names].values

# Fill in static_tensor (assuming static features are the same across dates)
for state in unique_states:
    state_idx = state_to_index[state]
    state_data = raw_data[raw_data['state'] == state]
    static_features = state_data[static_feature_names].iloc[0].values
    static_tensor[state_idx, :] = static_features

# Create date_range array
date_range = np.array(unique_dates)

# Print tensor shapes
print(f"dynamic_tensor shape: {dynamic_tensor.shape}")
print(f"static_tensor shape: {static_tensor.shape}")
print(f"date_range shape: {date_range.shape}")

# Create 'feat_name' dictionary similar to sample output
feat_name = {
    'dynamic': dynamic_feature_names,
    'static': static_feature_names,
    'date': unique_dates
}

# split it into X and y (y should be the hospitalization column)
X = dynamic_tensor[:, :, :]
y = dynamic_tensor[:, :, 4]  # hospitalization column

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")