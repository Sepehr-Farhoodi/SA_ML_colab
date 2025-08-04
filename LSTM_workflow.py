#!/usr/bin/env python3

import pandas as pd
from datetime import datetime, date
import os
from pathlib import Path
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION SECTION - All hyperparameters and settings
# ============================================================================

# Data periods configuration
DATA_PERIODS = {
    'training': {
        'start': datetime(1980, 10, 1),
        'end': datetime(1995, 9, 30),
        'description': 'Training period (15 years)'
    },
    'validation': {
        'start': datetime(1995, 10, 1),
        'end': datetime(2000, 9, 30),
        'description': 'Validation period (5 years)'
    },
    'test': {
        'start': datetime(1995, 10, 1),
        'end': datetime(2014, 9, 30),
        'description': 'Test period (14 years)'
    }
}

# LSTM model hyperparameters
LSTM_CONFIG = {
    'input_size': 5,           # Number of input features
    'hidden_size': 20,         # LSTM hidden layer size
    'num_layers': 2,           # Number of LSTM layers
    'output_size': 1,          # Number of output features
    'sequence_length': 365,    # Length of input sequences
    'num_samples': 512,        # Number of training samples to generate
    'batch_size': 64,          # Training batch size
    'learning_rate': 0.001,    # Learning rate
    'num_epochs': 50,         # Number of training epochs
    'dropout_rate': 0.2,        # Dropout rate for regularization
    'epochs_to_plot': [1, 2, 3, 5, 10, 20, 50] # Epochs to visualize
}

# File paths configuration
FILE_PATHS = {
    'project_root': Path(__file__).parent,
    'input_files': {
        'meteorological': "basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/01/01013500_lump_cida_forcing_leap.txt",
        'flow': "model_output_daymet/model_output/flow_timeseries/daymet/01/01013500_05_model_output.txt"
    }
}

# ============================================================================
# DATASET CLASS DEFINITION
# ============================================================================

class HydroDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # shape: (num_samples, seq_len, num_features)
        self.y = y  # shape: (num_samples, 1) or (num_samples, seq_len)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Always return float32 tensors
        X_item = torch.tensor(self.X[idx], dtype=torch.float32)
        y_item = torch.tensor(self.y[idx], dtype=torch.float32)
        return X_item, y_item

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        final_hidden = out[:, -1, :]
        prediction = self.fc(final_hidden)
        return prediction  

# ============================================================================
# Model Initialization
# ============================================================================
project_path = Path(__file__).parent
dataset_type = 'training'

# ============================================================================
# Main functions
# ============================================================================

# Helper function to get file paths
def get_output_file_path(dataset_type):
    """
    Generate output file path based on dataset type and period.
    
    Parameters:
    -----------
    dataset_type : str
        Type of dataset ('training', 'validation', 'test')
    project_path : Path
        Project root path
        
    Returns:
    --------
    str
        Output file path
    """
    period = DATA_PERIODS[dataset_type]
    start_year = period['start'].year
    end_year = period['end'].year
    
    return f"{project_path}/{dataset_type}_{start_year}_{end_year}_01013500_05_forcing_flow_normalized.txt"

# Helper function to get period dates
def get_period_dates(dataset_type):
    """
    Get start and end dates for a specific dataset type.
    
    Parameters:
    -----------
    dataset_type : str
        Type of dataset ('training', 'validation', 'test')
        
    Returns:
    --------
    tuple : (start_date, end_date)
        Start and end dates for the period
    """
    period = DATA_PERIODS[dataset_type]
    return period['start'], period['end']



def extract_flow_data(dataset_type):
    """
    Extract the flow observation and model output data from the input file.
    From start_date to end_date.
    
    Parameters:
    -----------
    input_file : str
        Path to the input file containing flow data
    stat_file : str
        Path to save the statistics of the extracted data
    start_date : datetime
        Start date for data extraction
    end_date : datetime
        End date for data extraction
    dataset_type : str, optional
        Type of dataset ('training' or 'validation'). Default is 'training'.
        This affects the normalization approach and output naming.
    
    Returns:
    --------
    tuple : (filtered_data, mean, std)
        filtered_data: DataFrame with extracted data
        mean: Dictionary of mean values for normalization
        std: Dictionary of std values for normalization
    """

    # Get dates from configuration
    start_date, end_date = get_period_dates(dataset_type)
    
    # Get file paths
    input_file = FILE_PATHS['input_files']['flow']
    
    
    print(f"Extracting {dataset_type} data from: {input_file}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Description: {DATA_PERIODS[dataset_type]['description']}")
    

    # Read the header line first
    with open(input_file, 'r') as f:
        header = f.readline().strip() # Read the 1st line as header
    
    # Read the data, skipping the header
    data = pd.read_csv(input_file, skiprows=1, delim_whitespace=True, 
                      names=['YR', 'MNTH', 'DY', 'HR', 'SWE', 'PRCP', 'RAIM', 'TAIR', 'PET', 'ET', 'MOD_RUN', 'OBS_RUN'])
    
    print(f"Total records in file: {len(data)}")
    print(f"File date range: {data['YR'].min()}-{data['MNTH'].min()}-{data['DY'].min()} to {data['YR'].max()}-{data['MNTH'].max()}-{data['DY'].max()}")
    
    # Create datetime column for easier filtering
    data['date'] = pd.to_datetime({
        'year': data['YR'],
        'month': data['MNTH'], 
        'day': data['DY']
    })
    
    # Filter data for the specified date range
    filtered_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)].copy()
    
    if len(filtered_data) == 0:
        raise ValueError(f"No data found in the specified date range: {start_date.date()} to {end_date.date()}")
    
    print(f"Records in the date range: {len(filtered_data)}")
    print(f"Filtered date range: {filtered_data['date'].min().date()} to {filtered_data['date'].max().date()}")
    
    # Remove the temporary date column and keep only needed columns
    filtered_data = filtered_data[['YR', 'MNTH', 'DY', 'OBS_RUN', 'MOD_RUN']]

    # Define header for output
    header = "YR MNTH DY OBS_RUN MOD_RUN"

    # Calculate normalized values for variables
    mean = {}
    std = {}
    variables_to_normalize = ['OBS_RUN', 'MOD_RUN']
    
    for var in variables_to_normalize:
        mean[var] = filtered_data[var].mean()
        std[var] = filtered_data[var].std()
        filtered_data[var] = (filtered_data[var] - mean[var]) / std[var]  # Fixed: use mean[var] and std[var]
    
    # Print some statistics
    print(f"\nData summary for {dataset_type} period:")
    print(f"Years covered: {filtered_data['YR'].nunique()}")
    print(f"Total days: {len(filtered_data)}")
    print(f"Expected days: {(end_date - start_date).days + 1}")
    
    # Print normalization statistics
    print(f"\nNormalization statistics:")
    for var in variables_to_normalize:
        print(f"{var}: mean={mean[var]:.4f}, std={std[var]:.4f}")
    
    return filtered_data, mean, std


def extract_meteo_data(dataset_type):
    """
    Extract meteorological data from the input file for a specified date range.
    
    Parameters:
    -----------
    input_file : str
        Path to the input file containing meteorological data
    output_file : str
        Path to save the extracted data
    start_date : datetime
        Start date for data extraction
    end_date : datetime
        End date for data extraction
    dataset_type : str, optional
        Type of dataset ('training' or 'validation'). Default is 'training'.
    
    Returns:
    --------
    DataFrame
        Extracted and normalized meteorological data
    """

    # Get dates from configuration
    start_date, end_date = get_period_dates(dataset_type)
    
    # Get file paths
    input_file = FILE_PATHS['input_files']['meteorological']
    
    print(f"Extracting {dataset_type} meteorological data from: {input_file}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Read the header line first
    with open(input_file, 'r') as f:
        for _ in range(3):
            f.readline()
        header = f.readline().strip() # Read the 4th line as header
    
    # Read the data, skipping the header
    data = pd.read_csv(input_file, skiprows=4, delim_whitespace=True, 
                      names=['Year', 'Mnth', 'Day', 'Hr', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)'])
    
    print(f"Total records in file: {len(data)}")
    print(f"File date range: {data['Year'].min()}-{data['Mnth'].min()}-{data['Day'].min()} to {data['Year'].max()}-{data['Mnth'].max()}-{data['Day'].max()}")
    
    # Create datetime column for easier filtering
    data['date'] = pd.to_datetime({
        'year': data['Year'],
        'month': data['Mnth'], 
        'day': data['Day']
    })
    
    # Filter data for the specified date range
    filtered_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)].copy()
    
    if len(filtered_data) == 0:
        raise ValueError(f"No data found in the specified date range: {start_date.date()} to {end_date.date()}")
    
    print(f"Records in the date range: {len(filtered_data)}")
    print(f"Filtered date range: {filtered_data['date'].min().date()} to {filtered_data['date'].max().date()}")
    
    # Remove the temporary date column and unnecessary columns
    filtered_data = filtered_data.drop(columns=['date', 'Hr', 'dayl(s)', 'swe(mm)'], axis=1)

    # Define header for output
    header = "Year Mnth Day prcp(mm/day) srad(W/m2) tmax(C) tmin(C) vp(Pa)"

    # Calculate normalized values for meteorological variables
    variables_to_normalize = ['prcp(mm/day)', 'srad(W/m2)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']
    
    print(f"\nNormalization statistics for {dataset_type} data:")
    for var in variables_to_normalize:
        mean = filtered_data[var].mean()
        std = filtered_data[var].std()
        filtered_data[var] = (filtered_data[var] - mean) / std
        print(f"{var}: mean={mean:.4f}, std={std:.4f}")
    
    # Print some statistics
    print(f"\nData summary for {dataset_type} period:")
    print(f"Years covered: {filtered_data['Year'].nunique()}")
    print(f"Total days: {len(filtered_data)}")
    print(f"Expected days: {(end_date - start_date).days + 1}")
    
    return filtered_data


def add_normalized_obs_flow_to_meteo(meteo_data, flow_data, dataset_type):
    """
    Adds the normalized observed flow as a new column to the normalized meteo file,
    and writes the result to output_file with an updated header.
    Assumes both files have the same number of rows and matching dates.
    """

    flow_data = flow_data.rename(columns={'YR': 'Year', 'MNTH': 'Mnth', 'DY': 'Day'})

    # Merge on Year, Mnth, Day to ensure alignment
    merged_df = pd.merge(meteo_data, flow_data, on=['Year', 'Mnth', 'Day'], how='inner')

    # New header
    new_header = "Year Mnth Day prcp(mm/day) srad(W/m2) tmax(C) tmin(C) vp(Pa) obs_flow mod_flow"

    merged_df = merged_df.rename(columns={'OBS_RUN': 'obs_flow', 'MOD_RUN': 'mod_flow'})

    output_file = get_output_file_path(dataset_type)

    # Write to output file
    with open(output_file, 'w') as f:
        f.write(new_header + '\n')
        merged_df.to_csv(f, sep=' ', index=False, header=False, float_format='%.7f')

    print(f"Combined meteo + normalized obs flow saved to: {output_file}")
    return merged_df

def nse(simulated, observed):
    """
    Nash-Sutcliffe Efficiency.
    Both inputs must be 1D NumPy arrays.
    """
    return 1 - np.sum((simulated - observed) ** 2) / np.sum((observed - np.mean(observed)) ** 2)


def LSTM_training(merged_data, flow_mean, flow_std):

    seq_len = LSTM_CONFIG['sequence_length']
    num_samples = LSTM_CONFIG['num_samples']

    # tensor creation

    # --- LOAD DATA ---
    forcing_obs = merged_data

    # --- CREATE SAMPLES ---
    X_list = []
    y_list = []

    # Set random seed for reproducibility
    np.random.seed(42)

    # Calculate the maximum possible start index
    max_start_idx = len(forcing_obs) - seq_len - 1

    print(f"Creating {num_samples} random samples...")
    print(f"Data length: {len(forcing_obs)}")
    print(f"Sequence length: {seq_len}")
    print(f"Max possible start index: {max_start_idx}")


    for i in range(num_samples):
        start_idx = np.random.randint(0, max_start_idx + 1)
        end_idx = start_idx + seq_len
        y_idx = end_idx  # The day after the last X day

        # Make sure we don't go out of bounds
        if y_idx >= len(forcing_obs):
            break

        # X: shape (365, 5)
        X_seq = forcing_obs.iloc[start_idx:end_idx][['prcp(mm/day)', 'srad(W/m2)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']].values
        # y: shape (1,)
        y_val = forcing_obs.iloc[y_idx]['obs_flow']

        X_list.append(X_seq)
        y_list.append([y_val])

        # Optional: Print some sample dates for verification
        if i < 3:  # Print first 3 samples
            start_date = f"{int(forcing_obs.iloc[start_idx]['Year'])}-{int(forcing_obs.iloc[start_idx]['Mnth']):02d}-{int(forcing_obs.iloc[start_idx]['Day']):02d}"
            end_date = f"{int(forcing_obs.iloc[end_idx-1]['Year'])}-{int(forcing_obs.iloc[end_idx-1]['Mnth']):02d}-{int(forcing_obs.iloc[end_idx-1]['Day']):02d}"
            target_date = f"{int(forcing_obs.iloc[y_idx]['Year'])}-{int(forcing_obs.iloc[y_idx]['Mnth']):02d}-{int(forcing_obs.iloc[y_idx]['Day']):02d}"
            print(f"Sample {i+1}: X period {start_date} to {end_date}, y target {target_date}")

    X_train = np.stack(X_list)  # shape (num_samples, 365, 5)
    y_train = np.stack(y_list)  # shape (num_samples, 1)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # Dataloader

    train_dataset = HydroDataset(X_train, y_train)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    print(f"Number of batches: {len(train_loader)}")

    # Define the loss function and optimizer

    model = LSTMModel(input_size=5, hidden_size=20, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model

    # Epochs to visualize
    epoch_preds = {}
    nse_per_epoch = {}

    for epoch in range(1, LSTM_CONFIG['num_epochs'] + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        # Evaluate and store predictions at selected epochs
        if epoch in LSTM_CONFIG['epochs_to_plot']:
            model.eval()
            with torch.no_grad():
                all_preds = []
                all_obs = []
                for X_batch, y_batch in train_loader:
                    y_pred = model(X_batch)
                    all_preds.append(y_pred.detach().cpu())
                    all_obs.append(y_batch.detach().cpu())

                # Concatenate all batches
                y_pred_full = torch.cat(all_preds).squeeze().numpy()
                y_obs_full = torch.cat(all_obs).squeeze().numpy()

                # Denormalize
                y_pred_denorm = y_pred_full * flow_std + flow_mean
                y_obs_denorm = y_obs_full * flow_std + flow_mean

                # compute_NSE
                nse_value = nse(y_pred_denorm, y_obs_denorm)
                nse_per_epoch[epoch] = nse_value

                # Store predictions
                epoch_preds[epoch] = (y_pred_denorm, y_obs_denorm)

    # Plot the NSE score over epochs

    epochs = sorted(nse_per_epoch.keys())
    nse_scores = [nse_per_epoch[e] for e in epochs]

    plt.figure(figsize=(12, 5))
    plt.plot(epochs, nse_scores, marker='o', linestyle='-', color='b')
    plt.title("NSE Score Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("NSE Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    
    # ============================================================================
    # STEP 1: EXTRACT TRAINING DATA (1995-2014)
    # ============================================================================
    print("\n" + "=" * 40)
    print(f"STEP 1: EXTRACTING TRAINING DATA {DATA_PERIODS['training']['description']}")
    print("=" * 40)
    
    
    # Extract training data
    extracted_data_meteo_train = extract_meteo_data(dataset_type='training')
    extracted_data_flow_train, flow_mean_train, flow_std_train = extract_flow_data(dataset_type='training')
    
    print(f'\nTraining flow normalization parameters:')
    print(f'mean = {flow_mean_train}')
    print(f'std = {flow_std_train}')
    
    # Combine training data
    merged_data_train = add_normalized_obs_flow_to_meteo(extracted_data_meteo_train, extracted_data_flow_train, dataset_type='training')
    
    # ============================================================================
    # STEP 2: EXTRACT TEST DATA (2015-2019)
    # ============================================================================
    print("\n" + "=" * 40)
    print(f"STEP 2: EXTRACTING TEST DATA {DATA_PERIODS['test']['description']}")
    print("=" * 40)
    
    # Extract test data
    extracted_data_meteo_test = extract_meteo_data(dataset_type='test')
    extracted_data_flow_test, flow_mean_test, flow_std_test = extract_flow_data(dataset_type='test')
    
    print(f'\nTest flow normalization parameters:')
    print(f'mean = {flow_mean_test}')
    print(f'std = {flow_std_test}')

    # Combine test data
    merged_data_test = add_normalized_obs_flow_to_meteo(extracted_data_meteo_test, extracted_data_flow_test, dataset_type='test')

    # ==================================================s==========================
    # STEP 3: SUMMARY
    # ============================================================================
    print("\n" + "=" * 40)
    print("STEP 3: WORKFLOW SUMMARY")
    print("=" * 40)
    
    print(f"\nTraining dataset:")
    print(f"- Period: {DATA_PERIODS['training']['start'].date()} to {DATA_PERIODS['training']['end'].date()}")
    print(f"- Total days: {len(merged_data_train)}")
    print(f"- Output file: {get_output_file_path(dataset_type='training')}")
    
    print(f"\nTest dataset:")
    print(f"- Period: {DATA_PERIODS['test']['start'].date()} to {DATA_PERIODS['test']['end'].date()}")
    print(f"- Total days: {len(merged_data_test)}")
    print(f"- Output file: {get_output_file_path(dataset_type='test')}")
    
    print(f"\nNormalization parameters saved for LSTM training:")
    print(f"- Training flow mean: {flow_mean_train}")
    print(f"- Training flow std: {flow_std_train}")
    print(f"- Test flow mean: {flow_mean_test}")
    print(f"- Test flow std: {flow_std_test}")
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Use training data for LSTM model training")
    print("2. Use test data for model evaluation")
    print("3. Apply normalization parameters consistently across datasets") 
    
    # ============================================================================
    # STEP 4: LSTM TRAINING
    # ============================================================================
    print("\n" + "=" * 40)
    print(f"STEP 4: LSTM TRAINING {DATA_PERIODS[dataset_type]['description']}")
    print("=" * 40)

    flow_std_train = flow_std_train['OBS_RUN']

    flow_mean_train = flow_mean_train['OBS_RUN']

    LSTM_training(merged_data_train, flow_mean_train, flow_std_train)
    

    