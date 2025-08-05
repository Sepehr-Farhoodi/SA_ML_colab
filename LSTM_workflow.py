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
import argparse

# ============================================================================
# CONFIGURATION SECTION - All hyperparameters and settings
# ============================================================================

# Watershed configuration
WATERSHED_CONFIG = {
    'watershed_id': '01013500',  # Change this for different watersheds
    'HUC_number': '01',
    'description': 'BASIN'
}

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
    'num_samples_test': 100,    # Number of test samples to generate
    'batch_size': 64,          # Training batch size
    'learning_rate': 0.001,    # Learning rate
    'num_epochs': 50,         # Number of training epochs
    'dropout_rate': 0.2,        # Dropout rate for regularization
    'epochs_to_plot': [1, 2, 3, 5, 10, 20, 50] # Epochs to visualize
}

# Define project root first
PROJECT_ROOT = Path(__file__).parent

# File paths configuration
FILE_PATHS = {
    'project_root': PROJECT_ROOT,
    'input_files': {
        'meteorological': PROJECT_ROOT.parent / f"basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/{WATERSHED_CONFIG['HUC_number']}/{WATERSHED_CONFIG['watershed_id']}_lump_cida_forcing_leap.txt",
        'flow': PROJECT_ROOT.parent / f"model_output_daymet/model_output/flow_timeseries/daymet/{WATERSHED_CONFIG['HUC_number']}/{WATERSHED_CONFIG['watershed_id']}_11_model_output.txt"
    }
}

# ============================================================================
# CLASS DEFINITION
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
        
    Returns:
    --------
    str
        Output file path
    """
    period = DATA_PERIODS[dataset_type]
    start_year = period['start'].year
    end_year = period['end'].year
    watershed_id = WATERSHED_CONFIG['watershed_id']
    
    return f"{FILE_PATHS['project_root']}/{dataset_type}_{start_year}_{end_year}_{watershed_id}_05_forcing_flow_normalized.txt"

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
    variables_to_normalize = ['OBS_RUN']
    
    for var in variables_to_normalize:
        mean[var] = filtered_data[var].mean() 
        std[var] = filtered_data[var].std()
        filtered_data[var] = (filtered_data[var] - mean[var]) / std[var] 
    
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
    best_nse = -float('inf')
    best_model_state = None

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

                # Save best model
                if nse_value > best_nse:
                    best_nse = nse_value
                    best_model_state = model.state_dict().copy()

    # Save the best model
    torch.save(best_model_state, FILE_PATHS['project_root'] / f"{WATERSHED_CONFIG['watershed_id']}_best_model.pth")
    print(f"Best model saved with NSE: {best_nse:.4f}")

    # Create a fresh model with best parameters
    best_model = LSTMModel(input_size=5, hidden_size=20, num_layers=2, output_size=1)
    best_model.load_state_dict(best_model_state)

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

    return best_model # Returns the best model

def LSTM_testing(merged_data, flow_mean, flow_std, trained_model=None):

    seq_len = LSTM_CONFIG['sequence_length']
    num_samples_test = LSTM_CONFIG['num_samples_test']

    # Load the trained model
    if trained_model is None:
        # Fallback: load from disk if no model provided
        model = LSTMModel(input_size=5, hidden_size=20, num_layers=2, output_size=1)
        model.load_state_dict(torch.load(FILE_PATHS['project_root'] / f"{WATERSHED_CONFIG['watershed_id']}_best_model.pth"))        
        print("Loaded trained model from disk")
    else:
        # Use the provided trained model
        model = trained_model
        print("Using provided trained model")
    
    model.eval()  # Set to evaluation mode

    # --- LOAD DATA ---
    test_df = merged_data

    # --- CREATE SAMPLES ---
    X_list = []
    y_obs_list = []
    y_mod_list = []

    # Set random seed for reproducibility
    np.random.seed(42)

    # Calculate the maximum possible start index
    max_start_idx = len(test_df) - seq_len - 1

    print(f"Creating {num_samples_test} random samples...")
    print(f"Data length: {len(test_df)}")
    print(f"Sequence length: {seq_len}")
    print(f"Max possible start index: {max_start_idx}")


    for i in range(num_samples_test):
        start_idx = np.random.randint(0, max_start_idx + 1)
        end_idx = start_idx + seq_len
        y_idx = end_idx  # The day after the last X day

        # Make sure we don't go out of bounds
        if y_idx >= len(test_df):
            break

        # X: shape (365, 5)
        X_seq = test_df.iloc[start_idx:end_idx][['prcp(mm/day)', 'srad(W/m2)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']].values
        # y: shape (1,)
        y_obs = test_df.iloc[y_idx]['obs_flow']
        y_mod = test_df.iloc[y_idx]['mod_flow']

        X_list.append(X_seq)
        y_obs_list.append([y_obs])
        y_mod_list.append([y_mod])

    X_test = np.stack(X_list)  # shape (num_samples, 365, 5)
    y_obs = np.stack(y_obs_list)  # shape (num_samples, 1)
    y_mod = np.stack(y_mod_list)  # shape (num_samples, 1)
    y_mod = y_mod.squeeze() # shape (num_samples,) squeeze to 1D array (so that it can be used correctly in the nse function)

    print("X_test shape:", X_test.shape)
    print("y_obs shape:", y_obs.shape)
    print("y_mod shape:", y_mod.shape)

    # Create test dataset and dataloader
    test_dataset = HydroDataset(X_test, y_obs)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # No shuffling for testing

    # Make predictions with the trained model
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_obs = []
        
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            all_preds.append(y_pred.detach().cpu())
            all_obs.append(y_batch.detach().cpu())

        # Concatenate all batches
        y_pred_full = torch.cat(all_preds).squeeze().numpy()
        y_obs_full = torch.cat(all_obs).squeeze().numpy()

        # Denormalize predictions
        y_pred_denorm = y_pred_full * flow_std + flow_mean
        y_obs_denorm = y_obs_full * flow_std + flow_mean

        # Calculate NSE for test set (LSTM model)
        nse_test_pred = nse(y_pred_denorm, y_obs_denorm)
        #print(f"Test NSE: {nse_test_pred:.4f}")

        # Calculate NSE for test set (physical model)
        nse_test_mod = nse(y_mod, y_obs_denorm)
        #print(f"Test NSE: {nse_test_mod:.4f}")

        # Plot results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(2, 2, 1)
        plt.scatter(y_obs_denorm, y_pred_denorm, alpha=0.6)
        plt.plot([y_obs_denorm.min(), y_obs_denorm.max()], [y_obs_denorm.min(), y_obs_denorm.max()], 'r--')
        plt.xlabel('Observed Flow')
        plt.ylabel('Predicted Flow')
        plt.title(f'LSTM Predictions vs Observed (NSE: {nse_test_pred:.4f})')

        plt.subplot(2, 2, 2)
        plt.scatter(y_obs_denorm, y_mod, alpha=0.6)
        plt.plot([y_obs_denorm.min(), y_obs_denorm.max()], [y_obs_denorm.min(), y_obs_denorm.max()], 'r--')
        plt.xlabel('Observed Flow')
        plt.ylabel('Modeled Flow')
        plt.title(f'Modeled Flow vs Observed Flow (NSE: {nse_test_mod:.4f})')
        
        plt.subplot(2, 2, 3)
        plt.plot(y_obs_denorm[:100], label='Observed', alpha=0.7)
        plt.plot(y_pred_denorm[:100], label='Predicted', alpha=0.7)
        plt.xlabel('Time Steps')
        plt.ylabel('Flow')
        plt.title('Time Series Comparison (First 100 points)')

        plt.subplot(2, 2, 4)
        plt.plot(y_obs_denorm[:100], label='Observed', alpha=0.7)
        plt.plot(y_mod[:100], label='Model', alpha=0.7)
        plt.xlabel('Time Steps')
        plt.ylabel('Flow')
        plt.title('Time Series Comparison (First 100 points)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    return nse_test_mod, nse_test_pred

def parse_arguments():
    parser = argparse.ArgumentParser(description='LSTM Hydrological Forecasting')
    parser.add_argument('--watershed_id', type=str, default='01013500',
                       help='Watershed ID (default: 01013500)'),
    parser.add_argument('--HUC_number', type=str, default='01',
                       help='HUC number (default: 01)')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to configuration file (optional)')
    return parser.parse_args() 


if __name__ == "__main__":

    args = parse_arguments()

    # Update watershed configuration
    WATERSHED_CONFIG['watershed_id'] = args.watershed_id
    WATERSHED_CONFIG['HUC_number'] = args.HUC_number
    
    # Update file paths with new watershed ID
    FILE_PATHS['input_files']['meteorological'] = PROJECT_ROOT.parent / f"basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/{WATERSHED_CONFIG['HUC_number']}/{WATERSHED_CONFIG['watershed_id']}_lump_cida_forcing_leap.txt"
    FILE_PATHS['input_files']['flow'] = PROJECT_ROOT.parent / f"model_output_daymet/model_output/flow_timeseries/daymet/{WATERSHED_CONFIG['HUC_number']}/{WATERSHED_CONFIG['watershed_id']}_11_model_output.txt"


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

    # ============================================================================
    # STEP 3: SUMMARY
    # ============================================================================
    print("\n" + "=" * 40)
    print("STEP 3: WORKFLOW SUMMARY")
    print("=" * 40)

    print(f"\nWatershed Configuration:")
    print(f"- Watershed ID: {WATERSHED_CONFIG['watershed_id']}")
    print(f"- Description: {WATERSHED_CONFIG['description']}")
    
    print(f"\nTraining dataset:")
    print(f"- Watershed: {WATERSHED_CONFIG['watershed_id']}")  
    print(f"- Period: {DATA_PERIODS['training']['start'].date()} to {DATA_PERIODS['training']['end'].date()}")
    print(f"- Total days: {len(merged_data_train)}")
    print(f"- Output file: {get_output_file_path(dataset_type='training')}")
    
    print(f"\nTest dataset:")
    print(f"- Watershed: {WATERSHED_CONFIG['watershed_id']}") 
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
    
    # ============================================================================
    # STEP 4: LSTM TRAINING
    # ============================================================================
    print("\n" + "=" * 40)
    print(f"STEP 4: LSTM TRAINING {DATA_PERIODS['training']['description']}")
    print("=" * 40)

    flow_std_train = flow_std_train['OBS_RUN']
    flow_mean_train = flow_mean_train['OBS_RUN']

    best_model = LSTM_training(merged_data_train, flow_mean_train, flow_std_train)

    # ============================================================================
    # STEP 5: LSTM TESTING
    # ============================================================================
    print("\n" + "=" * 40)
    print(f"STEP 5: LSTM TESTING {DATA_PERIODS['test']['description']}")
    print("=" * 40)

    flow_std_test = flow_std_test['OBS_RUN']
    flow_mean_test = flow_mean_test['OBS_RUN']


    # Pass the trained model to testing function
    nse_test_mod, nse_test_pred = LSTM_testing(
        merged_data_test, 
        flow_mean_test, 
        flow_std_test, 
        trained_model=best_model  # Pass the best model here
    )

    print(f"\nFinal Test Results:")
    print(f"Test NSE (LSTM): {nse_test_pred:.4f}")
    print(f"Test NSE (Physical): {nse_test_mod:.4f}")


    
    
    
    

    