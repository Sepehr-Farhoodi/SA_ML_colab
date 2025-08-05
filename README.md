This workflow is designed to simulate/predict streamflow by implementing different deep learning algorithms.
It consists of three main sections: 
1. Configuration section to define each model's hyperparameters.
2. Class definition section to create training tensors and ML method modules.
3. Main function section consists of different functions with clear explanations of what they do.

This pipeline implements LSTM for the CAMELS watersheds and compares the LSTM results with the benchmark modeled results (SAC-SMA)

Clone it to your local and: cd LSTM_proj_collab : python LSTM_workflow.py --watershed-id (enter the watershed_id of ineterest) 
