# ChaosESN
### Dependencies
- A .yml file is included for installing the conda environment this developed in.
- The code is written for GPUs in PyTorch 1.10. If your system has a single GPU you should set DEVICE=0.
- Python version is 3.8.12

### Model hyper-parameter search and refinement
- Performed by running any of the no_gamma files. For example, run the Z_no_gamma.ipynb notebook to find ESN's of various size N. 
- These models will store models and run statistics after each run in a dict.
- Change the structure of directories any way you like
- Currently (as cloned) the dicts are written to .json file after the all refinements for a size N are finished.
- Using Z as an example the .jsons are written to directory Dict for corresponding signal.
  So,<br> 
    Z_no_gamma_.ipynb writes its models and statistics to Dict/Z  and any numpy arrays to Dict/NpArrays.
                      
### Model output
- Plotting and calculating of results is done in any of the ouputs notebooks, e.g. outputsZ.ipynb
    
