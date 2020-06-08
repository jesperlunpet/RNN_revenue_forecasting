# Forecasting company key numbers from public data

Simple model used to forecast company key numbers from danish public annual reports. The neural network library used it Pytorch and the Jupyter Notebook contains instructions for both data scraping, data preparation, the model, training and testing. 

Additionally the final model from previous training is `model.pt` and it can be run using the simple `converter.py` by changing the first variable `x` to a sequence of annual reports

A data sample is already included as `shuffled_data.csv`, which means that the step of data scraping can be skipped in case the current method becomes depricated in the future

Run the notebook by using

```console
$ jupyter notebook
```
