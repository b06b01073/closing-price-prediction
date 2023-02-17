# Closing Price Prediction

## Introduction 
This is a simplified implementation of the research paper [1]. In this implementation, only the ANN were used and tested.

## Dataset
The dataset are

| |Training Dataset | Testing Dataset|
|-----|-------------|--------------|
|Time Interval | 01/01/2011 - 12/31/2021| 01/01/2022 - 02/17/2023|

Note that the first 13 days of both training set and testing set are dropped since there are no 14 days MA data.

The data is collected from yahoo finance by yfinance.

## Model 
An simple network architecture is listed below:
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                 [-1, 1024]          12,288
              ReLU-2                 [-1, 1024]               0
            Linear-3                  [-1, 256]         262,400
              ReLU-4                  [-1, 256]               0
            Linear-5                   [-1, 32]           8,224
              ReLU-6                   [-1, 32]               0
            Linear-7                    [-1, 1]              33
================================================================
Total params: 282,945
Trainable params: 282,945
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.02
Params size (MB): 1.08
Estimated Total Size (MB): 1.10
----------------------------------------------------------------

```

The 11 input features are:
* Variables mentioned in [1] (6 in total)
* OHLC of previous day (4 in total)
* 3 days MA



## Results

### RMSE and MAPE
|Company or Index|RMSE|MAPE|
|------|-----|----|
|0050|1.67|1.09%|
|Nike|2.89|1.93%|
|Boeing|4.53|2.12%|
|Apple|3.25|1.72%|
|Amazon|3.96|2.48%|

It is by no means a outstanding result in my opinion, but the fact that the model does have the capability to caputure trend with such an simple network is quite suprising.

### 0050
<p align="center">
  <img src="./result/0050.TW.png">
</p>

### Apple
<p align="center">
  <img src="./result/AAPL.png">
</p>

### Amazon
<p align="center">
  <img src="./result/AMZN.png">
</p>

### The Boeing Company 
<p align="center">
  <img src="./result/BA.png">
</p>

### Nike
<p align="center">
  <img src="./result/NKE.png">
</p>


## Reference

[1] Mehar Vijh, Deeksha Chandola, Vinay Anandikkiwal, Arun Kumar, Stock Closing Price Prediction using Machine Learning Techniques, Procedia Computer Science,
Volume 167, 2020, Pages 599-606, ISSN 1877-0509, https://doi.org/10.1016/j.procs.2020.03.326.
