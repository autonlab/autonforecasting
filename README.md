State-of-the-art time series forecasting for PyTorch. 

`AutonForecasting` is a Python library for time series forecasting with deep learning models. It includes *benchmark datasets*, *data-loading utilities*, *evaluation functions*, statistical *tests*, univariate model *benchmarks* and *SOTA models* implemented in PyTorch and PyTorchLightning. 

## Why Deep Learning on Time Series?
**Accuracy**:
- Global model is fitted simultaneously for several time series.
- Shared information helps with highly parametrized and flexible models.
- Useful for items/skus that have little to no history available.

**Efficiency:**
 - Automatic featurization processes.
 - Fast computations (GPU or TPU).


## Installation

Required dependencies are included in `environment.yml`.

##  Forecasting models

* [Neural Hierarchical Interpolation for Time Series Forecasting (N-HiTS)](https://arxiv.org/abs/2201.12886): A new model for long-horizon forecasting which incorporates novel hierarchical interpolation and multi-rate data sampling techniques to specialize blocks of its architecture to different frequency band of the time-series signal. It achieves SoTA performance on several benchmark datasets, outperforming current Transformer-based models by more than 25%. 

<p align="center">
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/indx_imgs/NHits.jpeg" width="300" title="N-HiTS" align="rigth">
</p>

* [Exponential Smoothing Recurrent Neural Network (ES-RNN)](https://www.sciencedirect.com/science/article/pii/S0169207019301153): A hybrid model that combines the expressivity of non linear models to capture the trends while it normalizes using a Holt-Winters inspired model for the levels and seasonals. This model is the winner of the M4 forecasting competition.

<p align="center">
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/indx_imgs/ESRNN.png" width="300" title="ES-RNN" align="rigth">
</p>

* [Neural Basis Expansion Analysis (N-BEATS)](https://arxiv.org/abs/1905.10437): A model from Element-AI (Yoshua Bengio’s lab) that has proven to achieve state-of-the-art performance on benchmark large scale forecasting datasets like Tourism, M3, and M4. The model is fast to train and has an interpretable configuration.

<p align="center">
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/indx_imgs/NBeats.png" width="300" title="N-BEATS" align="rigth">
</p>

* [Neural Basis Expansion Analysis with Exogenous Variables (N-BEATSx)](https://arxiv.org/abs/2104.05522): The neural basis expansion with exogenous variables is an extension to the original N-BEATS that allows it to include time dependent covariates.

<p align="center">
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/indx_imgs/NBEATSX.png" width="300" title="N-BEATSx" align="rigth">
</p>


* [Transformer-Based Models](https://arxiv.org/abs/1706.03762): Transformer-based framework for unsupervised representation learning of multivariate time series.
  - [Autoformer](https://arxiv.org/abs/2106.13008): Encoder-decoder model with decomposition capabilities and an approximation to attention based on Fourier transform.
  - [Informer](https://arxiv.org/abs/2012.07436): Transformer with MLP based multi-step prediction strategy, that approximates self-attention with sparsity.
  - [Transformer](): Classical vanilla Transformer.
