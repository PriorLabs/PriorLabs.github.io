# Time Series Tutorial

TabPFN can be used for time series forecasting by framing it as a tabular regression problem. This tutorial demonstrates how to use the [TabPFN Time Series package](https://github.com/liam-sbhoo/tabpfn-time-series) for accurate zero-shot forecasting.
It was developed by Shi Bin Hoo, Samuel Müller, David Salinas and Frank Hutter.

## Quick Start

[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liam-sbhoo/tabpfn-time-series/blob/main/demo.ipynb)

First, install the package:

```bash
!git clone https://github.com/liam-sbhoo/tabpfn-time-series.git
!pip install -r tabpfn-time-series/requirements.txt
```

See the [demo notebook](
https://colab.research.google.com/github/liam-sbhoo/tabpfn-time-series/blob/main/demo.ipynb) for a complete example.

## How It Works

TabPFN performs time series forecasting by:

1. Converting time series data into a tabular format
2. Extracting temporal features (trends, seasonality, etc.)
3. Using TabPFN's regression capabilities for prediction
4. Converting predictions back to time series format

This approach provides several benefits:

- **Zero-shot forecasting**: No training required - just fit and predict
- **Both point and probabilistic forecasts**: Get confidence intervals with your predictions
- **Support for exogenous variables**: Easily incorporate external factors
- **Fast inference**: Uses tabpfn-client for GPU-accelerated predictions

## Additional Resources

- [GitHub Repository](https://github.com/liam-sbhoo/tabpfn-time-series)
- [Research Paper](https://openreview.net/forum?id=H02X7RO3OC#discussion)

## Getting Help

Join our [Discord community](https://discord.com/channels/1285598202732482621/) for support and discussions about TabPFN time series forecasting.