import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import yfinance as yf

vix_data = yf.download('^VIX', start='2019-01-01', end='2024-11-01', interval='1d')
vix=vix_data['Adj Close']


#print(vix)

plt.figure(figsize=(10, 6))
plt.plot(vix)
plt.title('VIX')
plt.xlabel('Date')
plt.ylabel('VIX')
plt.show()

# Create a GARCH(1,1) model
print("nowhere?")
model = arch_model(vix, vol='GARCH', p=1, q=1, dist='normal')
print("there")
# Fit the model
model_fit = model.fit(disp='off')
print("here")
print(model_fit.summary())


# Forecast the next 5 periods
forecast_horizon = 5
forecasts = model_fit.forecast(horizon=forecast_horizon)

# Extract the forecasted variances
variance_forecasts = forecasts.variance[-1:]
print("Forecasted Variances:\n", variance_forecasts)

