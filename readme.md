# Forecasting Models

## 1. Naive Forecast (Random Walk / Persistence)

For forecast horizon \( h \ge 1 \), the naive forecast is:

\[
\hat{y}_{t+h \mid t} = y_t
\]

Equivalently, the one-step-ahead forecast is:

\[
\hat{y}_{t \mid t-1} = y_{t-1}
\]

This model serves as a simple baseline for highly persistent time series.

---

## 2. Linear Regression with Lag Features

Let the lag set be:

\[
\mathcal{L} = \{1, 3, 6, 12\}
\]

The regression model is:

\[
y_t = \beta_0 + \sum_{\ell \in \mathcal{L}} \beta_\ell y_{t-\ell} + \varepsilon_t
\]

where:

- \( y_t \): unemployment rate at month \( t \)
- \( y_{t-\ell} \): lagged unemployment rate
- \( \beta_0, \beta_\ell \): regression coefficients
- \( \varepsilon_t \): error term

The prediction equation is:

\[
\hat{y}_t = \hat{\beta}_0 + \sum_{\ell \in \mathcal{L}} \hat{\beta}_\ell y_{t-\ell}
\]

---

## 3. Elastic Net Regression

Elastic Net estimates coefficients by solving the following penalized least squares problem:

\[
\min_{\beta_0, \boldsymbol{\beta}}
\left\{
\frac{1}{n} \sum_{t=1}^{n}
\left(
y_t - \beta_0 - \mathbf{x}_t^\top \boldsymbol{\beta}
\right)^2
+ \lambda
\left(
\alpha \lVert \boldsymbol{\beta} \rVert_1
+ \frac{1-\alpha}{2} \lVert \boldsymbol{\beta} \rVert_2^2
\right)
\right\}
\]

where:

- \( \mathbf{x}_t = [y_{t-1}, y_{t-3}, y_{t-6}, y_{t-12}]^\top \)
- \( \lVert \boldsymbol{\beta} \rVert_1 = \sum_j |\beta_j| \): L1 penalty (sparsity)
- \( \lVert \boldsymbol{\beta} \rVert_2^2 = \sum_j \beta_j^2 \): L2 penalty (shrinkage)
- \( \lambda \ge 0 \): regularization strength
- \( \alpha \in [0,1] \): mixing parameter
  - \( \alpha = 1 \): Lasso
  - \( \alpha = 0 \): Ridge

---

## 4. ARIMA Model — ARIMA(1,1,1)

Let the differenced series be defined as:

\[
w_t = (1 - L)^d y_t
\]

where \( L \) is the lag operator:

\[
L y_t = y_{t-1}
\]

The general ARIMA(\(p,d,q\)) model is:

\[
\phi(L) w_t = c + \theta(L)\varepsilon_t
\]

with:

\[
\phi(L) = 1 - \phi_1 L - \cdots - \phi_p L^p
\]

\[
\theta(L) = 1 + \theta_1 L + \cdots + \theta_q L^q
\]

### ARIMA(1,1,1)

For the specific case \( (p,d,q) = (1,1,1) \):

\[
(1 - \phi_1 L)(1 - L)y_t = c + (1 + \theta_1 L)\varepsilon_t
\]

where:

- \( \phi_1 \): autoregressive coefficient
- \( \theta_1 \): moving average coefficient
- \( d = 1 \): first differencing
- \( c \): constant term
- \( \varepsilon_t \): white noise error

This model captures both autoregressive and moving average dynamics after differencing the series once to achieve stationarity.
