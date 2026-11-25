# Sequence Models (RNN, LSTM, GRU) and a Time-Series Forecasting App

A one-week deep learning project on sequential data: recurrent networks from scratch, the vanishing gradient, LSTM and GRU, a sequence pattern classifier, and a **monthly forecasting project with a Streamlit app**. Every neural-network result is compared with simple baselines, and the test set is used once.

## The app
`app.py` is a Streamlit app that forecasts a monthly series with a trained GRU and shows it next to simple baselines.
- **Forecast tab:** 1 to 12 months ahead, choose which models to show, download the forecast as CSV.
- **Backtest tab:** pick a starting month, compare every method with what really happened, and see the average error by horizon.
- **Upload your own CSV** (one row per month, a date column and a positive value column, at least 60 months): the app trains a linear model and a small GRU on your data without the last 12 months, and backtests on those 12 months.
- **About tab:** the results and limitations below.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Forecasting project: monthly airline passengers (1949 to 1960)
- **Data:** 144 monthly values with a clear trend and yearly seasonality. Split by time: train 1949 to 1956 (96 months), validation 1957 to 1958, test 1959 to 1960. Scaling uses training statistics only.
- **Baselines first:** naive (last month), seasonal naive (same month last year), seasonal naive with average growth, linear regression.
- **Problem found:** the validation and test values are higher than anything in the training data. A plain GRU on the scaled values predicted too low and was worse than every baseline.
- **Fix:** forecast the seasonal log-change `d_t = log(y_t) - log(y_(t-12))` and convert back with `y_t = y_(t-12) * exp(d_t)`. This keeps the numbers on a stable scale.
- **Model selection:** LSTM vs GRU, hidden size 16/32/64, 1 or 2 layers, 3 seeds each, chosen on validation MAE with early stopping. Winner: GRU, 16 units, 2 layers.

### Test results (one month ahead, thousands of passengers, lower is better)
| Model | Test MAE | Test RMSE | Test MAPE % |
|---|---|---|---|
| Naive (last month) | 44.21 | 51.78 | 9.73 |
| Seasonal naive | 47.58 | 49.99 | 10.52 |
| Seasonal naive x avg growth | 14.72 | 19.09 | 3.34 |
| Linear regression (levels) | 13.49 | 17.28 | 3.12 |
| Linear regression (on d) | 12.93 | 16.99 | 2.79 |
| Plain GRU on scaled values (mean of 3 seeds) | 53.48 | 68.57 | 10.86 |
| **GRU on d (mean of 3 seeds, range 12.8 to 13.6)** | **13.30** | 18.32 | 3.03 |

### Multi-step backtest on the test period (MAE by months ahead)
| Months ahead | GRU on d | Linear on d | Seasonal naive x avg growth | Seasonal naive | Naive |
|---|---|---|---|---|---|
| 1 | 12.78 | 12.93 | 14.72 | 47.58 | 44.21 |
| 2 | 16.80 | 15.01 | 14.09 | 48.78 | 71.52 |
| 3 | 19.26 | 17.89 | 13.72 | 49.91 | 91.55 |
| 4 | 20.14 | 19.64 | 13.97 | 50.19 | 106.67 |
| 5 | 18.15 | 22.31 | 14.54 | 50.30 | 110.40 |
| 6 | 18.02 | 25.68 | 15.08 | 49.95 | 108.63 |

### What I take from this (honest summary)
- The GRU clearly beats the naive and seasonal naive baselines, but it **only matches** a linear regression that uses the same transformation (13.3 vs 12.9, within the seed spread). Most of the improvement came from **changing the target**, not from the recurrent network.
- Beyond 1 month, the simplest baseline that needs no training (last year's value times the average growth) was the most accurate in this backtest.
- The GRU looked much better than linear on validation (about 7 vs 11.5) but not on test: scores on data used for tuning are optimistic.
- The test period has only 24 values and the multi-step forecasts overlap, so small differences are not reliable.
- Daily stock prices are close to a random walk, so the naive rule is very hard to beat there (shown with a simulation in the Day 5 notebook).

## Notebooks (Google Colab, CPU is enough)
| Day | Notebook | What it covers |
|---|---|---|
| 1 | `rnn_day1_sequences_and_rnn_cell.ipynb` | RNN cell by hand in NumPy, sliding windows, chronological split, first baselines |
| 2 | `rnn_day2_pytorch_rnn_vanishing_gradient.ipynb` | `nn.RNN`, measuring the vanishing and exploding gradient, a memory task where a plain RNN struggles |
| 3 | `rnn_day3_lstm_gru.ipynb` | LSTM and GRU, memory test with several seeds, parameter counts, LSTM gates over time |
| 4 | `rnn_day4_sequence_classifier.ipynb` | Sequence classification (sine, square, sawtooth, triangle, noise), confusion matrix, hidden-state PCA |
| 5 | `rnn_day5_forecasting_data_and_baselines.ipynb` | Airline data, trend and seasonality, leak-free windows, baselines, walk-forward evaluation |
| 6 | `rnn_day6_lstm_gru_forecaster.ipynb` | LSTM/GRU forecaster, seasonal log-change, model selection, test once, multi-step forecast |
| 7 | `app.py`, `forecast_utils.py` | Streamlit app and the reusable forecasting code |

## Other results
- **Vanishing gradient (Day 2):** for a 100-step sequence the gradient of the last hidden state with respect to the input fell from about 2.8e-01 at the last step to 1.9e-22 at the first step. With larger recurrent weights it grew to about 10 (exploding), which is why gradient clipping is used.
- **Memory test (Days 2 and 3):** the label is whether the first number of a random sequence is positive. Plain RNN results were unstable across seeds. At length 100, mean accuracy over 3 seeds: RNN 0.791, GRU 0.494, LSTM 0.951. Results depend strongly on learning rate and initialisation, so several seeds are needed.
- **Sequence classifier (Day 4):** a GRU chosen on validation (0.904) reached 0.918 test accuracy on five signal types. Square, sawtooth and noise were classified perfectly; the errors were between sine and triangle.

## Repository structure
```
app.py                    Streamlit app
forecast_utils.py         model, baselines, multi-step forecasting, training on uploaded data
forecaster.pt             trained GRU weights (from the Day 6 notebook)
forecaster_meta.json      model settings and scaling numbers
data/airline_passengers.csv
notebooks/                the six notebooks
requirements.txt
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repository.
2. Go to share.streamlit.io, sign in with GitHub, choose the repository, branch `main` and main file `app.py`, then deploy.
3. The first build takes a few minutes because PyTorch is installed (the CPU version is selected in `requirements.txt`).

## Limitations
- Only 144 monthly values, one series and a 24-month test period.
- The app assumes monthly data, a yearly pattern and positive values. A GRU trained on your own upload uses a single seed.
- A learning project, not a tool for business decisions.

## Sources
- Karpathy, The Unreasonable Effectiveness of Recurrent Neural Networks: https://karpathy.github.io/2015/05/21/rnn-effectiveness/
- Colah, Understanding LSTM Networks: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- PyTorch `nn.RNN`, `nn.LSTM`, `nn.GRU` documentation
- Airline passengers dataset (Box and Jenkins), via seaborn
