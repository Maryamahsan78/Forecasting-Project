"""Streamlit app: monthly forecasting with a GRU, compared honestly with simple baselines.
Run:  streamlit run app.py
"""
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import forecast_utils as fu

st.set_page_config(page_title='Monthly Forecaster: GRU vs baselines', layout='wide')
COLORS = {'GRU (on d)': '#d62728', 'Linear regression (on d)': '#1f77b4', 'Seasonal naive x avg growth': '#2ca02c', 'Seasonal naive': '#ff7f0e', 'Naive (last month)': '#7f7f7f'}


@st.cache_resource(show_spinner='Loading the trained GRU...')
def get_airline():
    s = fu.load_airline(); return s, fu.load_airline_bundle(s.values)


@st.cache_resource(show_spinner='Training on your data (about 10 to 30 seconds)...')
def train_on_upload(values_bytes, train_gru):
    return fu.fit_bundle_from_series(np.frombuffer(values_bytes, dtype=float), train_gru=train_gru)


def plot(history, fcs, future_index, title, actual=None, history_months=36):
    fig, ax = plt.subplots(figsize=(10, 4))
    h = history.iloc[-history_months:]; ax.plot(h.index, h.values, 'k', lw=2, label='history')
    if actual is not None: ax.plot(actual.index, actual.values, 'k--o', lw=2, ms=4, label='actual')
    for name, p in fcs.items(): ax.plot(future_index, p, marker='o', ms=4, color=COLORS[name], label=name)
    ax.set_title(title); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.3); fig.autofmt_xdate(); fig.tight_layout(); return fig


# ------------------------------------------------------------------ sidebar
st.sidebar.header('Data')
source = st.sidebar.radio('Series', ['Airline passengers (built in)', 'Upload my own CSV'])
H = st.sidebar.slider('Forecast horizon (months)', 1, 12, 6)
series, bundle, first_origin, backtest_note = None, None, None, ''

if source.startswith('Airline'):
    series, bundle = get_airline(); first_origin = 120
    backtest_note = 'The GRU was trained on 1949-1956 and tuned on 1957-1958. The backtest origins (1959 and 1960) were never used for training or tuning.'
else:
    up = st.sidebar.file_uploader('CSV with a date column and a value column (one row per month)', type=['csv'])
    if up is None:
        st.title('Monthly Forecaster'); st.info('Upload a monthly CSV in the sidebar (at least 60 months, positive values), or switch to the built-in airline data.'); st.stop()
    raw = pd.read_csv(up)
    if raw.shape[1] < 2: st.error('The CSV needs at least two columns (date and value).'); st.stop()
    date_col = st.sidebar.selectbox('Date column', raw.columns, index=0); value_col = st.sidebar.selectbox('Value column', [c for c in raw.columns if c != date_col], index=0)
    try: series = fu.parse_series(raw, date_col, value_col)
    except ValueError as e: st.error(str(e)); st.stop()
    if len(series) < 60: st.error(f'Only {len(series)} months found, 60 are needed.'); st.stop()
    use_gru = st.sidebar.checkbox('Also train a GRU (slower)', value=True)
    key = hashlib.md5(series.values.tobytes()).hexdigest() + str(use_gru)
    if st.sidebar.button('Train models on my data') or st.session_state.get('key') == key:
        bundle = train_on_upload(series.values.astype(float).tobytes(), use_gru); st.session_state['key'] = key
    else:
        st.title('Monthly Forecaster'); st.line_chart(series); st.info('Check the data above, then press **Train models on my data** in the sidebar.'); st.stop()
    first_origin = len(series) - 12
    backtest_note = 'The models were trained without the last 12 months. The backtest origins are inside that held-out year.'

st.title('Monthly Forecaster: GRU vs simple baselines')
st.caption('The network forecasts the seasonal log-change (growth versus the same month last year), then converts it back to the original units. Every neural result is shown next to simple baselines on purpose.')
st.sidebar.caption(bundle.info)
tab1, tab2, tab3 = st.tabs(['Forecast', 'Backtest (how good is it?)', 'About and results'])

# ------------------------------------------------------------------ tab 1: forecast the future
with tab1:
    chosen = st.multiselect('Models to show', bundle.available(), default=[m for m in ['GRU (on d)', 'Linear regression (on d)', 'Seasonal naive x avg growth'] if m in bundle.available()], key='m1')
    fut = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=H, freq='MS')
    fcs = bundle.forecast(series.values, H); fcs = {k: v for k, v in fcs.items() if k in chosen}
    if fcs:
        st.pyplot(plot(series, fcs, fut, f'{H}-month forecast after {series.index[-1]:%B %Y}'))
        table = pd.DataFrame(fcs, index=fut.strftime('%b %Y')).round(1); st.dataframe(table)
        st.download_button('Download forecast as CSV', table.to_csv().encode(), 'forecast.csv', 'text/csv')
    else: st.warning('Select at least one model.')
    st.warning('A forecast of the future cannot be checked yet. Use the **Backtest** tab to see how accurate each method was on months it has not seen. The longer the horizon, the less reliable the forecast.')

# ------------------------------------------------------------------ tab 2: backtest
with tab2:
    st.write(backtest_note)
    origins = list(range(first_origin, len(series) - 1)); labels = [f'{series.index[o]:%b %Y}' for o in origins]
    c1, c2 = st.columns([1, 2]); start = c1.selectbox('Forecast starts at', labels, index=0); o = origins[labels.index(start)]
    Hb = min(H, len(series) - o); fb = bundle.forecast(series.values[:o], Hb); idx = series.index[o:o + Hb]; actual = series.iloc[o:o + Hb]
    shown = [m for m in bundle.available() if m in ['GRU (on d)', 'Linear regression (on d)', 'Seasonal naive x avg growth', 'Seasonal naive', 'Naive (last month)']]
    st.pyplot(plot(series.iloc[:o], {k: fb[k] for k in shown}, idx, f'Forecast made at {start} versus what really happened ({Hb} months)', actual=actual))
    mae = pd.DataFrame({'MAE for this one forecast': {k: float(np.abs(actual.values - fb[k]).mean()) for k in shown}}).sort_values('MAE for this one forecast'); c2.dataframe(mae.round(1))
    st.subheader('Average error by horizon (all backtest starting points)')
    tab = bundle.backtest_table(series.values, first_origin, 6); st.dataframe(tab.round(2))
    st.caption('Mean absolute error in the original units. The longer horizons average fewer forecasts (last column), and the forecasts overlap, so small differences between methods are not reliable.')

# ------------------------------------------------------------------ tab 3: about
with tab3:
    st.markdown("""
### What this project shows
- **Baselines first.** Naive, seasonal naive, seasonal naive with average growth and a linear model are always shown next to the GRU.
- **A plain neural network struggled.** On the raw airline numbers the test values are higher than anything in training, and a plain GRU was far worse than the baselines (test MAE about 53 passengers).
- **Changing the target helped.** Forecasting the seasonal log-change (`log(y_t) - log(y_(t-12))`) put the numbers on a stable scale.
- **The honest result.** On the airline test period (24 months) the GRU reached a test MAE of about 13.3 passengers. That clearly beats naive (44.2) and seasonal naive (47.6), but it only **matches** a linear regression that uses the same transformation (12.9). Most of the gain came from the transformation, not from the recurrent network.
- **Multi-step backtest.** Looking 2 to 6 months ahead, the simplest baseline that needs no training (last year's value times the average growth) had the lowest error (about 14 passengers), better than the GRU (about 17 to 20) and the linear model. The GRU was best only at 1 month, by a very small margin.

### Test results (airline data, one-step-ahead, thousands of passengers)""")
    st.table(pd.DataFrame({'Test MAE': [44.21, 47.58, 14.72, 13.49, 12.93, 53.48, 13.30]}, index=['Naive (last month)', 'Seasonal naive', 'Seasonal naive x avg growth', 'Linear regression (levels)', 'Linear regression (on d)', 'Plain GRU on scaled values', 'GRU on d (mean of 3 seeds)']))
    st.markdown("""
### Limitations
- Only 144 monthly values, and a test period of just 24 months: small differences between models are noise.
- One series, one clean pattern. Results will not transfer automatically to other data. Daily stock prices are close to a random walk and are not a good target for this kind of model.
- The model assumes monthly data with a yearly pattern and positive values.
- Forecasts further than a few months ahead are much less reliable. This app is a learning project, not a tool for business decisions.
""")
