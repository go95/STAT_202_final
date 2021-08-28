import pandas as pd

NUM_SYMBOLS = 10
NUM_VARS = 5
TIMESTEPS_A_DAY = 5040

def time_inverse_transform(time):
    dt_val = pd.Series(pd.to_datetime('06:00:00') + pd.to_timedelta(time * 5, unit='s'))
    return dt_val.dt.strftime('%H:%M:%S')
 