from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.metrics import make_scorer
from evaluation_utils import Evaluator

class Validator(object):
    """
    Valudates the dataset against the selected train/test splits and outputs the score.
    """
    def __init__(self, data, evaluation_size=9, splits=[65, 70, 75, 80], midpoint=3):
        self.evaluation_size = evaluation_size
        self.midpoint = midpoint
        self.splits = splits
        endog_mask = ~data.columns.isin([('day', ''), ('time', ''), ('morning', '')])
        exog_mask = data.columns.isin([('day', ''), ('time', ''), ('morning', '')])
        self.y = pv.iloc[:, endog_mask].to_numpy()
        self.X = pv.iloc[:, exog_mask].to_numpy()
        time = np.tile(data['time'][:TIMESTEPS_A_DAY], evaluation_size)
        self.X_predict = np.column_stack([
            np.repeat(np.arange(np.max(self.X[:,0]) + 1, np.max(self.X[:,0]) + evaluation_size + 1), TIMESTEPS_A_DAY),
            time,
            (time < 600) # magic constant
        ])

        self.evaluator = Evaluator('../input/test_solutions.csv')

    def result(self, model):
        model.fit(self.X, self.y)
        res = model.predict(self.X_predict)
        X_rep = np.repeat(self.X_predict, NUM_SYMBOLS, axis=0)
        df = pd.DataFrame(dict(
            date=(X_rep[:, 0] - max(self.X[:, 0]) - 1).astype(int),
            time=time_inverse_transform(X_rep[:, 1]),
            symbol=np.tile(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], (len(self.X_predict),)),
            open=res[:, :10].flatten()
        ))
        df['id'] = df['symbol'].astype(str) + '-' + df['date'].astype(str) + '-' + df['time'].astype(str)
        return df[['id', 'open']].set_index('id')

    def validate(self, model, avg=True):
        # Mean constant prediction for each symbol
        score = cross_validate(
            model, self.X, self.y,
            cv=IndexedTimeSeriesSplit(self.splits, 0, test_size=self.evaluation_size),
            scoring={'open_mse_start': make_scorer(
                         evaluation, norm=False,
                         greater_is_better=False,
                         subset=slice(0,self.midpoint*TIMESTEPS_A_DAY)
                     ),
                     'open_mse_end': make_scorer(
                         evaluation, norm=False,
                         greater_is_better=False,
                         subset=slice(self.midpoint*TIMESTEPS_A_DAY, self.evaluation_size*TIMESTEPS_A_DAY)
                     )}
        )
        if avg:
            res = self.result(model)
            eval_res = self.evaluator._evaluate_submission(res)
            retval = {k: np.mean(v) for k, v in score.items()}
            retval.update(eval_res)
            retval['model'] = model
            return retval
        else:
            return score
        
 
def evaluation(y, y_pred, subset=slice(None), norm=True):
    """
    Evaluation function as presented in the task promt.
    Basically it is MSE multiplied by the number of symbols
    """
    # mb to have less variation divide by total ss
    diff_sq = (y[subset, 2:12] - y_pred[subset,2:12]) ** 2
    
    if norm:
        return np.mean(diff_sq) * 10 / np.var(y[subset, 2:12])
    else:
        return np.mean(diff_sq) * 10