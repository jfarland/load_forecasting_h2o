import pandas as pd
import numpy as np
import math as math
import scipy.stats

import h2o
from h2o.estimators.gam import H2OGeneralizedAdditiveEstimator
h2o.init()

# Example Usage:
all_features = pd.read_csv("data/all_features.csv")
all_features.head()

all_features["target"] = all_features["kw"]
all_features = all_features.drop(["Unnamed: 0", "timestamp", "kw", "label", "cdh", "hdh", "thi"], axis = 1)

target_date = '2012-05-01'
train_data = all_features[all_features.date < target_date]
test_data = all_features[all_features.date == target_date]

print(f'shape of training data: {train_data.shape}')
print(f'shape of training data: {test_data.shape}')

# def create_features(df):

#     # create 24 hour displacement lags for 168 back
#     df["target_lag_168"] = df["target"].shift(168)
#     df["target_lag_192"] = df["target"].shift(168 + (1*24))
#     df["target_lag_216"] = df["target"].shift(168 + (2*24))
#     df["target_lag_240"] = df["target"].shift(168 + (3*24))
#     df["target_lag_264"] = df["target"].shift(168 + (4*24))
#     df["target_lag_288"] = df["target"].shift(168 + (5*24))
#     df["target_lag_312"] = df["target"].shift(168 + (6*24))

#     return df


def metrics(act, fcst):

    """
    Function to calculate forecasting and other relevant metrics from vectors of actuals and predictions

    :param int act: numeric vector the same length as `fcst`
    :param int fcst: numeric vector the same length as `act`
    """

    e = act - fcst

    naive = np.roll(act, -1)
    #naive[0] = np.NaN
    
    scale = sum(abs(act - naive)) / (len(e) - 1)

    sse = sum((fcst - np.mean(act))**2)
    sst = sum((act - np.mean(act))**2)
    
    # Use Pearson Correlation Coefficient
    rho = scipy.stats.pearsonr(act, fcst)
    #print(rho)

    #print(f'sum of squared errors: {sse.item()}')
    #print(f'sum of squared total: {sst.item()}')
    
    metrics = dict({
        "scale" : scale.item(),
        "n" : len(e),
        "ybar" : np.mean(act),
        "r2" : rho[0] ** 2,
        "rmse" : math.sqrt((sum(e**2) * (1 / len(e)))),
        "cvrmse" : math.sqrt((sum(e**2) * (1 / len(e)))) / np.mean(act),
        "nmbe" : (sum(e) * (1 / len(e))).item() / np.mean(act),
        "mape" : np.mean(abs(e / act)) * 100,
        "mae" : np.mean(abs(e)),
        "mase" : np.mean(abs(e / scale)) 
    })

    return metrics

print(metrics(np.random.rand(100), np.random.rand(100)))

def calc_load_research_stats(y):

    mean = np.mean(y)

    peak = np.max(y)

    load_factor = peak / mean

    stats = dict({
        'mean': mean,
        'peak': peak,
        'load_factor': load_factor
    })
    return stats

print(calc_load_research_stats(all_features["target"]))


class SimDayForecaster:

    """ Algorithm to create an average load profile based on historically similar days
    Step 1 - Score historical days based on their similarity to the target date
    Step 2 - Filter historical days based on (recency) criterion
    Step 3 - Sort and keep top N results
    Step 4 - Form average load profile based on historical conditions (see predict method)
    """

    def __init__(self, data, target_var, match_var):
        self.data = data
        self.target_var = target_var
        self.match_var = match_var

    def fit(self, target_date, top_n = 5, rec_n = 10):

        """
        Determine top_n most similar days out of the last rec_n

        :param str target_date: Date in which to match against.
        :param int top_n: Integer indicationg the number of top matched days to return.
        :param int rec_n: Integer indicating how many recent days to consider in the matching process.

        :returns A new DataFrame with matched dates and their (dis)similarity scores

        """

        data = self.data

        if target_date is None:
            target_date = datetime.today()

        # identify historical data
        hdata = data[data.date < target_date]

        #TODO: identify eligible baseline days 

        target_vector = data[data.date == target_date][self.match_var].values

        date_list = []
        score_list = []

        for candidate in hdata.date.unique():

            candidate_vector = hdata[hdata.date == candidate][self.match_var].values

            if len(candidate_vector) < 24:
                continue

            # calculate squared differences
            scores = (target_vector - candidate_vector) ** 2

            # Sum of Squared Differences
            score_sum = sum(scores)

            date_list.append(candidate)
            score_list.append(score_sum)

        rs = pd.DataFrame({"candidate": date_list, "scores": score_list})

        # sort, limit to last rec_n days
        rs = rs.sort_values(by = ["candidate"], ascending = False).head(rec_n)

        # sort, keep top_n results
        rs = rs.sort_values(by = ["scores"]).head(top_n)
        print(f'Showing matching results for {target_date}')
        self.match = rs

        return rs


    def predict(self):

        data = self.data
    
        try: 
            match = self.match
        except:
            print("No Historically Matched Days Found!")
            return
            

        df = data.set_index("date").join(match.set_index("candidate"))

        fcst = df.groupby("hour").agg({self.target_var: ['mean']})

        fcst.columns = ["fcst"]
        fcst.reset_index(inplace = True)
        self.fcst = fcst

        return fcst

model = SimDayForecaster(all_features, target_var = "target", match_var = "temperature")
print(model.fit(target_date = '2012-05-02'))
print(model.predict())

class BenchmarkLoadForecaster:

    """ Benchmark Load Forecasting Model

    Follows the semiparametric methodology of `https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=2256&context=theses`

    """
    def __init__(self, train_data):
        self.train_data = h2o.H2OFrame(train_data)

    def fit(self):

        X = self.train_data.drop(["target"], axis = 1).columns
        y = "target"

        gam_model = H2OGeneralizedAdditiveEstimator(family='gaussian',
                                            gam_columns=["temperature","humidity","visibility", "windSpeed"],
                                            model_id = "gam_model",
                                            standardize = True)


        gam_model.train(x=X, y=y, training_frame=self.train_data)
        self.model = gam_model
        return gam_model


    def predict(self, test_data):
        fcst = self.model.predict(h2o.H2OFrame(test_data))
        self.fcst = fcst
        return fcst

model = BenchmarkLoadForecaster(train_data)
print(model.fit())
print(model.predict(test_data))

h2o.cluster().shutdown

