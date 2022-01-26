# H2O Imports

from xml.sax.handler import all_features
from h2o_wave import main, app, Q, ui, data
from h2o_wave_ml import build_model, ModelType

import h2o
from h2o.estimators.gam import H2OGeneralizedAdditiveEstimator
h2o.init()

import pandas as pd
import numpy as np
import math as math
import scipy.stats
from datetime import datetime, timedelta
from typing import Optional

def get_form_items(value: Optional[str]):
    return [
        ui.text(f'date_trigger={value}'),
        ui.date_picker(name='date_trigger', label='Pick a date', trigger=True),    
    ]

def metrics(act, fcst):

    """
    Function to calculate forecasting and other relevant metrics from vectors of actuals and predictions

    :param int act: numeric vector the same length as `fcst`
    :param int fcst: numeric vector the same length as `act`
    """

    e = act - fcst
    naive = np.roll(act, -1)
    scale = sum(abs(act - naive)) / (len(e) - 1)

    sse = sum((fcst - np.mean(act))**2)
    sst = sum((act - np.mean(act))**2)
    
    # Use Pearson Correlation Coefficient
    rho = scipy.stats.pearsonr(act, fcst)
  
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

#print(metrics(np.random.rand(100), np.random.rand(100)))

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

#print(calc_load_research_stats(all_features["target"]))

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

        if len(target_vector) < 24:
            return 

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

# model = SimDayForecaster(all_features, target_var = "target", match_var = "temperature")
# print(model.fit(target_date = '2012-05-02'))
# print(model.predict())

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

# model = BenchmarkLoadForecaster(train_data)
# print(model.fit())
# print(model.predict(test_data))

@app('/load_forecasting')
async def serve(q: Q):

    if q.args.train:

        # copy inputs
        q.client.train_beg = datetime.strptime(q.args.train_beg, '%Y-%m-%d')
        q.client.train_end = datetime.strptime(q.args.train_end, '%Y-%m-%d')

        print(q.client.train_beg)
        print(q.client.train_end)

        # Limit training data based on date pickers
        ml_input = q.client.all_features[(q.client.all_features["timestamp"] >= q.args.train_beg) & (q.client.all_features["timestamp"] <= q.args.train_end)]
        print(q.client.all_features.shape)
        print(ml_input.shape)

        # train WaveML Model using H2O-3 AutoML
        q.client.wave_model = build_model(
            train_df = ml_input,
            target_column = 'kw',
            #feature_columns = ["hour", "LCLid", "temperature"],
            model_type = ModelType.H2O3,
            _h2o3_max_runtime_secs = 60*5,
            _h2o3_nfolds = 5,
            _h2o3_keep_cross_validation_predictions = True,
            _h2o3_keep_cross_validation_fold_assignment = True,
            _h2o3_stopping_metric = "RMSE",
            _h2o3_stopping_tolerance = 0.01,
            _h2o3_stopping_rounds = 5
        )
        
        model_id = q.client.wave_model.model.model_id
        mean_abs_err = round(q.client.wave_model.model.mae(), 2)

        # show training details and prediction option
        q.page['inputs'].items[1].buttons.items[1].button.disabled = False
        q.page['inputs'].items[2].message_bar.type = 'success'
        q.page['inputs'].items[2].message_bar.text = 'Training successfully completed!'
        q.page['inputs'].items[3].text.content = f'''**H2O AutoML model id:** {model_id} <br />
            **Accuracy:** {mean_abs_err}%'''
        q.page['inputs'].items[4].text.content = ''

    elif q.args.predict:

        q.client.fcst_beg = q.client.train_end + timedelta(days = 1)
        q.client.fcst_end = q.client.train_end + timedelta(days = 7)
        #q.client.fcst_dates = pd.date_range(start = q.client.fcst_beg, end = q.client.fcst_end)

        # filter forecast inputs
        test_data = q.client.all_features[(q.client.all_features["timestamp"] >= q.client.fcst_beg) & (q.client.all_features["timestamp"] <= q.client.fcst_end)]
        #print(test_data.shape)

        # Generate Benchmark Predictiions
        bench = h2o.load_model("./models/bench/gam_model")
        bench_fcst = bench.predict(h2o.H2OFrame(test_data)).as_data_frame()

        # Parse Output
        bench_out = pd.DataFrame()
        bench_out["timestamp"] = test_data["timestamp"].astype(str)
        bench_out["kw"] = bench_fcst["predict"].values
        bench_out["label"] = "benchmark" 

        # Similar Day Matching Baseline
        simday_fcst = []
        simday_time = []

        for d in test_data["date"].unique():
            model = SimDayForecaster(q.client.all_features, target_var = "kw", match_var = "temperature")
            model.fit(target_date = d)
            fcst = model.predict()

            #extract timestamp
            time = test_data[test_data["date"] == d].timestamp
            
            if len(time) < 24:
                continue

            simday_fcst.append(fcst)

            if simday_fcst is None:
                simday_fcst = np.repeat(np.Nan, [24], axis = 0)

            simday_time.append(time)

        # Parse Output
        simday_fcst = pd.concat(simday_fcst, axis = 0)
        simday_time = pd.concat(simday_time, axis = 0)

        simday_out = pd.DataFrame()
        simday_out["timestamp"] = simday_time.astype(str)
        simday_out["kw"] = simday_fcst["fcst"].values
        simday_out["label"] = "simday"

        # Generate H2O Load Forecasting Predictions
        h2o_fcst = pd.DataFrame(q.client.wave_model.predict(test_df = test_data))
        h2o_out = pd.DataFrame()
        h2o_out["timestamp"] = test_data["timestamp"].astype(str)
        h2o_out["kw"] = h2o_fcst.values
        h2o_out["label"] = "H2O"

        test_data["label"] = "actual"
        act_out = test_data[["kw", "label", "timestamp"]]
        act_out["timestamp"] = act_out["timestamp"].astype(str)
        act_out.columns = ['kw', 'label','timestamp']

        output = pd.concat([act_out, h2o_out, bench_out, simday_out])

        # Create data buffer using training data
        ts_plot_data = data('kw label timestamp', rows = [tuple(x) for x in output.to_numpy()])

        q.page['timeseries'] = ui.plot_card(box = ui.box('timeseries', height = '1000px'), 
           title = 'Interconnected Grid Demand',
           data = ts_plot_data, # TODO figure out the best way to manage memory here
           plot = ui.plot([
               ui.mark(type='point', x_scale='time',  x='=timestamp', y='=kw', color='=label', y_min=0, color_range="#000000 #3399FF #FF9966 #336600 ", y_title="kWh"),
               ui.mark(type='path', x_scale='time',  x='=timestamp', y='=kw', color='=label', y_min=0, color_range="#000000 #3399FF #FF9966 #336600", y_title="kWh")
               ])
           )

        # show predictions
        q.page['inputs'].items[2].message_bar.text = 'Prediction successfully completed!'

    else:

        # Load Full Data Set + remove dates without full AMI coverage
        all_features = pd.read_csv("data/all_features.csv")
        date_count = all_features.groupby("date")["kw"].count()
        good_dates = pd.DataFrame(date_count[date_count >= 24]).reset_index().drop("kw", axis = 1)
        all_features = all_features.set_index("date").join(good_dates.set_index("date"), how = "inner").reset_index()

        # Copy into client
        q.client.all_features = all_features
        q.client.all_features["timestamp"] = pd.to_datetime(q.client.all_features["timestamp"])
        q.client.all_features["date"] = pd.to_datetime(q.client.all_features["date"] , format = '%Y-%m-%d')

        # structure UI TODO: update flex layout to account for additional breakpoints (for various devices)
        q.page['meta'] = ui.meta_card(box='', layouts=[
            ui.layout(
                breakpoint='xl',
                width='1800px',
                zones=[
                    ui.zone('header'),
                    ui.zone('forecasting', direction = ui.ZoneDirection.ROW, zones = [
                        ui.zone('inputs', size = '20%'),
                        ui.zone('timeseries', size = '80%')
                    ]),
                    ui.zone('breakdown', direction = ui.ZoneDirection.ROW, zones = [
                        ui.zone('segmentation', size = '20%'),
                        ui.zone('mli', size = '50%'),
                        ui.zone('stats', size = '30%')
                    ])
                ]
            )
        ])

        # Icons
        #icon_path, = q.site.upload(['/Users/jfarland/Documents/products/wave/wave-apps/load_forecasting_h2o/telecommunications_Blog_2.png'])

        # Header Card
        q.page['header'] = ui.header_card(box=ui.box('header'), 
               title = 'Energy Data Science',
               subtitle = 'Load Forecasting for the Smart Grid',
               icon = 'ExploreData', # TODO: make this an h2o brand image
               icon_color='$blue'
        )

        # Modeling Input Cards
        q.page['inputs'] = ui.form_card(box=ui.box('inputs'), 
            items=[
                ui.text_l(content='H2O Load Forecaster'),
                ui.buttons(items=[
                    ui.button(name='train', label='Retrain', primary=True),
                    ui.button(name='predict', label='Predict', primary=True, disabled=True),
                ]),
                ui.date_picker(name="train_beg", label = "Training Begin", value = "2012-07-01"),
                ui.date_picker(name="train_end", label = "Training End", value = "2013-07-01"),
                ui.message_bar(type='warning', text='Training time limited to 5 minutes...'),
                ui.checklist(name='feature_selector', label='Choose explanatory variables...', choices = [
                    ui.choice(name='temp', label = "Drybulb Temperature (Celsius)"), 
                    ui.choice(name='hum', label = "Humidity (%)"), 
                    ui.choice(name='thi', label = "Temperature-Humidity Index"),
                    ui.choice(name='cdh', label = "Cooling-degree Hours (base 21)"),
                    ui.choice(name='hdh', label = "Heating-degree Hours (base 12)"),
                    ui.choice(name='vis', label = "Visibility (%)"),
                    ui.choice(name='ws', label = "Wind Speed (km/h)"),
                    ui.choice(name='wb', label = "Wind Bearing"),
                    ui.choice(name='lags', label = "Recency Lags"),
                    ui.choice(name='cal', label = "Calendar Effects")
                ], values=['temp', 'hum', 'vis', 'ws', 'wb', 'lags', 'cal']),
                ui.text(content=''),
                ui.text(content='')
        ])


        all_features_lim = q.client.all_features[["kw", "label", "timestamp"]]
        all_features_lim['timestamp'] = all_features_lim['timestamp'].astype(str)
        all_features_lim.columns = ['kw', 'label','timestamp']

        # Create data buffer using training data
        ts_plot_data = data('kw label timestamp', rows = [tuple(x) for x in all_features_lim.to_numpy()])
   
        # Time Series Plots TODO: this should show the time series training data, and predictions when predicted
        # Reference: https://wave.h2o.ai/docs/examples/plot-line-groups
        q.page['timeseries'] = ui.plot_card(box = ui.box('timeseries', height = '1000px'), 
            title = 'Distribution Feeder - Interconnected Demand',
            #data = data('kw label timestamp', rows = train_agg.head().to_dict("list")), # TODO figure out the best way to manage memory here
            data = ts_plot_data, # TODO figure out the best way to manage memory here
            plot = ui.plot([
                ui.mark(type='path', x_scale='time',  x='=timestamp', y='=kw', color='=label', y_min=0, color_range="#3399FF #FF9966", y_title="kWh")])
        )

        # Time Series Segmentation Card
        q.page['segmentation'] = ui.form_card(box = ui.box('segmentation'), items = [ui.text('Customer Segmentation')])

        # Model Interpretability / Diagnostic Card 
        q.page['mli'] = ui.form_card(box = ui.box('mli'), items = [ui.text('Modeling Diagnostics')])

        # Summary Statistics Card - TODO: turn this into a table https://wave.h2o.ai/docs/examples/table
        q.page['stats'] = ui.form_card(box = ui.box('stats'), items = [
            ui.text('Summary Stats')])

    await q.page.save()
