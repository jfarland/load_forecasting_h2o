import h2o
from h2o.estimators.gam import H2OGeneralizedAdditiveEstimator

# Initialize h2o cluster for use with benchmark GAM model
# TODO: determine if a persisted h2o 3 model is the best way to do this.
#h2o.init()

from h2o_wave import Q, app, handle_on, main, on, ui, data, copy_expando
from h2o_wave_utils import wave_utils_handle_error, wave_utils_table_from_df, wave_utils_cards_to_clean
from h2o_wave_utils import *
from h2o_wave_ml import build_model, ModelType

import driverlessai
import h2osteam  
from h2osteam.clients import DriverlessClient

### IMPORT LOAD FORECASTING SPECIFIC UTILITIES ###
#from load_forecasting_utils import *

from xml.sax.handler import all_features

import pandas as pd
import numpy as np
import math as math
import scipy.stats
from datetime import datetime, timedelta
from typing import Optional

from tqdm import tqdm

h2o.init()

steam_url = "https://steam.cloud-internal.h2o.ai"
steam_user = "jon.farland@h2o.ai"
steam_pw = "pat_50intc1tsv28vqngd4xut6posmsm6nzwa3m1"

# Model Endpoints
model_dict = {
    "load_forecaster": "https://model.cloud-internal.h2o.ai/b526c0a6-fa87-446c-adff-d1124c8ed5c0/model",
    "solar_forecaster": "https://model.cloud-internal.h2o.ai/5a772bb2-08a2-4aaf-9c89-0018daccb0bf/model"
}


def init_app(q: Q):

    #############################################
    ### LOAD ALL DATA THAT SHIPS WITH THE APP ###
    #############################################

    # Load historical transactional and prepared data for daily and hourly frequencies
    q.client.energy_df = pd.read_csv("data/all_features.csv")
    q.client.energy_df["label"] = "historical"

    q.client.pv_df = pd.read_csv("data/pvwatts_hourly.csv").head(8759)
    q.client.pv_df["label"] = "historical"

    #q.client.hourly_data = pd.read_csv("../data/prepared-data/hour_output.csv")
    #q.client.hourly_data["label"] = "historical"

    #q.client.all_features["timestamp"] = pd.to_datetime(q.client.all_features["timestamp"])
    #q.client.energy_df["date"] = pd.to_datetime(q.client.energy_df["timestamp"] , format = '%Y-%m-%d %H')
    #q.client.hourly_data["time"] = pd.to_datetime(q.client.hourly_data["time"] , format = '%Y-%m-%d %H')

    # Load Full Data Set + remove dates without full AMI coverage
    all_features = pd.read_csv("data/all_features.csv")
    date_count = all_features.groupby("date")["kw"].count()
    good_dates = pd.DataFrame(date_count[date_count >= 24]).reset_index().drop("kw", axis = 1)
    all_features = all_features.set_index("date").join(good_dates.set_index("date"), how = "inner").reset_index()

    # Copy into client
    q.client.all_features = all_features
    q.client.all_features["timestamp"] = pd.to_datetime(q.client.all_features["timestamp"])
    q.client.all_features["date"] = pd.to_datetime(q.client.all_features["date"] , format = '%Y-%m-%d')

    # Copy into client
    #
    q.client.pv_df["timestamp"] = pd.to_datetime(q.client.all_features["timestamp"])
    #q.client.pv_df["date"] = pd.to_datetime(q.client.all_features["date"] , format = '%Y-%m-%d')


    # layout
    q.page['meta'] = ui.meta_card(box='', layouts=[
        ui.layout(
            breakpoint='xl',
            width='1800px',
            zones=[
                ui.zone('header'),
                ui.zone('menu', direction = ui.ZoneDirection.ROW),
                ui.zone('forecasting', direction = ui.ZoneDirection.ROW, zones = [
                    ui.zone('inputs', size = '20%'),
                    ui.zone('timeseries', size = '70%')
                ]),
                ui.zone('footer')
            ]
        ),
        ui.layout(
            breakpoint="l",
            width="1800px",
            zones=[
                ui.zone('header'),
                ui.zone('menu', direction = ui.ZoneDirection.ROW),
                ui.zone('forecasting', direction = ui.ZoneDirection.ROW, zones = [
                    ui.zone('inputs', size = '20%'),
                    ui.zone('timeseries', size = '70%')
                ]),
                ui.zone('footer')
            ]
        )],
        theme='h2o-dark' # https://wave.h2o.ai/docs/color-theming
    ) 

    q.page["menu"] = ui.tab_card(
        box=ui.box("menu", height="50px"),
        items=[
            ui.tab(name="#home", label="Home", icon="Home"),
            ui.tab(name="#load", label="Electrical Load", icon="ExploreData"),
            ui.tab(name="#solar", label="Solar Generation", icon="ExploreData"),
            ui.tab(name="#data", label="Data", icon="Table")
        ],
    )
    
    #TODO: add logo
    #logo_url = 'https://static4.arrow.com/-/media/arrow/images/arrow-logo/logo-white.svg'

    # Header Card
    q.page['header'] = ui.header_card(
        box=ui.box('header'), 
        title = 'Energy Data Science',
        subtitle = 'Load Forecasting for the Smart Grid',
        icon = 'ExploreData', # TODO: make this an h2o brand image
        icon_color='$blue')

    q.page["footer"] = ui.footer_card(
        box="footer", caption="(c) 2022 H2O.ai. All rights reserved."
    )

@on("#home")
async def main_menu(q: Q):

    # Icons
    #icon_path, = q.site.upload(['/Users/jfarland/Documents/products/wave/wave-apps/load_forecasting_h2o/telecommunications_Blog_2.png'])

    # Intro / Landing Page
    intro_md = '''Welcome to the H2O Load Forecasting Application!'''     

    q.page['inputs'] = ui.markdown_card(box=ui.box('inputs'), 
        title = "Welcome",
        content = intro_md)

    q.page["timeseries"] = ui.form_card(
        box = ui.box("timeseries"), 
        items = [
            ui.text('<img src="https://imgur.com/2iqFu0X.png" width="1200">')
            ]
    )

    q.page["menu"] = ui.tab_card(
        box=ui.box("menu", height="50px"),
        value = "#home",
        items=[
            ui.tab(name="#home", label="Home", icon="Home"),
            ui.tab(name="#load", label="Electrical Load", icon="ExploreData"),
            ui.tab(name="#solar", label="Solar Generation", icon="ExploreData"),
            ui.tab(name="#data", label="Data", icon="Table")
        ],
    )

@on("#data")
async def data_menu(q: Q):  

    q.page["menu"] = ui.tab_card(
        box=ui.box("menu", height="50px"),
        value = "#data",
        items=[
            ui.tab(name="#home", label="Home", icon="Home"),
            ui.tab(name="#load", label="Electrical Load", icon="ExploreData"),
            ui.tab(name="#solar", label="Solar Generation", icon="ExploreData"),
            ui.tab(name="#data", label="Data", icon="Table")
        ],
    )

    q.page["inputs"] = ui.form_card(box = ui.box("inputs"), 
        items = [
            ui.text_xl("Manage Your Data"),
            ui.file_upload(name='user_files', label="Upload", multiple=True)
        ])

@on("#load")
async def elec_load_menu(q: Q):

    ENDPOINT = model_dict["load_forecaster"]

    q.page["menu"] = ui.tab_card(
        box=ui.box("menu", height="50px"),
        value = "#load",
        items=[
            ui.tab(name="#home", label="Home", icon="Home"),
            ui.tab(name="#load", label="Electrical Load", icon="ExploreData"),
            ui.tab(name="#solar", label="Solar Generation", icon="ExploreData"),
            ui.tab(name="#data", label="Data", icon="Table")
        ],
    )

    if q.args.predict:

        #q.client.fcst_beg = q.client.train_end + timedelta(days = 1)
        #q.client.fcst_end = q.client.train_end + timedelta(days = 7)
        #q.client.fcst_dates = pd.date_range(start = q.client.fcst_beg, end = q.client.fcst_end)
        #print(f'FORECAST BEG: {q.client.fcst_beg}')
        #print(f'FORECAST END: {q.client.fcst_end}')

        q.client.fcst_beg = pd.to_datetime("2013-07-02 00:00:00")
        q.client.fcst_end = pd.to_datetime("2013-07-08 00:00:00")

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
        sample_response = mlops_get_schema(ENDPOINT)

        fields = sample_response["fields"]
        
        scoring_url = ENDPOINT + "/score"
        print(f'TESTING URL: {scoring_url}')

        input = test_data[fields].fillna(0)
        batch_size = (6 * 24) + 1

        # TODO: add logic here that looks at what kind of model is being called.
        h2o_fcst = score(input, endpoint = scoring_url, batch_size = batch_size, fields = fields)

        #h2o_fcst = pd.DataFrame(q.client.wave_model.predict(test_df = test_data))
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
                ui.mark(type='point', x_scale='time',  x='=timestamp', y='=kw', color='=label', y_min=0, color_range="#CCFFFF #3399FF #FF9966 #336600 ", y_title="kWh"),
                ui.mark(type='path', x_scale='time',  x='=timestamp', y='=kw', color='=label', y_min=0, color_range="#CCFFFF #3399FF #FF9966 #336600", y_title="kWh")
                ])
            )

        # show predictions
        q.page['inputs'].items[2].message_bar.text = 'Prediction successfully completed!'

    elif q.args.train:
        # copy inputs
        q.client.train_beg = datetime.strptime(q.args.train_beg, '%Y-%m-%d')
        q.client.train_end = datetime.strptime(q.args.train_end, '%Y-%m-%d')

        # Limit training data based on date pickers
        ml_input = q.client.all_features[(q.client.all_features["timestamp"] >= q.args.train_beg) & (q.client.all_features["timestamp"] <= q.args.train_end)]

        #TODO: change this for a DAI interface

        # train WaveML Model using H2O-3 AutoML
        q.client.wave_model = build_model(
            train_df = ml_input,
            target_column = 'kw',
            #feature_columns = ["hour", "LCLid", "temperature"],
            model_type = ModelType.H2O3,
            _h2o3_max_runtime_secs = 60*1,
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

    else: 

        q.page['inputs'] = ui.form_card(box=ui.box('inputs'), 
            items=[
                ui.text_l(content='H2O Load Forecaster'),
                ui.buttons(items=[
                    ui.button(name='train', label='Retrain', primary=True),
                    ui.button(name='predict', label='Predict', primary=True),
                ]),
                ui.date_picker(name="train_beg", label = "Training Begin", value = "2012-07-01"),
                ui.date_picker(name="train_end", label = "Training End", value = "2013-07-01"),
                ui.message_bar(type='warning', text='Training time limited to 60 seconds...'),
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
            data = ts_plot_data, # TODO figure out the best way to manage memory here
            plot = ui.plot([
                ui.mark(type='path', x_scale='time',  x='=timestamp', y='=kw', color='=label', y_min=0, color_range="#3399FF #FF9966", y_title="kWh")])
        )

@on("#solar")
async def solar_load_menu(q: Q):

    ENDPOINT = model_dict["solar_forecaster"]

    q.page["menu"] = ui.tab_card(
        box=ui.box("menu", height="50px"),
        value = "#solar",
        items=[
            ui.tab(name="#home", label="Home", icon="Home"),
            ui.tab(name="#load", label="Electrical Load", icon="ExploreData"),
            ui.tab(name="#solar", label="Solar Generation", icon="ExploreData"),
            ui.tab(name="#data", label="Data", icon="Table")
        ],
    )

    if q.args.predict:

        #q.client.fcst_beg = q.client.train_end + timedelta(days = 1)
        #q.client.fcst_end = q.client.train_end + timedelta(days = 7)
        #q.client.fcst_dates = pd.date_range(start = q.client.fcst_beg, end = q.client.fcst_end)
        #print(f'FORECAST BEG: {q.client.fcst_beg}')
        #print(f'FORECAST END: {q.client.fcst_end}')

        #q.client.fcst_beg = pd.to_datetime("2021-07-01 00:00:00")
        #q.client.fcst_end = pd.to_datetime("2021-07-07 00:00:00")

        # filter forecast inputs
        #test_data = q.client.pv_df[(q.client.pv_df["timestamp"] >= q.client.fcst_beg) & (q.client.pv_df["timestamp"] <= q.client.fcst_end)]
        test_data = q.client.pv_df.tail(168)

        print(f'SOLAR TEST DATA SHAPE: {test_data.shape}')
        #print(test_data.shape)

        # Generate Benchmark Predictiions
        #bench = h2o.load_model("./models/bench/gam_model")
        #bench_fcst = bench.predict(h2o.H2OFrame(test_data)).as_data_frame()

        # Parse Output
        #bench_out = pd.DataFrame()
        #bench_out["timestamp"] = test_data["timestamp"].astype(str)
        #bench_out["kw"] = bench_fcst["predict"].values
        #bench_out["label"] = "benchmark" 

        # Similar Day Matching Baseline
        # simday_fcst = []
        # simday_time = []

        # for d in test_data["date"].unique():
        #     model = SimDayForecaster(q.client.all_features, target_var = "kw", match_var = "temperature")
        #     model.fit(target_date = d)
        #     fcst = model.predict()

        #     #extract timestamp
        #     time = test_data[test_data["date"] == d].timestamp
            
        #     if len(time) < 24:
        #         continue

        #     simday_fcst.append(fcst)

        #     if simday_fcst is None:
        #         simday_fcst = np.repeat(np.Nan, [24], axis = 0)

        #     simday_time.append(time)

        # # Parse Output
        # simday_fcst = pd.concat(simday_fcst, axis = 0)
        # simday_time = pd.concat(simday_time, axis = 0)

        # simday_out = pd.DataFrame()
        # simday_out["timestamp"] = simday_time.astype(str)
        # simday_out["kw"] = simday_fcst["fcst"].values
        # simday_out["label"] = "simday"

        # Generate H2O Load Forecasting Predictions
        sample_response = mlops_get_schema(ENDPOINT)

        fields = sample_response["fields"]
        
        scoring_url = ENDPOINT + "/score"
        print(f'TESTING URL: {scoring_url}')

        input = test_data[fields].fillna(0)
        batch_size = 168

        # TODO: add logic here that looks at what kind of model is being called.
        h2o_fcst = score(input, endpoint = scoring_url, batch_size = batch_size, fields = fields)

        #h2o_fcst = pd.DataFrame(q.client.wave_model.predict(test_df = test_data))
        h2o_out = pd.DataFrame()
        h2o_out["timestamp"] = test_data["timestamp"].astype(str)
        h2o_out["watts"] = h2o_fcst.values
        h2o_out["label"] = "H2O"

        test_data["label"] = "actual"
        act_out = test_data[["AC System Output (W)", "label", "timestamp"]]
        act_out["timestamp"] = act_out["timestamp"].astype(str)
        act_out.columns = ['watts', 'label','timestamp']

        print(f'SUMMARY OF SOLAR ACTUAL: {act_out.head(168)}')


        #output = pd.concat([act_out, h2o_out, bench_out, simday_out])
        output = pd.concat([act_out, h2o_out])

        # Create data buffer using training data
        ts_plot_data = data('watts label timestamp', rows = [tuple(x) for x in output.to_numpy()])

        q.page['timeseries'] = ui.plot_card(box = ui.box('timeseries', height = '1000px'), 
            title = 'Behind the Meter Solar Generation',
            data = ts_plot_data, # TODO figure out the best way to manage memory here
            plot = ui.plot([
                ui.mark(type='point', x_scale='time',  x='=timestamp', y='=watts', color='=label', y_min=0, color_range="#CCFFFF #3399FF #FF9966 #336600 ", y_title="Watts (AC)"),
                ui.mark(type='path', x_scale='time',  x='=timestamp', y='=watts', color='=label', y_min=0, color_range="#CCFFFF #3399FF #FF9966 #336600", y_title="Watts (AC)")
                ])
            )

        # show predictions
        q.page['inputs'].items[2].message_bar.text = 'Prediction successfully completed!'

    elif q.args.train:
        # copy inputs
        q.client.train_beg = datetime.strptime(q.args.train_beg, '%Y-%m-%d')
        q.client.train_end = datetime.strptime(q.args.train_end, '%Y-%m-%d')

        # Limit training data based on date pickers
        ml_input = q.client.all_features[(q.client.all_features["timestamp"] >= q.args.train_beg) & (q.client.all_features["timestamp"] <= q.args.train_end)]

        #TODO: change this for a DAI interface

        # train WaveML Model using H2O-3 AutoML
        q.client.wave_model = build_model(
            train_df = ml_input,
            target_column = 'kw',
            #feature_columns = ["hour", "LCLid", "temperature"],
            model_type = ModelType.H2O3,
            _h2o3_max_runtime_secs = 60*1,
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

    else: 

        q.page['inputs'] = ui.form_card(box=ui.box('inputs'), 
            items=[
                ui.text_l(content='H2O Solar Forecaster'),
                ui.buttons(items=[
                    ui.button(name='train', label='Retrain', primary=True),
                    ui.button(name='predict', label='Predict', primary=True),
                ]),
                ui.date_picker(name="train_beg", label = "Training Begin", value = "2012-07-01"),
                ui.date_picker(name="train_end", label = "Training End", value = "2013-07-01"),
                ui.message_bar(type='warning', text='Training time limited to 60 seconds...'),
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

        pv_lim = q.client.pv_df[["AC System Output (W)", "label", "timestamp"]]
        pv_lim['timestamp'] = pv_lim['timestamp'].astype(str)
        pv_lim.columns = ['watts', 'label','timestamp']

        # Create data buffer using training data
        ts_plot_data = data('watts label timestamp', rows = [tuple(x) for x in pv_lim.to_numpy()])

        q.page['timeseries'] = ui.plot_card(box = ui.box('timeseries', height = '1000px'), 
            title = 'Behind the Meter Solar Generation',
            data = ts_plot_data, # TODO figure out the best way to manage memory here
            plot = ui.plot([
                ui.mark(type='point', x_scale='time',  x='=timestamp', y='=watts', color='=label', y_min=0, color_range="#CCFFFF #3399FF #FF9966 #336600 ", y_title="Watts (AC)"),
                ui.mark(type='path', x_scale='time',  x='=timestamp', y='=watts', color='=label', y_min=0, color_range="#CCFFFF #3399FF #FF9966 #336600", y_title="Watts (AC)")
                ])
            )

# Main loop
@app('/')
async def serve(q: Q):
    init_app(q)
    copy_expando(q.args, q.client)
    if not q.app.initialized:
        q.app.initialized = True

    if not await handle_on(q):
        await main_menu(q)

    await q.page.save()


############################
### ADDITIONAL FUNCTIONS ###
############################

def clean_cards(q):
    cards_to_clean = ["main"] + wave_utils_cards_to_clean
    for card in cards_to_clean:
        del q.page[card]

def mlops_get_schema(endpoint_url):
    url = endpoint_url + "/sample_request"
    response = requests.get(url)

    # Return response if status is valid
    if response.status_code == 200:
        return json.loads(response.text)

# Custom Functions for App
def score(data, endpoint, fields, batch_size = 5):
    '''Function to score an endpoint using input data'''

    request_data = list()
    response_data = []

    #fields = mlops_get_schema(endpoint)["fields"]
    print(f'FIELDS: {fields}')

    for index, row in tqdm(data.iterrows()):
        _data = row.values.tolist()
        request_data.append(list(map(str, _data)))
        
        # If we've reached the batch_size in our loop, then score and reset
        if len(request_data) == batch_size:  

            request_json = {"fields": fields, "rows": request_data}
        
            #print(f'REQUEST JSON: {request_json}')

            # Retrieve Response From Endpoint
            response = requests.post(url=endpoint, json=request_json)
        
            #print(f'RESPONSE: {response}')

            # Parse Forecasts
            fcsts = parse_fcst(response)

            #print(f'PARSED FCSTS: {fcsts}')

            request_data = list()
            response_data.append(fcsts)

    response_data = pd.concat(response_data)

    return response_data

def parse_fcst(fcst):
    '''Function to take an endpoint response and format it as a DataFrame'''

    parsed = json.loads(fcst.text)["score"]

    #print(f'PARSED: {parsed}')

    parsed_df = pd.DataFrame(parsed).astype(float)

    #print(f'PARSED DF: {parsed_df}')

    parsed_df.columns = ["fcst"]
    return parsed_df


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

# Example Usage

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


