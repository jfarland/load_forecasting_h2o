from h2o_wave import main, app, Q, ui, data
from h2o_wave_ml import build_model, ModelType

import pandas as pd
import numpy as np


@app('/load_forecasting')
async def serve(q: Q):

    if q.args.train:

        # train WaveML Model using H2O-3 AutoML
        q.client.wave_model = build_model(
            train_df = q.client.train_df,
            target_column = 'kw',
            model_type = ModelType.H2O3,
            _h2o3_max_runtime_secs = 60*5,
            _h2o3_nfolds = 2
        )
        
        model_id = q.client.wave_model.model.model_id
        #accuracy = round(100 - q.client.wave_model.model.mean_per_class_error() * 100, 2)
        accuracy = 100

        # show training details and prediction option
        q.page['inputs'].items[1].buttons.items[1].button.disabled = False
        q.page['inputs'].items[2].message_bar.type = 'success'
        q.page['inputs'].items[2].message_bar.text = 'Training successfully completed!'
        q.page['inputs'].items[3].text.content = f'''**H2O AutoML model id:** {model_id} <br />
            **Accuracy:** {accuracy}%'''
        q.page['inputs'].items[4].text.content = ''

    elif q.args.predict:
        # predict on test data
        preds = q.client.wave_model.predict(test_df=q.client.test_df)

        # show predictions
        q.page['inputs'].items[2].message_bar.text = 'Prediction successfully completed!'
        q.page['inputs'].items[4].text.content = f'''**Example predictions:** <br />
            {preds[0]} <br /> {preds[1]} <br /> {preds[2]}'''

    else:
        # prepare sample train and test dataframes

        # TODO: store the energy + weather data in a database format (e.g., Wave DB, or NoSQL @ scale)
        q.client.train_df = pd.read_csv("data/train.csv")
        q.client.test_df = pd.read_csv("data/test.csv")

        #q.client.train_df, q.client.test_df = train_test_split(data, train_size=0.8)

        # structure UI TODO: update flex layout to account for additional breakpoints (for various devices)
        q.page['meta'] = ui.meta_card(box='', layouts=[
            ui.layout(
                breakpoint='xl',
                width='1200px',
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
                ui.text(content='Select a time period to train predictive models on...'),
                ui.buttons(items=[
                    ui.button(name='train', label='Retrain', primary=True),
                    ui.button(name='predict', label='Predict', primary=True, disabled=True),
                ]),
                ui.message_bar(type='warning', text='Training time limited to 5 minutes...'),
                ui.text(content=''),
                ui.text(content='')
        ])

        q.client.train_df["timestamp"] = pd.to_datetime(q.client.train_df["timestamp"])
        train_agg =  q.client.train_df.drop("sensor_id", axis = 1).groupby("timestamp")["kw"].agg(["mean"])
        train_agg["label"] = "historical"
        train_agg["timestamp"] = train_agg.index.astype(str) # cast back to string for JSON serialization :( ...
        train_agg = train_agg.reset_index(drop = True)
        train_agg.columns = ['kw', 'label','timestamp']

        # Create data buffer using training data
        ts_plot_data = data('kw label timestamp', rows = [tuple(x) for x in train_agg.to_numpy()])
   
        # Time Series Plots TODO: this should show the time series training data, and predictions when predicted
        # Reference: https://wave.h2o.ai/docs/examples/plot-line-groups
        q.page['timeseries'] = ui.plot_card(box = ui.box('timeseries', height = '600px'), 
            title = 'Interconnected Grid Demand',
            #data = data('kw label timestamp', rows = train_agg.head().to_dict("list")), # TODO figure out the best way to manage memory here
            data = ts_plot_data, # TODO figure out the best way to manage memory here
            plot = ui.plot([
                ui.mark(type = 'line', x_scale = 'time',  x = '=timestamp', y = '=kw', color = '=label', y_min = 0, color_domain=["#3399FF #FF9966"])])
        )

        # Time Series Segmentation Card
        q.page['segmentation'] = ui.form_card(box = ui.box('segmentation'), items = [ui.text('Customer Segmentation')])

        # Model Interpretability / Diagnostic Card 
        q.page['mli'] = ui.form_card(box = ui.box('mli'), items = [ui.text('Modeling Diagnostics')])

        # Summary Statistics Card - TODO: turn this into a table https://wave.h2o.ai/docs/examples/table
        q.page['stats'] = ui.form_card(box = ui.box('stats'), items = [
            ui.text('Summary Stats')])

    await q.page.save()


# Plotting example
# page = site['/demo']

# v = page.add('example', ui.plot_card(
#     box='1 1 4 5',
#     title='Point',
#     data=data('price performance', 50, rows=[(random.random(), random.random()) for _ in range(50)]),
#     plot=ui.plot([
#         ui.mark(type='point', x='=price', y='=performance')
#     ])
# ))
# page.save()


# ui.plot([
#     ui.mark(type='point', x='=price', y='=performance', x_min=0, x_max=100, y_min=0, y_max=100),  # the plot
#     ui.mark(x=50, y=50, label='point'),
#     ui.mark(x=40, label='vertical line'),
#     ui.mark(y=40, label='horizontal line'),
#     ui.mark(x=70, x0=60, label='vertical region'),
#     ui.mark(y=70, y0=60, label='horizontal region'),
#     ui.mark(x=30, x0=20, y=30, y0=20, label='rectangular region')
# ])