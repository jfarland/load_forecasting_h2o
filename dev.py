import random

from synth import FakeTimeSeries, FakeMultiTimeSeries, FakeCategoricalSeries, FakeMultiCategoricalSeries, FakeScatter
from h2o_wave import main, app, data, Q, ui


n = 10
k = 3
f = FakeTimeSeries()
f_multi = FakeMultiTimeSeries()
f_cat = FakeCategoricalSeries()
f_cat_multi = FakeMultiCategoricalSeries(groups=k)
f_scat = FakeScatter()


@app('/demo')
async def serve(q: Q):
    if not q.client.initialized:
        
        q.page['meta'] = ui.meta_card(box='', layouts=[
            ui.layout(breakpoint='xs', zones=[
                ui.zone(name='title'),
                ui.zone(name='plots', direction=ui.ZoneDirection.ROW, wrap='start', justify='center'),
            ]),
            ui.layout(breakpoint='m', zones=[
                ui.zone(name='title'),
                ui.zone(name='plots', direction=ui.ZoneDirection.ROW, wrap='start', justify='center'),
            ]),
            ui.layout(breakpoint='xl', zones=[
                ui.zone(name='title'),
                ui.zone(name='plots', direction=ui.ZoneDirection.ROW, wrap='start', justify='center'),
            ]),
        ])

        q.client.active_theme = 'default'
        q.page['title'] = ui.section_card(
            box='title',
            title='Plot theme demo',
            subtitle='Toggle theme to see default plot colors change!',
            items=[ui.toggle(name='toggle_theme', label='Dark theme', trigger=True)],
        )

        v = q.page.add('point-sized', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Point, sized',
            data=data('price performance discount', n),
            plot=ui.plot([ui.mark(type='point', x='=price', y='=performance', size='=discount')])
        ))
        v.data = [(x, y, random.randint(1, 10)) for x, y in [f_scat.next() for _ in range(n)]]

        v = q.page.add('point-shapes', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Point, shapes',
            data=data('product price performance', n * 2),
            plot=ui.plot([ui.mark(type='point', x='=price', y='=performance', shape='=product', shape_range='circle square')])
        ))
        v.data = [('G1', x, y) for x, y in [f_scat.next() for _ in range(n)]] + [('G2', x, y) for x, y in [f_scat.next() for _ in range(n)]]

        k1, k2 = 20, 10
        v = q.page.add('poit-size', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Points, size-encoded',
            data=data('country product profit', k1 * k2),
            plot=ui.plot([ui.mark(type='point', x='=country', y='=product', size='=profit', shape='circle')])
        ))
        rows = []
        for i in range(k1):
            for j in range(k2):
                x = f.next()
                rows.append((f'A{i + 1}', f'B{j + 1}', x))
        v.data = rows

        v = q.page.add('area', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Area',
            data=data('date price', n),
            plot=ui.plot([ui.mark(type='area', x_scale='time', x='=date', y='=price', y_min=0)])
        ))
        v.data = [(t, x) for t, x, dx in [f.next() for _ in range(n)]]

        v = q.page.add('area-line', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Area + Line',
            data=data('date price', n),
            plot=ui.plot([
                ui.mark(type='area', x_scale='time', x='=date', y='=price', y_min=0),
                ui.mark(type='line', x='=date', y='=price')
            ])
        ))
        v.data = [(t, x) for t, x, dx in [f.next() for _ in range(n)]]

        v = q.page.add('area-line-smooth', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Area + Line, smooth',
            data=data('date price', n),
            plot=ui.plot([
                ui.mark(type='area', x_scale='time', x='=date', y='=price', curve='smooth', y_min=0),
                ui.mark(type='line', x='=date', y='=price', curve='smooth')
            ])
        ))
        v.data = [(t, x) for t, x, dx in [f.next() for _ in range(n)]]

        v = q.page.add('area-line-groups', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Area + Line, groups',
            data=data('product date price', n * 5),
            plot=ui.plot([
                ui.mark(type='area', x_scale='time', x='=date', y='=price', color='=product', y_min=0),
                ui.mark(type='line', x='=date', y='=price', color='=product')
            ])
        ))
        v.data = [(g, t, x) for x in [f_multi.next() for _ in range(n)] for g, t, x, dx in x]

        v = q.page.add('area-groups', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Area, groups',
            data=data('product date price', n * 5),
            plot=ui.plot([ui.mark(type='area', x_scale='time', x='=date', y='=price', color='=product', y_min=0)])
        ))
        v.data = [(g, t, x) for x in [f_multi.next() for _ in range(n)] for g, t, x, dx in x]

        v = q.page.add('area-stacked', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Area, stacked',
            data=data('product date price', n * 5),
            plot=ui.plot(
                [ui.mark(type='area', x_scale='time', x='=date', y='=price', color='=product', stack='auto', y_min=0)])
        ))
        v.data = [(g, t, x) for x in [f_multi.next() for _ in range(n)] for g, t, x, dx in x]

        f_negative = FakeTimeSeries(min=-50, max=50, start=0)
        v = q.page.add('area-negative-values', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Area, negative values',
            data=data('date price', n),
            plot=ui.plot([ui.mark(type='area', x_scale='time', x='=date', y='=price')])
        ))
        v.data = [(t, x) for t, x, dx in [f_negative.next() for _ in range(n)]]

        v = q.page.add('area-range', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Area, range',
            data=data('date low high', n),
            plot=ui.plot([ui.mark(type='area', x_scale='time', x='=date', y0='=low', y='=high')])
        ))
        v.data = [(t, x - random.randint(3, 8), x + random.randint(3, 8)) for t, x, dx in [f.next() for _ in range(n)]]

        v = q.page.add('area-smooth', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Area, smooth',
            data=data('date price', n),
            plot=ui.plot([ui.mark(type='area', x_scale='time', x='=date', y='=price', curve='smooth', y_min=0)])
        ))
        v.data = [(t, x) for t, x, dx in [f.next() for _ in range(n)]]

        v = q.page.add('example', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Label Customization',
            data=data('product price', n),
            plot=ui.plot([
                ui.mark(type='interval', x='=product',
                        y='=${{intl price minimum_fraction_digits=2 maximum_fraction_digits=2}}', y_min=0,
                        label='=${{intl price minimum_fraction_digits=2 maximum_fraction_digits=2}}',
                        label_offset=0, label_position='middle', label_rotation='-90', label_fill_color='#fff',
                        label_font_weight='bold')])
        ))
        v.data = [(c, x) for c, x, dx in [f_cat.next() for _ in range(n)]]

        v = q.page.add('interval', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Interval',
            data=data('product price', n),
            plot=ui.plot([ui.mark(type='interval', x='=product', y='=price', y_min=0)])
        ))
        v.data = [(c, x) for c, x, dx in [f_cat.next() for _ in range(n)]]

        v = q.page.add('interval-annotation', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Categorical-Numeric',
            data=data('product price', n),
            plot=ui.plot([
                ui.mark(type='interval', x='=product', y='=price', y_min=0, y_max=100),
                ui.mark(x='C20', y=80, label='point'),
                ui.mark(x='C23', label='vertical line'),
                ui.mark(y=40, label='horizontal line'),
                ui.mark(x='C26', x0='C23', label='vertical region'),
                ui.mark(y=70, y0=60, label='horizontal region')
            ])
        ))
        v.data = [(c, x) for c, x, dx in [f_cat.next() for _ in range(n)]]

        v = q.page.add('histogram', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Histogram',
            data=data('lo hi price', n),
            plot=ui.plot([ui.mark(type='interval', y='=price', x1='=lo', x2='=hi', y_min=0)])
        ))
        v.data = [(i * 10, i * 10 + 10, x) for i, (c, x, dx) in enumerate([f_cat.next() for _ in range(n)])]

        v = q.page.add('interval-range', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Histogram',
            data=data('lo hi price', n),
            plot=ui.plot([ui.mark(type='interval', y='=price', x1='=lo', x2='=hi', y_min=0)])
        ))
        v.data = [(i * 10, i * 10 + 10, x) for i, (c, x, dx) in enumerate([f_cat.next() for _ in range(n)])]

        v = q.page.add('interval-range', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='35%'),
            ),
            title='Interval, range',
            data=data('product low high', 3),
            plot=ui.plot([ui.mark(type='interval', x='=product', y0='=low', y='=high')])
        ))
        v.data = [(c, x - random.randint(3, 10), x + random.randint(3, 10)) for c, x, _ in [f.next() for _ in range(3)]]

        v = q.page.add('interval-transpose', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='29%'),
            ),
            title='Categorical-Numeric',
            data=data('product price', 20),
            plot=ui.plot([
                ui.mark(type='interval', y='=product', x='=price', x_min=0, x_max=100),
            ])
        ))
        v.data = [(c, x) for c, x, dx in [f_cat.next() for _ in range(20)]]

        v = q.page.add('intervals-theta-stacked', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Intervals, theta, stacked',
            data=data('country product price', n * k),
            plot=ui.plot([
                ui.mark(coord='theta', type='interval', x='=product', y='=price', color='=country', stack='auto', y_min=0)
            ])
        ))
        v.data = [(g, t, x) for x in [f_cat_multi.next() for _ in range(n)] for g, t, x, dx in x]

        v = q.page.add('interval-helix', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Interval, helix',
            data=data('product price', 60),
            plot=ui.plot([ui.mark(coord='helix', type='interval', x='=product', y='=price', y_min=0)])
        ))
        v.data = [(c, x) for c, x, dx in [f_cat.next() for _ in range(60)]]

        v = q.page.add('interval-polar', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Interval, polar',
            data=data('product price', n),
            plot=ui.plot([ui.mark(coord='polar', type='interval', x='=product', y='=price', y_min=0)])
        ))
        v.data = [(c, x) for c, x, dx in [f_cat.next() for _ in range(n)]]

        v = q.page.add('intervals-groups', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Intervals, groups',
            data=data('country product price', n * k),
            plot=ui.plot([ui.mark(type='interval', x='=product', y='=price', color='=country', dodge='auto', y_min=0)])
        ))
        v.data = [(g, t, x) for x in [f_cat_multi.next() for _ in range(n)] for g, t, x, dx in x]

        v = q.page.add('intervals-stacked', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Intervals, stacked',
            data=data('country product price', n * k),
            plot=ui.plot([ui.mark(type='interval', x='=product', y='=price', color='=country', stack='auto', y_min=0)])
        ))
        v.data = [(g, t, x) for x in [f_cat_multi.next() for _ in range(n)] for g, t, x, dx in x]

        v = q.page.add('point-annotation', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Numeric-Numeric',
            data=data('price performance', n),
            plot=ui.plot([
                ui.mark(type='point', x='=price', y='=performance', x_min=0, x_max=100, y_min=0, y_max=100),  # the plot
                ui.mark(x=50, y=50, label='point'),  # A single reference point
                ui.mark(x=40, label='vertical line'),
                ui.mark(y=40, label='horizontal line'),
                ui.mark(x=70, x0=60, label='vertical region'),
                ui.mark(y=70, y0=60, label='horizontal region'),
                ui.mark(x=30, x0=20, y=30, y0=20, label='rectangular region')
            ])
        ))
        v.data = [f_scat.next() for _ in range(n)]

        v = q.page.add('path', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Path',
            data=data('profit sales', n),
            plot=ui.plot([ui.mark(type='path', x='=profit', y='=sales')])
        ))
        v.data = [(x, y) for x, y in [f_scat.next() for _ in range(n)]]

        v = q.page.add('step', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Line, step',
            data=data('date price', n),
            plot=ui.plot([ui.mark(type='line', x_scale='time', x='=date', y='=price', curve='step', y_min=0)])
        ))
        v.data = [(t, x) for t, x, dx in [f.next() for _ in range(n)]]

        v = q.page.add('line-annotation', ui.plot_card(
            box=ui.boxes(
                ui.box('plots', width='100%'),
                ui.box('plots', width='48%'),
                ui.box('plots', width='32%'),
            ),
            title='Time-Numeric',
            data=data('date price', n),
            plot=ui.plot([
                ui.mark(type='line', x_scale='time', x='=date', y='=price', y_min=0, y_max=100),
                ui.mark(x=50, y=50, label='point'),
                ui.mark(x='2010-05-15T19:59:21.000000Z', label='vertical line'),
                ui.mark(y=40, label='horizontal line'),
                ui.mark(x='2010-05-24T19:59:21.000000Z', x0='2010-05-20T19:59:21.000000Z', label='vertical region'),
                ui.mark(y=70, y0=60, label='horizontal region'),
                ui.mark(x='2010-05-10T19:59:21.000000Z', x0='2010-05-05T19:59:21.000000Z', y=30, y0=20,
                        label='rectangular region')
            ])
        ))
        v.data = [(t, x) for t, x, dx in [f.next() for _ in range(n)]]

        q.client.initialized = True

    if q.args.toggle_theme is not None:
        q.client.active_theme = 'neon' if q.args.toggle_theme else 'default'
        q.page['meta'].theme = q.client.active_theme
        q.page['title'].items[0].toggle.value = q.client.active_theme == 'neon'
    await q.page.save()

# from h2o_wave import main, app, Q, ui
# from h2o_wave_ml import build_model, ModelType

# import pandas as pd
# import numpy as np

# #from sklearn.datasets import load_wine
# #from sklearn.model_selection import train_test_split

# @app('/load_forecasting')
# async def serve(q: Q):
#     if q.args.train:

#         # train WaveML Model using H2O-3 AutoML
#         q.client.wave_model = build_model(
#             train_df = q.client.train_df,
#             target_column = 'kw',
#             model_type = ModelType.H2O3,
#             _h2o3_max_runtime_secs = 120,
#             _h2o3_nfolds = 2
#         )

#         model_id = q.client.wave_model.model.model_id
#         #accuracy = round(100 - q.client.wave_model.model.mean_per_class_error() * 100, 2)
#         accuracy = 100

#         # show training details and prediction option
#         q.page['example'].items[1].buttons.items[1].button.disabled = False
#         q.page['example'].items[2].message_bar.type = 'success'
#         q.page['example'].items[2].message_bar.text = 'Training successfully completed!'
#         q.page['example'].items[3].text.content = f'''**H2O AutoML model id:** {model_id} <br />
#             **Accuracy:** {accuracy}%'''
#         q.page['example'].items[4].text.content = ''
#     elif q.args.predict:
#         # predict on test data
#         preds = q.client.wave_model.predict(test_df=q.client.train_df)

#         # show predictions
#         q.page['example'].items[2].message_bar.text = 'Prediction successfully completed!'
#         q.page['example'].items[4].text.content = f'''**Example predictions:** <br />
#             {preds[0]} <br /> {preds[1]} <br /> {preds[2]}'''
#     else:
#         # prepare sample train and test dataframes
#         #data = load_wine(as_frame=True)['frame']

#         #q.client.train_df, q.client.test_df = train_test_split(data, train_size=0.8)

#         q.client.train_df = pd.read_csv("data/energy-df-sample.csv").sample(1000)

#         # display ui
#         q.page['example'] = ui.form_card(
#             box='1 1 -1 -1',
#             items=[
#                 ui.text(content='''The sample dataset used is the
#                     <a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html" target="_blank">wine dataset</a>.'''),
#                 ui.buttons(items=[
#                     ui.button(name='train', label='Train', primary=True),
#                     ui.button(name='predict', label='Predict', primary=True, disabled=True),
#                 ]),
#                 ui.message_bar(type='warning', text='Training will take a few seconds'),
#                 ui.text(content=''),
#                 ui.text(content='')
#             ]
#         )

#     await q.page.save()