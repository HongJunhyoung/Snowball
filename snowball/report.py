import numpy as np
import pandas as pd
from plotly.offline import iplot
import plotly.graph_objs as go
import plotly.io as pio
from IPython.display import display, Markdown, HTML, Image

pio.templates.default='none' # plotly style


def calc_stats(returns, trades=None):
    '''
    Calculate stats of the portfolio.

    Parameters
    ----------
    returns : pd.Series
        daily return of portfolio

    Returns
    -------
    stats : dictionary
        MTD, YTD, 1/5/10year return, Total Return, CAGR, Vol, Sharpe Ratio, MDD, etc.
    '''
    last_day = returns.index[-1]
    nav = (1 + returns).cumprod()

    stats = dict()

    t0 = last_day.replace(day=1) - pd.Timedelta(days=1) # end of previous month
    stats['MTD'] = nav[-1] / nav[:t0][-1] - 1 if not nav[:t0].empty else np.nan

    t0 = last_day.replace(month=1, day=1) - pd.Timedelta(days=1) # end of previous year
    stats['YTD'] = nav[-1] / nav[:t0][-1] - 1 if not nav[:t0].empty else np.nan

    years = [1, 5, 10]
    for n in years:
        t0 = last_day - pd.Timedelta(days=(365*n + 1)) # n years ago
        stats[f'{n}Y'] = nav[-1] / nav[:t0][-1] - 1 if not nav[:t0].empty else np.nan

    stats['Total Return'] = nav[-1] - 1

    days_per_year = 252
    stats['CAGR'] = nav[-1] ** (days_per_year / len(returns)) - 1
    stats['Volatility'] = returns.std() * np.sqrt(252)  # default ddof = 1
    stats['Sharpe Ratio'] = returns.mean() * days_per_year / stats['Volatility']

    running_max = np.maximum.accumulate(nav)
    drawdown = -((running_max - nav) / running_max)
    stats['MDD'] = np.min(drawdown)
    stats['MDD Date'] = drawdown.idxmin().strftime('%Y-%m-%d')

    # Monthly return metrics
    if returns.iloc[0] == 0:
        # Exclude initial rebalancing date
        _returns = returns.iloc[1:]
    else:
        _returns = returns
    mr = _returns.groupby([_returns.index.year, _returns.index.month]).apply(lambda x: (1+x).cumprod()[-1] - 1)
    stats['Best Month'] = mr.max()
    stats['Worst Month'] = mr.min()
    stats['Positive Months'] = f'{len(mr[mr>0])} out of {len(mr)}'

    # Annual turnover
    if trades is not None:
        turnover = trades.abs().groupby(trades.index.get_level_values(0)).sum()
        turnover = turnover.iloc[1:]  # Exclude initial trades
        average_monthly_turnover = turnover.sum() / len(mr) 
        stats['Annual Turnover'] = average_monthly_turnover * 12
    else:
        stats['Annual Turnover'] = np.NaN

    return stats


def report_log(log):
    '''
    Display the backtest log.

    Parameters
    ----------
    log : components.BacktestLog
        The object contains bactest log
    '''
    display(Markdown('__[ Backtest Log ]__'))
    report = []
    for event in log['event'].unique():
        _df = log[log['event']==event] 
        cnt = _df.shape[0]
        for i, row in enumerate(_df[['date', 'message']].values):
            if cnt > 5 and 2 < i < (cnt - 2):
                continue
            elif cnt > 5 and i == 2:
                report.append(['\u00B7' * 3, ' ', '  ' + '\u00B7' * 5, '\u00B7' * 5]) 
            elif i == 0:
                report.append([str(i+1), event, row[0].strftime('%Y-%m-%d'), row[1]])
            else:
                report.append([str(i+1), ' ', row[0].strftime('%Y-%m-%d'), row[1]])
    report = pd.DataFrame(report, columns=['No', 'Event', 'Date', 'Message'])
    display(HTML(report.to_html(index=False)))
    print('\n')


def report_perf(returns, gross_returns=None, trades=None, weights=None, benchmark=None, charts='interactive'):
    '''
    Report the portfolio performance.

    Parameters
    ----------
    returns : pd.Series
    gross_returns : pd.Series
    trades : pd.Series
    weights : pd.Series
    benchmark : pd.Series
    '''
    start = returns.index[0]
    end = returns.index[-1]
    months = (end.year - start.year) * 12 + end.month - start.month
    yr = months // 12
    mn = months % 12
    period = f'{yr} years' if yr >= 1 else ''
    period += f' {mn} months' if mn > 0 else ''
    stats = calc_stats(returns, trades)
    stats = pd.DataFrame.from_dict(stats, orient='index', columns=['Portfolio'])
    if benchmark is not None:
        bm_stats = calc_stats(benchmark)
        bm_stats = pd.DataFrame.from_dict(bm_stats, orient='index', columns=['Benchmark'])
        stats = pd.concat([stats, bm_stats], axis=1)

    if charts:
        display(Markdown('__[ Porfolio Performance ]__'))

    display(Markdown(f"- _{start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')} ( {period} )_"))
    display((stats.T
             .style.format('{:+.2%}', na_rep='N/A') # .set_properties(**{'width': '50px'}) 
                   .format('{:.2f}', subset=['Sharpe Ratio'])
                   .format('{:s}', subset=['MDD Date'])
                   .format('{:s}', subset=['Positive Months'])
                   .format('{:.2%}', subset=['Volatility', 'Annual Turnover'], na_rep='N/A')
           ))

    if charts:
        # draw charts
        fig1 = make_history_chart(returns, gross_returns, trades, weights, benchmark)
        fig2 = make_periodic_chart(returns)
        if charts == 'interactive':
            iplot(fig1)
            iplot(fig2)
        elif charts == 'static':
            display(Image(fig1.to_image(format='png', engine='kaleido', width=850, height=800)))
            display(Image(fig2.to_image(format='png', engine='kaleido', width=850, height=400)))
        else:
            raise ValueError('Charts should be interactive or static.')


def make_history_chart(returns, gross_returns, trades, weights, benchmark):
    nav = (1 + returns).cumprod() * 100
    if gross_returns is not None:
        gross_nav = (1 + gross_returns).cumprod() * 100
    running_max = np.maximum.accumulate(nav)
    drawdown = -100 * ((running_max - nav) / running_max)
 
    data = []
    data.append(go.Scatter(x=returns.index, 
                           y=nav.round(2), 
                           name='Portfolio', 
                           opacity=0.8, 
                           yaxis='y'))

    if benchmark is not None:
        bm = ((1 + benchmark).cumprod() * 100)
        data.append(go.Scatter(x=benchmark.index, 
                            y=bm.round(2), 
                            name='Benchmark', 
                            opacity=0.8, 
                            line=dict(color='grey'),
                            yaxis='y'))
    data.append(go.Bar(x=drawdown.index, 
                       y=drawdown.round(2), 
                       name='drawdown', 
                       opacity=0.8, 
                       yaxis='y2', 
                       showlegend=False))
    data.append(go.Bar(x=returns.index, 
                       y=(returns * 100).round(2), 
                       name='daily return', 
                       opacity=0.8, 
                       yaxis='y3', 
                       showlegend=False, 
                       marker_color='rgb(169,169,169)'))
    if weights is not None:
        weights = weights.loc[trades.index.get_level_values(0).tolist(), :] # Show only rebalanced weights
        df = weights.unstack()
        for col in df.columns:
            data.append(go.Bar(x=df.index, 
                               y=(df[col] * 100).round(2), 
                               name=str(col), 
                               opacity=0.8, 
                               yaxis='y4', 
                               showlegend=False))
    if trades is not None:
        # Do not draw initial trades which are not turnover.
        first_rebal_dt = trades.index.get_level_values(0).unique()[1]
        trades = trades.loc[first_rebal_dt:]
        buy = trades[trades>0].groupby('date').sum().astype(float)
        sell = trades[trades<0].groupby('date').sum().astype(float)
        data.append(go.Bar(x=buy.index, 
                           y=(buy * 100).round(2), 
                           name='Buy', 
                           opacity=0.8, 
                           yaxis='y5', 
                           showlegend=False, 
                           marker_color='rgb(244,109,67)'))
        data.append(go.Bar(x=sell.index, 
                           y=(sell * 100).round(2), 
                           name='Sell', 
                           opacity=0.8, 
                           yaxis='y5', 
                           showlegend=False, 
                           marker_color='rgb(69,117,180)'))

    if gross_returns is not None:
        data.append(go.Scatter(x=gross_returns.index,
                               y=gross_nav.round(2),
                               name='Before cost',
                               opacity=0.3, line_color='gray',
                               yaxis='y'))

    yaxis_lin = dict(title='NAV (Linear scale)', autorange=True, domain=[0.50, 1])
    yaxis_log = dict(title='NAV (Log scale)', autorange=True, domain=[0.50, 1], type='log')

    layout = go.Layout(
        height=1000,
        margin=go.layout.Margin(b=30, r=40),
        xaxis=dict(
            title='',
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1M', step='month', stepmode='backward'),
                    dict(count=3, label='3M', step='month', stepmode='backward'),
                    dict(count=6, label='6M', step='month', stepmode='backward'),
                    dict(count=1, label='1Y', step='year', stepmode='backward'),
                    dict(count=3, label='3Y', step='year', stepmode='backward'),
                    dict(count=5, label='5Y', step='year', stepmode='backward'),
                    dict(count=10, label='10Y', step='year', stepmode='backward'),
                    dict(count=15, label='15Y', step='year', stepmode='backward'),
                    dict(count=20, label='20Y', step='year', stepmode='backward'),
                    dict(count=1, label='MTD', step='month', stepmode='todate'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    ])),
            type='date',
            autorange=True,
            showspikes=True,
        ),
        yaxis=yaxis_log,
        yaxis2=dict(
            title='drawdown(%)',
            autorange=True,
            domain=[0.36, 0.46],
            anchor='x'
        ),
        yaxis3=dict(
            title='return(%)',
            autorange=True,
            domain=[0.24, 0.34],
            anchor='x'
        ),
        yaxis4=dict(
            title='weights(%)',
            autorange=True,
            domain=[0.12, 0.22],
            anchor='x'
        ),
        yaxis5=dict(
            title='trades(%)',
            autorange=True,
            domain=[0.00, 0.10],
            anchor='x'
        ),
        barmode='relative',
        updatemenus=list([
            dict(type='buttons',
                 active=0,
                 buttons=list([
                     dict(label='Log',
                          method='update',
                          args=[{}, {'yaxis': yaxis_log}]),
                     dict(label='Linear',
                          method='update',
                          args=[{}, {'yaxis': yaxis_lin}]),
                 ]),
            )
        ]),

        legend=dict(orientation='h', xanchor='right', x=1, yanchor='bottom', y=1.01)
    )

    figure = go.Figure(data=data, layout=layout)
    return figure

def make_periodic_chart(returns):
    # Annual returns 
    yr = returns.groupby(returns.index.year).apply(lambda x: (1+x).cumprod()[-1] - 1)
    trace_annual = go.Bar(x=yr * 100,
                          y=yr.index,
                          name='Annual Return',
                          marker=dict(color='#7EC0EE'),
                          orientation='h',
                          hoverinfo='text',
                          hovertext=[f'{y:n}   {v:+.2%}' for y, v in yr.reset_index().values],
                          xaxis='x'
                          )
    trace_mean = go.Scatter(x=[yr.mean() * 100] * len(yr.index),
                            y=yr.index,
                            name='Average',
                            line=dict(dash='dot'),
                            hoverinfo='text',
                            hovertext=[f'{yr.mean():+.2%}'] * len(yr.index),
                            xaxis='x'
                            )

    # Monthly returns
    mr = returns.groupby([returns.index.year, returns.index.month]).apply(lambda x: (1+x).cumprod()[-1] - 1)
    hm = mr.unstack() # unstack months to columns
    x = hm.columns # month
    y = hm.index   # year
    z = (hm * 100).values.tolist()
    # hoverinfo
    text = []
    for year_l_index, year_l in enumerate(hm.values.tolist()):
        text_temp = []
        for month_index, value in enumerate(year_l):
            yyyymm = str(hm.index.tolist()[year_l_index]) + '-' + str(hm.columns.tolist()[month_index]).rjust(2, '0')
            text_temp.append(f'{yyyymm}   {value:+.2%}')
        text.append(text_temp)
    colorscale = [
        [0.0, 'rgb(49,54,149)'],
        [0.1111111111111111, 'rgb(69,117,180)'],
        [0.2222222222222222, 'rgb(116,173,209)'],
        [0.3333333333333333, 'rgb(171,217,233)'],
        [0.4444444444444444, 'rgb(224,243,248)'],
        [0.5555555555555555, 'rgb(254,224,144)'],
        [0.6666666666666666, 'rgb(253,174,97)'],
        [0.7777777777777777, 'rgb(244,109,67)'],
        [0.8888888888888888, 'rgb(215,48,39)'],
        [1.0, 'rgb(165,0,38)']]
    sigma = np.std(mr.dropna().values * 100)
    trace_monthly = go.Heatmap(x=x,
                               y=y,
                               z=z,
                               zauto=False,
                               zmin=-3*sigma,
                               zmax=3*sigma,
                               hoverinfo='text',
                               hovertext=text,
                               colorscale=colorscale,
                               showscale=True,
                               xaxis='x2',
    )

    layout = go.Layout(height=400,
                       margin=go.layout.Margin(t=25, l=120),
                       xaxis=dict(title='Annual return', domain=[0.00, 0.4]),
                       xaxis2=dict(title='Monthly return', domain=[0.45, 1.0]),
                       yaxis=dict(title='Year', autorange='reversed', tickformat='d'),
                       hovermode='closest',
                       autosize=True,  # 사이즈 자동
                       showlegend=False,
    )

    data = [trace_annual, trace_mean, trace_monthly]
    figure = go.Figure(data=data, layout=layout)
    return figure