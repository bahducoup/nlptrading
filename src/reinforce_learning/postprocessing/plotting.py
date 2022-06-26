import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "vscode"

def plot_portfolio_traces(traces:list, time_ind):
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)



    fig.update_layout(
        legend=dict(
            x=0,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=15,
                color="black"
            ),
            bgcolor="White",
            bordercolor="white",
            borderwidth=2
            
        ),
    )
    #fig.update_layout(legend_orientation="h")
    fig.update_layout(title={
            #'text': "Cumulative Return using FinRL",
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    #with Transaction cost
    #fig.update_layout(title =  'Quarterly Trade Date')
    fig.update_layout(
    #    margin=dict(l=20, r=20, t=20, b=20),

        paper_bgcolor='rgba(1,1,0,0)',
        plot_bgcolor='rgba(1, 1, 0, 0)',
        #xaxis_title="Date",
        yaxis_title="Cumulative Return",
    xaxis={'type': 'date', 
        'tick0': time_ind[0], 
            'tickmode': 'linear', 
        'dtick': 86400000.0 *80}

    )
    fig.update_xaxes(showline=True,linecolor='black',showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',mirror=True)
    fig.update_yaxes(showline=True,linecolor='black',showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',mirror=True)
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='LightSteelBlue')

    return fig