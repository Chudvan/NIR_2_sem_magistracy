from dash import Dash, dash_table, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from tools import pa_fields, seven_fields, facs_fields, model_types
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

input_fields = []
output_fields = []


def create_first_or_third(dim=None, first=True):
    #print(dim, first)
    id = 'first_col' if first else 'third_col'

    if dim is None:
        return dbc.Card(dbc.CardBody([]), id=id,
                        style={"height": "500px",
                            "border": "none"})
    if dim == 2:
        fields = pa_fields
        data = [{f: 0 for f in fields}]
    elif dim == 7:
        fields = seven_fields
        data = [{f: 0 for f in fields}]
    elif dim == 42:
        fields = ['Action name', 'Value']
        data = [{fields[0]: f, fields[1]: 0} for f in facs_fields]

    editable = True if first else False
    data_table_id = 'input-table' if first else 'output-table'

    card_bodys_children = [
        dash_table.DataTable(
            id=data_table_id,
            columns=(
                [{'id': f, 'name': f} for f in fields]
            ),
            data=data,
            editable=editable
        )
    ]

    if dim == 2 or dim == 7:
        graph_id = 'input-graph' if first else 'output-graph'
        card_bodys_children.append(dcc.Graph(id=graph_id))
        df = pd.DataFrame([[0 for i in range(len(fields))]], columns=fields)
        if dim == 2:
            print('firstly')
            fig = px.scatter(df, x=fields[0], y=fields[1])
        elif dim == 7:
            df = df.T.rename(columns={0: 'Value'})
            df['Emotion'] = df.index
            fig = px.bar(df, x='Emotion', y='Value', range_y=[0, 1])
        card_bodys_children[-1].figure = fig

    card = dbc.Card(
        dbc.CardBody(card_bodys_children),
        id=id,
        style={"border": "none"}
    )
    return card


def create_second():
    second_card = dbc.Card(
        dbc.CardBody(
            [
                dbc.Card(
                    style={"height": "33%",
                           "border": "none"}
                ),
                dbc.Card(
                    dbc.CardBody([
                        dbc.Card(
                            dcc.Dropdown(model_types,
                                         placeholder="Выберите тип модели (преобразования)",
                                         id='dropdown'),
                        ),
                        dbc.Card(
                            dcc.Upload(
                                id="upload-model",
                                children=html.Div(
                                    ["Перетащите или щёлкните, чтобы выбрать модель(ли) для загрузки."]
                                ),
                                style={  # N
                                    "width": "100%",
                                    "height": "60px",
                                    "lineHeight": "60px",
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "textAlign": "center",
                                },
                                multiple=True,
                            )
                        ),
                        dbc.Card(
                            html.Button('Вычислить', id='calculate', n_clicks=0)
                        )]),
                    style={"height": "33%",
                           "border": "none"}
                ),
                dbc.Card(
                    style={"height": "33%",
                           "border": "none"}
                )
            ]
        ),
        style={'textAlign': 'center',
               "border": "none"},
        id='second_col'
    )
    return second_card


cards = dbc.CardGroup([
    create_first_or_third(),
    create_second(),
    create_first_or_third(first=False)
])

app.layout = cards


@app.callback(
    Output('first_col', 'children'),
    Output('third_col', 'children'),
    Input('dropdown', 'value'))
def change_widgets(model_type):
    #print(model_type)
    if model_type is None:
        first_dim = third_dim = None
    else:
        first_dim = int(model_type.split('_')[0])
        third_dim = int(model_type.split('_')[1])

    first_col = create_first_or_third(first_dim)
    third_col = create_first_or_third(third_dim, first=False)

    return first_col, third_col


@app.callback(Output('input-graph', 'figure'),
              Input('input-table', 'data'),
              Input('input-table', 'columns'))
def update_graph(rows, cols):
    print('here')
    cols = [c['name'] for c in cols]
    color_field = 'color'
    cols.append(color_field)
    print(rows)
    for key in rows[0]:
        rows[0][key] = float(rows[0][key])
    rows[0][color_field] = ' '
    center_point = [0, 0, '']
    rows.insert(0, dict(zip(cols, center_point)))
    print(rows)
    df = pd.DataFrame(rows, columns=cols)
    print(df)
    if len(cols[:-1]) == 2:
        print(df.iloc[0][0], df.columns)
        fig = px.scatter(df, x=cols[0], y=cols[1],
                         color=color_field, range_x=[-1, 1], range_y=[-1, 1],
                         color_discrete_sequence=['white', 'red'])
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
        fig.update_traces(marker={'size': 10})
        fig.update_layout(showlegend=False)
        fig.update_traces(hovertemplate=cols[0] + ': %{x}<br>' + cols[1] + ': %{y}')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
