from dash import Dash, dash_table, dcc, html
import dash_bootstrap_components as dbc
from tools import pa_fields, seven_fields

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

cards = dbc.CardGroup([
    dbc.Card(dbc.CardBody(
        [
            dash_table.DataTable(
                id='input-table',
                columns=(
                    [{'id': f, 'name': f} for f in pa_fields]
                ),
                data=[
                    {f: 0 for f in pa_fields}
                ],
                editable=True
            ),
            dcc.Graph(id='input-graph')
        ]
    )),
    dbc.Card(
        dbc.CardBody(
            [
                dbc.Card(
                    style={"height": "33%",
                           "borderWidth": "0px"}
                ),
                dbc.Card(
                    dbc.CardBody([
                        dbc.Card(
                            dcc.Dropdown([
                                {'label': '2 -> 7', 'value': '2_7'},
                                {'label': '7 -> 2', 'value': '7_2'},
                                {'label': '7 -> 42', 'value': '7_42'},
                                {'label': '42 -> 7', 'value': '42_7'},
                                {'label': '2 -> 42', 'value': '2_42'},
                                {'label': '42 -> 2', 'value': '42_2'},
                            ],
                            placeholder="Выберите тип модели (преобразования)",
                            id='dropdown'),
                        ),
                        dbc.Card(
                            dcc.Upload(
                                id="upload-model",
                                children=html.Div(
                                    ["Перетащите или щёлкните, чтобы выбрать модель(ли) для загрузки."]
                                ),
                                style={ # N
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
                            "borderWidth": "0px"}
                ),
                dbc.Card(
                    style={"height": "33%",
                           "borderWidth": "0px"}
                )
            ]
        ),
        style={'textAlign': 'center'}
    ),
    dbc.Card(dbc.CardBody(
        [
            dash_table.DataTable(
                id='output-table',
                columns=(
                    [{'id': f, 'name': f} for f in seven_fields]
                ),
                data=[
                    {f: 0 for f in seven_fields}
                ]
            ),
            dcc.Graph(id='output-graph')
        ]
    ))
])

app.layout = cards

@app.callback()
def foo():
    pass


if __name__ == '__main__':
    app.run_server(debug=True)
