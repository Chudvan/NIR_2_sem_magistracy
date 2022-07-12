from dash import Dash, dash_table, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from tools import pa_fields, seven_fields, facs_fields, model_types
import plotly.express as px

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

input_fields = []
output_fields = []


def create_first(dim=None, editable=False):
    print(dim)
    if dim == 2:
        input_fields = pa_fields
        data = [{f: 0 for f in input_fields}]
    elif dim == 7:
        input_fields = seven_fields
        data = [{f: 0 for f in input_fields}]
    elif dim == 42:
        input_fields = ['Action name', 'Value']
        data = [{input_fields[0]: f, input_fields[1]: 0} for f in facs_fields]

    children = [
        dash_table.DataTable(
            id='input-table',
            columns=(
                [{'id': f, 'name': f} for f in input_fields]
            ),
            data=data,
            editable=editable
        )
    ]

    if dim == 2 or dim == 7:
        children.append(dcc.Graph(id='input-graph'))
    if dim == 7:
        children[-1].children =

    first_card = dbc.Card(
        dbc.CardBody(children),
        id='first_col'
    )
    return first_card


def create_second():
    second_card = dbc.Card(
        dbc.CardBody(
            [
                dbc.Card(
                    style={"height": "33%",
                           "borderWidth": "0px"}
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
                           "borderWidth": "0px"}
                ),
                dbc.Card(
                    style={"height": "33%",
                           "borderWidth": "0px"}
                )
            ]
        ),
        style={'textAlign': 'center'},
        id='second_col'
    )
    return second_card


def create_third(model_type=None):
    third_card = dbc.Card(
        dbc.CardBody(
            [
                dash_table.DataTable(
                    id='output-table',
                    columns=(
                        [{'id': f, 'name': f} for f in output_fields]
                    ),
                    data=[
                        {f: 0 for f in output_fields}
                    ]
                ),
                dcc.Graph(id='output-graph')
            ]
        ),
        id='third_col'
    )
    return third_card


cards = dbc.CardGroup([
    create_first(),
    create_second(),
    create_third()
])

app.layout = cards


@app.callback(
    Output('first_col', 'children'),
    Output('third_col', 'children'),
    Input('dropdown', 'value'))
def change_widgets(model_type):
    first = create_first(model_type)
    third = create_third(model_type)
    return first, third


if __name__ == '__main__':
    app.run_server(debug=True)
