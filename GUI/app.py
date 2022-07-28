from dash import Dash, dash_table, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from tools import pa_fields, seven_fields, facs_fields, \
    model_types, type_model_dict, create_tempfile_from_content, \
    get_model_type, delete_tempfiles, get_most_frequent
from model_interfaces import ModelFacade
import plotly.express as px
import pandas as pd

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
model_facade = ModelFacade()

input_fields = []
output_fields = []


def create_first_or_third(dim=None, first=True):
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
            fig = px.scatter(df, x=fields[0], y=fields[1], range_x=[-1, 1],
                             range_y=[-1, 1], color_discrete_sequence=['red'])
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(fixedrange=True)
            fig.update_traces(marker={'size': 10})
            fig.update_layout(showlegend=False)
            fig.update_traces(hovertemplate=fields[0] + ': %{x}<br>' + fields[1] + ': %{y}')
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
                            dcc.ConfirmDialog(id='confirm-error',
                                              message='',),
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
                            html.Button('Вычислить', id='calculate',
                                        n_clicks=0, disabled=True)
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
def change_first_third_cols(model_type):
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
    cols = [c['name'] for c in cols]
    if len(cols) == 2 or len(cols) == 7:
        for key in rows[0]:
            rows[0][key] = float(rows[0][key])
        if len(cols) == 2:
            color_field = 'color'
            cols.append(color_field)
            rows[0][color_field] = ''
            center_point = [0, 0, ' ']
            rows.insert(0, dict(zip(cols, center_point)))
            df = pd.DataFrame(rows, columns=cols)
            fig = px.scatter(df, x=cols[0], y=cols[1],
                             color=color_field, range_x=[-1, 1], range_y=[-1, 1],
                             color_discrete_sequence=['white', 'red'])
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(fixedrange=True)
            fig.update_traces(marker={'size': 10})
            fig.update_layout(showlegend=False)
            fig.update_traces(hovertemplate=cols[0] + ': %{x}<br>' + cols[1] + ': %{y}')
        elif len(cols) == 7:
            df = pd.DataFrame(rows, columns=cols)
            df = df.T.rename(columns={0: 'Value'})
            df['Emotion'] = df.index
            fig = px.bar(df, x='Emotion', y='Value', range_y=[0, 1])
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(fixedrange=True)
    return fig


@app.callback(Output('dropdown', 'value'),
              Output('confirm-error', 'displayed'),
              Output('confirm-error', 'message'),
              Input('upload-model', 'filename'),
              Input('upload-model', 'contents'),
              State('dropdown', 'value'))
def upload_model_chages(uploaded_filenames, uploaded_file_contents, model_type_dropdown):
    displayed = False
    message = ''
    if (uploaded_filenames is not None) and (uploaded_file_contents is not None):
        tempfile_list = []
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            cur_path_to_tempfile = create_tempfile_from_content(data)
            model_type_file = get_model_type(name, cur_path_to_tempfile)
            if not model_type_file:
                delete_tempfiles(tempfile_list)
                displayed = True
                message = f'Модель {name} имеет некорректный формат.'
                return model_type_dropdown, displayed, message
            tempfile_list.append((name, cur_path_to_tempfile, model_type_file))
        unique_model_type = set(map(lambda x: x[2], tempfile_list))
        if len(unique_model_type) != len(tempfile_list):
            delete_tempfiles(tempfile_list)
            displayed = True
            most_frequent_t = get_most_frequent(tempfile_list)
            message = f'Вы выбрали несколько моделей одинакового типа. \
Модель типа {most_frequent_t[0]} встречается {most_frequent_t[1]} раз.'
            return model_type_dropdown, displayed, message
        try:
            cur_values_from_model_facade = {}
            pass
            for model_type_file in unique_model_type:
                pass
        except Exception:
            raise('Ошибка!!!') # N
        # N change example logic for VA_CLEAR model
        ## model_facade.model_va_clear = 1
        # down to here
    return model_type_dropdown, displayed, message


@app.callback(Output('calculate', 'disabled'),
              Output('upload-mo del', 'children'),
              Input('dropdown', 'value'))
def change_disabled_button(model_type):
    upload_children = html.Div(
        ["Перетащите или щёлкните, чтобы выбрать модель(ли) для загрузки."]
    )
    if model_type is None:
        disabled_button = True
    else:
        if getattr(model_facade, type_model_dict[model_type]) is not None:
            disabled_button = False
            upload_children = html.Div(
                ["Модель успешно загружена."]
            )
        else:
            disabled_button = True

    return disabled_button, upload_children


if __name__ == '__main__':
    app.run_server(debug=True)
