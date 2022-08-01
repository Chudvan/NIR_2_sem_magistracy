from dash import Dash, dash_table, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from tools import pa_fields, seven_fields, facs_fields, \
    model_types, type_model_dict, create_tempfile_from_content, \
    get_model_type, delete_tempfiles, get_most_frequent, \
    type_model_interface_key_to_type_model_key, data_table_to_data_frame, \
    data_frame_to_data_table
from model_interfaces import *
import plotly.express as px
import pandas as pd
import sys

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
model_facade = ModelFacade()

input_fields = []
output_fields = []
N_CLICKS = 0

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
                dcc.ConfirmDialog(id='confirm-error',
                                  message='', ),
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
              Output('output-graph', 'figure'),
              Input('input-table', 'data'),
              Input('input-table', 'columns'),
              Input('output-table', 'data'),
              Input('output-table', 'columns'))
def update_input_graph(rows_input, cols_input, rows_output,
                       cols_output):
    figures = []
    for i in range(2):
        if i == 0:
            rows = rows_input
            cols = cols_input
        else:
            rows = rows_output
            cols = cols_output
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
            figures.append(fig)
    return figures


@app.callback(Output('dropdown', 'value'),
              Output('confirm-error', 'displayed'),
              Output('confirm-error', 'message'),
              Input('upload-model', 'filename'),
              Input('upload-model', 'contents'),
              State('dropdown', 'value'))
def upload_model_changes(uploaded_filenames, uploaded_file_contents, model_type_dropdown):
    print('upload_model_changes')
    displayed = False
    message = ''
    print(uploaded_filenames)
    if (uploaded_filenames is not None) and (uploaded_file_contents is not None):
        tempfile_list = []
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            cur_path_to_tempfile = create_tempfile_from_content(data)
            # Check model_type_file
            model_type_file = get_model_type(name, cur_path_to_tempfile)
            if not model_type_file:
                delete_tempfiles(tempfile_list)
                displayed = True
                message = f'Модель {name} имеет некорректный формат.'
                return model_type_dropdown, displayed, message
            # Add model_file info
            tempfile_list.append((name, cur_path_to_tempfile, model_type_file))
        unique_model_types = set(map(lambda x: x[2], tempfile_list))
        # Check all - unique
        if len(unique_model_types) != len(tempfile_list):
            delete_tempfiles(tempfile_list)
            displayed = True
            most_frequent_t = get_most_frequent(tempfile_list)
            message = f'Вы выбрали несколько моделей одинакового типа. \
Модель типа {most_frequent_t[0]} встречается {most_frequent_t[1]} раз.'
            return model_type_dropdown, displayed, message
        cur_attrs_model_facade = {}
        try:
            # Save current model_facade models (attrs)
            for model_attr in type_model_dict.values():
                cur_attrs_model_facade[model_attr] = getattr(model_facade, model_attr)
            # Create new model_facade models (attrs)
            for cur_filename, cur_path_to_tempfile, model_type_file in tempfile_list:
                type_model_key = type_model_interface_key_to_type_model_key(model_type_file)
                model_attr_name = cur_attr = type_model_dict[type_model_key]
                model_attr_val = getattr(sys.modules[__name__],
                        type_model_interface_dict[model_type_file])(cur_path_to_tempfile)
                setattr(model_facade, model_attr_name, model_attr_val)
        except Exception:
            # Error while creating one of model_facade's models
            print('Exception')
            delete_tempfiles(tempfile_list)
            # Откат до cur_attrs_model_facade
            for model_attr_name, model_attr_val in cur_attrs_model_facade.items():
                setattr(model_facade, model_attr_name, model_attr_val)
            # Window 'Cant create model...'
            displayed = True
            message = f'Не удаётся создать модель типа {cur_attr} из файла {cur_filename}.'
    return model_type_dropdown, displayed, message


@app.callback(Output('calculate', 'disabled'),
              Output('upload-model', 'children'),
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


@app.callback(Output('output-table', 'data'),
              Input('calculate', 'n_clicks'),
              State('dropdown', 'value'),
              State('input-table', 'data'),
              State('input-table', 'columns'))
def update_output_table(n_clicks, model_type_dropdown, rows, cols):
    global N_CLICKS
    if n_clicks != N_CLICKS:
        type_model = model_type_dropdown.replace(' -> ', '_')
        model_attr_name = type_model_dict[type_model]
        model_attr_val = getattr(model_facade, model_attr_name)
        df = data_table_to_data_frame(rows, cols)
        output_df = model_attr_val.predict(df)
        data = data_frame_to_data_table(output_df)
        N_CLICKS = n_clicks
    return data


if __name__ == '__main__':
    app.run_server(debug=True)
