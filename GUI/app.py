from dash import Dash, dash_table, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import tools
from model_interfaces import *
import plotly.express as px
import pandas as pd
import sys
import traceback


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
model_facade = ModelFacade()

input_fields = []
output_fields = []
N_CLICKS = 0
ERROR_STATE = False

def create_first_or_third(dim=None, first=True):
    id = 'first_col' if first else 'third_col'

    if dim is None:
        return dbc.Card(dbc.CardBody([]), id=id,
                        style={"height": "500px",
                            "border": "none"})
    if dim == 2:
        fields = tools.pa_fields
        data = [{f: 0 for f in fields}]
    elif dim == 7:
        fields = tools.seven_fields
        data = [{f: 0 for f in fields}]
    elif dim == 42:
        fields = ['Action name', 'Value']
        data = [{fields[0]: f, fields[1]: 0} for f in tools.facs_fields]

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

    graph_id = 'input-graph' if first else 'output-graph'
    if dim == 2 or dim == 7:
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
    elif dim == 42:
        card_bodys_children.append(dcc.Graph(id=graph_id, style={'display': 'none'}))

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
                            dcc.Dropdown(tools.model_types,
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


@app.callback(
    Output('first_col', 'children'),
    Output('third_col', 'children'),
    Input('dropdown', 'value'),
    State('first_col', 'children'),
    State('third_col', 'children'))
def change_first_third_cols(model_type, first_col, third_col):
    global ERROR_STATE
    if ERROR_STATE:
        ERROR_STATE = False
        return first_col, third_col
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
              Input('output-table', 'columns'),
              State('input-graph', 'style'),
              State('output-graph', 'style'))
def update_graphs(rows_input, cols_input, rows_output,
                       cols_output, style_input, style_output):
    figures = []
    for i in range(2):
        if i == 0:
            rows = rows_input
            cols = cols_input
            style = style_input
        else:
            rows = rows_output
            cols = cols_output
            style = style_output
        if style is None:
            # dim == 2 or dim == 7
            cols = [c['name'] for c in cols]
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
        else:
            # dim == 42
            fig = {'data': None, 'layout': None}
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
        global ERROR_STATE
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            cur_path_to_tempfile = tools.create_tempfile_from_content(data)
            # Check model_type_file
            model_type_file = tools.get_model_type(name, cur_path_to_tempfile)
            tempfile_list.append((name, cur_path_to_tempfile, model_type_file))
            if not model_type_file:
                tools.delete_tempfiles(tempfile_list)
                displayed = True
                message = f'Модель {name} имеет некорректный формат.'
                ERROR_STATE = True
                return model_type_dropdown, displayed, message
            # Add model_file info
        unique_types_model_interfaces = set(map(lambda x: x[2], tempfile_list))
        # Check all - unique
        if len(unique_types_model_interfaces) != len(tempfile_list):
            tools.delete_tempfiles(tempfile_list)
            displayed = True
            most_frequent_t = tools.get_most_frequent(tempfile_list, temp=True)
            message = f'Вы выбрали несколько моделей одинакового вида. \
Модель вида {most_frequent_t[0]} встречается {most_frequent_t[1]} раз.'
            ERROR_STATE = True
            return model_type_dropdown, displayed, message
        all_types_models = [tools.type_model_interface_key_to_type_model_key(m)
                              for m in unique_types_model_interfaces]
        if len(set(all_types_models)) != len(unique_types_model_interfaces):
            tools.delete_tempfiles(tempfile_list)
            displayed = True
            most_frequent_t = tools.get_most_frequent(all_types_models)
            message = f'Вы выбрали несколько моделей одинакового типа. \
Модель типа {" -> ".join(most_frequent_t[0].split("_"))} встречается {most_frequent_t[1]} раз.'
            ERROR_STATE = True
            return model_type_dropdown, displayed, message
        cur_attrs_model_facade = {}
        try:
            # Save current model_facade models (attrs)
            for model_attr in tools.type_model_dict.values():
                cur_attrs_model_facade[model_attr] = getattr(model_facade, model_attr)
            # Create new model_facade models (attrs)
            for cur_filename, cur_path_to_tempfile, model_type_file in tempfile_list:
                type_model_key = tools.type_model_interface_key_to_type_model_key(model_type_file)
                model_attr_name = cur_attr = tools.type_model_dict[type_model_key]
                model_attr_val = getattr(sys.modules[__name__],
                        type_model_interface_dict[model_type_file])(cur_filename, cur_path_to_tempfile)
                setattr(model_facade, model_attr_name, model_attr_val)
            tools.delete_tempfiles(tempfile_list)
        except Exception:
            # Error while creating one of model_facade's models
            print('Exception')
            print(traceback.format_exc())
            tools.delete_tempfiles(tempfile_list)
            # Откат до cur_attrs_model_facade
            for model_attr_name, model_attr_val in cur_attrs_model_facade.items():
                setattr(model_facade, model_attr_name, model_attr_val)
            # Window 'Cant create model...'
            displayed = True
            message = f'Не удаётся создать модель типа {model_type_file} из файла {cur_filename}.'
            ERROR_STATE = True
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
        model_attr_val = getattr(model_facade, tools.type_model_dict[model_type])
        if model_attr_val is not None:
            disabled_button = False
            upload_children = html.Div(
                [f"Модель '{model_attr_val.filename}' успешно загружена."]
            )
        else:
            disabled_button = True

    return disabled_button, upload_children


@app.callback(Output('output-table', 'data'),
              Input('calculate', 'n_clicks'),
              State('dropdown', 'value'),
              State('input-table', 'data'),
              State('input-table', 'columns'),
              State('output-table', 'data'),
              State('input-graph', 'style'),
              State('output-graph', 'style'))
def update_output_table(n_clicks, model_type_dropdown, rows_input,
                        cols_input, rows_output, style_input, style_output):
    global N_CLICKS
    data = rows_output
    if n_clicks != N_CLICKS:
        try:
            type_model = model_type_dropdown.replace(' -> ', '_')
            model_attr_name = tools.type_model_dict[type_model]
            model_attr_val = getattr(model_facade, model_attr_name)
            if style_input is None:
                df = tools.data_table_to_data_frame(rows_input, cols_input)
            else:
                df = tools.data_table_to_data_frame(rows_input, cols_input, T=True)
            output_df = model_attr_val.predict(df)
            if style_output is None:
                data = tools.data_frame_to_data_table(output_df)
            else:
                data = tools.data_frame_to_data_table(output_df, T=True)
        except Exception:
            traceback.format_exc()
            N_CLICKS = n_clicks
            raise
        N_CLICKS = n_clicks
    return data


cards = dbc.CardGroup([
    create_first_or_third(),
    create_second(),
    create_first_or_third(first=False)
])

app.layout = cards

if __name__ == '__main__':
    app.run_server(debug=True)
