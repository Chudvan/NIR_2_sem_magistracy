import os
import sqlite3
import sys

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import display_html
from itertools import chain,cycle
import tarfile
import tempfile
import base64


pa_fields = [
    'Valence', # x
    'Arousal' # y
]

seven_fields = [
    'Neutral', 
    'Happy', 
    'Sad', 
    'Angry', 
    'Surprised', 
    'Scared', 
    'Disgusted'
]

facs_fields = [
    'Action_Unit_01___Inner_Brow_Raiser',
    'Action_Unit_02___Outer_Brow_Raiser',
    'Action_Unit_04___Brow_Lowerer',
    'Action_Unit_05___Upper_Lid_Raiser',
    'Action_Unit_06___Cheek_Raiser',
    'Action_Unit_07___Lid_Tightener',
    'Action_Unit_09___Nose_Wrinkler',
    'Action_Unit_10___Upper_Lip_Raiser',
    'Action_Unit_12___Lip_Corner_Puller',
    'Action_Unit_14___Dimpler',
    'Action_Unit_15___Lip_Corner_Depressor',
    'Action_Unit_17___Chin_Raiser',
    'Action_Unit_18___Lip_Pucker',
    'Action_Unit_20___Lip_Stretcher',
    'Action_Unit_23___Lip_Tightener',
    'Action_Unit_24___Lip_Pressor',
    'Action_Unit_25___Lips_Part',
    'Action_Unit_26___Jaw_Drop',
    'Action_Unit_27___Mouth_Stretch',
    'Action_Unit_43___Eyes_Closed',
    'Action_Unit_01___Left___Inner_Brow_Raiser',
    'Action_Unit_02___Left___Outer_Brow_Raiser',
    'Action_Unit_04___Left___Brow_Lowerer',
    'Action_Unit_05___Left___Upper_Lid_Raiser',
    'Action_Unit_06___Left___Cheek_Raiser',
    'Action_Unit_07___Left___Lid_Tightener',
    'Action_Unit_12___Left___Lip_Corner_Puller',
    'Action_Unit_14___Left___Dimpler',
    'Action_Unit_15___Left___Lip_Corner_Depressor',
    'Action_Unit_20___Left___Lip_Stretcher',
    'Action_Unit_43___Left___Eyes_Closed',
    'Action_Unit_01___Right___Inner_Brow_Raiser',
    'Action_Unit_02___Right___Outer_Brow_Raiser',
    'Action_Unit_04___Right___Brow_Lowerer',
    'Action_Unit_05___Right___Upper_Lid_Raiser',
    'Action_Unit_06___Right___Cheek_Raiser',
    'Action_Unit_07___Right___Lid_Tightener',
    'Action_Unit_12___Right___Lip_Corner_Puller',
    'Action_Unit_14___Right___Dimpler',
    'Action_Unit_15___Right___Lip_Corner_Depressor',
    'Action_Unit_20___Right___Lip_Stretcher',
    'Action_Unit_43___Right___Eyes_Closed'
]

happy_index = [4, 24, 35, 8, 26, 37]
sad_index = [0, 20, 31, 2, 22, 33, 10, 28, 39]
surprised_index = [0, 20, 31, 1, 21, 32, 3, 23, 34, 17]
scared_index = [0, 20, 31, 1, 21, 32, 2, 22, 33, 3, 23, 34, 5, 25, 36, 13, 29, 40, 17]
angry_index = [2, 22, 33, 3, 23, 34, 5, 25, 36, 14]
disgusted_index = [6, 10, 28, 39]
contempt_index = [8, 26, 37, 9, 27, 38]
other_facs_index = [7, 11, 12, 15, 16, 18, 19, 30, 41]
all_unique_index = list(range(20))

naming_list = ['happy', 'sad', 'surprised', 'scared', 'angry',
               'disgusted', 'contempt', 'other_facs', 'all_unique']

for name in naming_list:
    index_var = getattr(sys.modules[__name__], name + '_index')
    setattr(sys.modules[__name__], name + '_fields', [facs_fields[i] for i in index_var])

model_types = [
    {'label': '2 -> 7', 'value': '2_7'},
    {'label': '7 -> 2', 'value': '7_2'},
    {'label': '7 -> 42', 'value': '7_42'},
    {'label': '42 -> 7', 'value': '42_7'},
    {'label': '2 -> 42', 'value': '2_42'},
    {'label': '42 -> 2', 'value': '42_2'},
]

type_model_dict = {
    '2_7': 'model_va_clear',
    '7_2': 'model_clear_va',
    '7_42': 'model_clear_facs',
    '42_7': 'model_facs_clear',
    '2_42': 'model_va_facs',
    '42_2': 'model_facs_va'
}

fields = seven_fields + pa_fields

metrics = ['mean', 'norm', 'stat']

clear_count_dict = {
    'Neutral': 200,
    'Happy': 200,
    'Sad': 14,
    'Angry': 44,
    'Surprised': 30,
    'Scared': 12,
    'Disgusted': 30
}


def save_to_db(db_path, name_db, df):
    connection = sqlite3.connect(db_path)
    df_columns = [field.replace('-', '_') for field in df.columns]
    df_columns = [field.replace(' ', '_') for field in df_columns]
    try:
        i = df_columns.index('3d_Landmarks')
        df_columns[i] = 'three_d_Landmarks'
    except ValueError:
        pass
    fields = ',\n'.join([f'\t{field} TEXT' for field in df_columns])
    create_costs_table_query = f"""
create table {name_db} (
{fields}
)
"""
    connection.execute(create_costs_table_query)
    connection.commit()
    values = ', '.join(['?' for _ in range(len(df.columns))])
    for row in df.iterrows():
        connection.execute(f"INSERT OR IGNORE INTO {name_db} VALUES({values})", tuple(row[1]))
    connection.commit()
    return connection

def groupby(df, by=None, prediction=2, other=False, other_groupby=True):
    if by is None:
        by = pa_fields
        
    df_copy = df[seven_fields + pa_fields].copy()
    
    for field in pa_fields:
        df_copy[field] = df_copy[field].apply(lambda x: round(float(x), prediction))
    for field in seven_fields:
        df_copy[field] = df_copy[field].apply(lambda x: float(x))
    
    df_copy.index = df['Index_']
    
    groupby_fields_sorted = list(sorted(df_copy.groupby(by), key=lambda x: -len(x[1])))
    for group in groupby_fields_sorted:
        for field in seven_fields:
            group[1][field] = round(group[1][field].mean(), prediction)
            
    df_train = pd.DataFrame()
    if other:
        df_other = pd.DataFrame()
    
    for group in groupby_fields_sorted:
        len_group = len(group[1])
        ln_ = np.log10(len_group)
        rand_set = set()
        for _ in range(int(round(ln_, 0)) + 1):
            i = random.randint(0, len_group - 1)
            while i in rand_set:
                i = random.randint(0, len_group - 1)
            rand_set.add(i)
            df_train = pd.concat([df_train, group[1].iloc[i:i + 1]], axis=0)
        if other:
            all_i_without_rand_set = set(range(len_group)) - rand_set
            if other_groupby:
            	df_other = pd.concat([df_other, group[1].iloc[list(all_i_without_rand_set)]], axis=0)
            else:
            	df_other = pd.concat([df_other, df.iloc[list(all_i_without_rand_set)]], axis=0)
    if other:
        for field in seven_fields + pa_fields:
            df_other[field] = df_other[field].apply(lambda x: float(x))
        return df_train, df_other[seven_fields + pa_fields]
    return df_train

def apply_float(df_, columns):
    for field in columns:
        df_[field] = df_[field].apply(lambda el: float(el))
        
def make_valid_df(df_, columns=None):
    if columns is not None:
        apply_float(df_, columns)
    df_.index = df_['Index_']
    
def refitting(models, test, df_metrics, df_train=None, v=1, 
              layer='first', epochs=20, batch_size=20, type_='diff'):
    for nn_list in models:
        nn_list[0] = nn_list[0].split('_')[0] + f'_{v}'
        nn = nn_list[2]
        print('refit', nn_list[0])
        if type_ == 'diff':
            df_train = nn.create_train_df_from_diff(test)
        elif type_ == 'split' and df_train is not None:
            pass
        else:
            raise Exception('Unknown refitting type.')
        nn.fit(df_train, epochs=epochs, batch_size=batch_size)
        entry_dict = {'model': nn_list[0], 'layer': layer, 'N': nn_list[1]}
        entry_dict.update({metric: nn.model_metric(test, metric) for metric in metrics})
        df_metrics = df_metrics.append(entry_dict, ignore_index = True)
        print(entry_dict)
    return df_metrics

def plot_emotions(models, df_clear, df_clear_metrics, scale=False, figsize=(20, 15)):
    plt.figure(figsize=figsize)
    for i, model_tuple in enumerate(models):
        entry_dict = {'model': model_tuple[0]}
        nn = model_tuple[2]
        clear_metric, emotion_mean_values = nn.model_metric(df_clear, 'clear', scale=scale)
        entry_dict.update({'clear': clear_metric})
        for j, emotion in enumerate(df_clear.columns[:7]):
            entry_dict.update({emotion: emotion_mean_values[j]})
        
        plt.plot(seven_fields, emotion_mean_values, label=model_tuple[0])
        # entry_dict.update({metric: df_metrics.iloc[i][metric] for metric in metrics})
        df_clear_metrics = df_clear_metrics.append(entry_dict, ignore_index = True)
    plt.xlabel("Эмоции")
    plt.ylabel("Средние значения предсказанных чистых эмоций / Средние значения чистых эмоций")
    plt.legend()
    plt.show()
    return df_clear_metrics

def create_metric_df_dict(metrics, df_metrics, df_clear_metrics):
    metric_df_dict = {metric: df_metrics for metric in metrics[:-1]}
    metric_df_dict.update({metrics[-1]: df_clear_metrics})
    return metric_df_dict

def plot_metrics(metric_df_dict, layer='first'):
    # dependencies
    mean_ = 'mean'
    clear = 'clear'
    
    x = []
    y = []
    
    df_metrics = metric_df_dict[mean_]
    metrics = list(metric_df_dict.keys())
    
    for metric in metrics:
        if layer == 'first':
            x.append(df_metrics['N'])
        else:
            x.append(df_metrics.index)
        df_ = metric_df_dict[metric]
        y.append(df_[metric])
    
    for i in range(len(metrics)):
        plt.plot(x[i], y[i], label=metrics[i])
        plt.xlabel("Число нейронов N в 1 скрытом слое")
        if metrics[i] == clear:
            plt.ylabel("Сумма средних значений предсказанных чистых эмоций / Сумму средних значений чистых эмоций")
        else:
            plt.ylabel("Ошибка")
        plt.legend()
        plt.show()

def save_models(models, path_to_saved_models, layer='first', v=1):
    dir_path = os.path.join(path_to_saved_models, layer, f'_{v}')
    for model_list in models:
        N = model_list[1]
        nn = model_list[2]
        save_name = f'model_{layer}_{N}_{v}'
        path = os.path.join(dir_path, save_name)
        nn.model.save(path)

def _removeprefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def load_models(path_to_saved_models, df, models_list=None, layer='first', 
                v=1, sort=True, csv_test_file=None):
    from tensorflow.keras.models import load_model
    from nn_train.neural_network import NeuralNetwork

    dir_path = os.path.join(path_to_saved_models, layer, f'_{v}')
    if models_list is None:
        models = [el for el in list(os.walk('..')) if dir_path in el[0]][0][1]
    else:
        prefix = f'model_{layer}_'
        if (layer == 'first' or layer == 'third') and v == 1:
            models = [prefix + N for N in models_list]
        elif layer == 'third' and v != 2:
            models = [N + f'_{v}' for N in models_list]
        else:
            models = [prefix + N + f'_{v}' for N in models_list]
#     print(models)
    for i in range(len(models)):
        model_layers_v = _removeprefix(models[i], f'model_{layer}_')
        N = model_layers_v.split('_')[0]
        path = os.path.join(dir_path, models[i])
        model = load_model(path)
        nn = NeuralNetwork(df[pa_fields], df[seven_fields], model, csv_test_file)
        models[i] = [model_layers_v, N, nn]
    
    if sort:
    	models.sort(key=lambda x: list(map(int, x[1].split('.'))))
    
    return models

def create_df_metrics(models, test, df_metrics, layer='first'):
    for model_list in models:
        entry_dict = {'model': model_list[0], 'layer': layer, 'N': model_list[1]}
        entry_dict.update({metric: model_list[2].model_metric(test, metric) for metric in metrics})
        df_metrics = df_metrics.append(entry_dict, ignore_index = True)
    return df_metrics
    
def display_dfs(*args, titles=cycle(['']), mode='column'):
    html_str=''
    for df,title in zip(args, chain(titles,cycle(['</br>']))):
        cur_html_str = '<td style="vertical-align:top">'
        if mode == 'column':
            cur_html_str+=f'<h2 align="left">{title}</h2>'
        elif mode == 'row':
            cur_html_str+=f'<h2 align="center">{title}</h2>'
        else:
            raise Exception(f'Unknown mode: {mode}')
        cur_html_str+=df.to_html().replace('table','table style="display:inline" align="left"')
        cur_html_str+='</td>'
        if mode == 'column':
            cur_html_str = '<tr align="left">' + cur_html_str + '</tr>'
        elif mode == 'row':
            pass
        else:
            raise Exception(f'Unknown mode: {mode}')
        html_str += cur_html_str
    display_html(html_str,raw=True)

def get_model_type(filename, path_to_tempfile):
    from model_interfaces import type_model_interface_dict
    _, file_extension = os.path.splitext(filename)
    if file_extension != '.gz':
        return False
    with tarfile.open(path_to_tempfile, 'r:gz') as tar:
        try:
            type_filename = 'type'
            model_type = tar.extractfile(type_filename).read().decode().strip()
        except KeyError:
            return False
        if model_type not in type_model_interface_dict:
            return False
    return model_type

def create_tempfile_from_content(data):
    data = data.encode("utf8").split(b";base64,")[1]
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(base64.decodebytes(data))
        path_to_tempfile = f.name
    return path_to_tempfile

def delete_tempfiles(tempfile_list):
    for path_to_tempfile in list(map(lambda x: x[1], tempfile_list)):
        os.remove(path_to_tempfile)

def get_most_frequent(list_, temp=False):
    d = {}
    if temp:
        for t in list_:
            if t[2] not in d:
                d[t[2]] = 0
            d[t[2]] += 1
    else:
        for e in list_:
            if e not in d:
                d[e] = 0
            d[e] += 1
    return sorted(list(d.items()), key=lambda x: x[1])[-1]

def type_model_interface_key_to_type_model_key(key):
    return '_'.join(key.split(' ')[0].split('->'))

def data_table_to_data_frame(rows, cols, T=False):
    cols = [c['name'] for c in cols]
    df = pd.DataFrame(rows, columns=cols)
    if T:
        df = df.T
        rows = list(df.iterrows())
        cols = list(rows[0][1])
        vals = list(rows[1][1])
        df = pd.DataFrame([vals], columns=cols)
    for c in cols:
        df[c] = df[c].apply(lambda x: float(x))
    return df

def data_frame_to_data_table(df, T=False):
    if T: # dim == 42
        cols = df.columns
        vals = df.values[0]
        fields = ['Action name', 'Value']
        data = [{fields[0]: col, fields[1]: val} for col, val in zip(cols, vals)]
        return data
    l = [] # dim == 2, dim == 7
    for row in df.iterrows():
        l.append(dict(row[1]))
    return l

def change_df_accuracy(df, digits=2):
    for c in df.columns:
        df[c] = df[c].apply(lambda x: f"{x:.2f}")
    return df

def cast_to_float(vals):
    return [float(v) for v in vals]
