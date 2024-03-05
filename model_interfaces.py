import traceback
from abc import ABC, abstractmethod
import GUI.tools as tools

import tarfile
import os
import sys
import shutil
import pandas as pd
from tensorflow.keras.models import load_model
from joblib import load
import pickle


DIR_PATH = '/tmp'
TYPE_FILENAME = 'type'
FACADE_CLASS_NAME = 'ModelFacade'
MODEL_CLASS_PREFIX = 'Model'

class ModelFacade:
    def __init__(self):
        self.model_va_clear = None
        self.model_clear_va = None
        self.model_clear_facs = None
        self.model_facs_clear = None
        self.model_va_facs = None
        self.model_facs_va = None

    @property
    def model_va_clear(self):
        return self._model_va_clear

    @model_va_clear.setter
    def model_va_clear(self, model):
        self._model_va_clear = model

    @property
    def model_clear_va(self):
        return self._model_clear_va

    @model_clear_va.setter
    def model_clear_va(self, model):
        self._model_clear_va = model

    @property
    def model_clear_facs(self):
        return self._model_clear_facs

    @model_clear_facs.setter
    def model_clear_facs(self, model):
        self._model_clear_facs = model

    @property
    def model_facs_clear(self):
        return self._model_facs_clear

    @model_facs_clear.setter
    def model_facs_clear(self, model):
        self._model_facs_clear = model

    @property
    def model_va_facs(self):
        return self._model_va_facs

    @model_va_facs.setter
    def model_va_facs(self, model):
        self._model_va_facs = model

    @property
    def model_facs_va(self):
        return self._model_facs_va

    @model_facs_va.setter
    def model_facs_va(self, model):
        self._model_facs_va = model

    def predict(self, type_, input_df):
        return getattr(self, tools.type_model_dict[type_]).predict(input_df)


class AbstractModel(ABC):
    _model = None
    _file_name = None
    _MODEL_ATTR_PREFIX = '_model'

    def __init__(self, filename, path):
        self.filename = filename
        self.loadmodel(path)

    @classmethod
    def unzip_model(cls, path):
        with tarfile.open(path, 'r:gz') as tar:
            type_model = tools.type_model_interface_key_to_type_model_key(cls.type_)
            model_attr = tools.type_model_dict[type_model]
            dir_path = os.path.join(DIR_PATH, model_attr)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            for file in tar:
                if file.name != TYPE_FILENAME:
                    tar.extract(file.name, dir_path)
        return dir_path

    @classmethod
    @property
    @abstractmethod
    def type_(cls):
        pass

    @property
    def filename(self):
        return self._file_name

    @filename.setter
    def filename(self, val):
        self._file_name = val

    @abstractmethod
    def loadmodel(self, path):
        pass

    def _get_model_attrs(self):
        model_attrs = []
        for attr in dir(self):
            if attr.startswith(self._MODEL_ATTR_PREFIX):
                model_attrs.append(attr)
        return model_attrs

    @abstractmethod
    def predict(self, input_):
        pass


def load_models(self, path):
    dir_path = self.unzip_model(path)
    model_attrs = self._get_model_attrs()
    if len(os.listdir(dir_path)) < len(model_attrs):
        raise Exception(f'Число моделей < {len(model_attrs)}.')
    print(model_attrs)
    print(list(os.walk(dir_path)))
    model_files_list = list(os.walk(dir_path))[0][1:]
    model_files_list = model_files_list[0] + model_files_list[1]
    model_files_ext_dict = {} # dict с ext для файлов моделей
    for model_filename in model_files_list:
    	model_name, ext = os.path.splitext(model_filename)
    	model_files_ext_dict[model_name] = ext
    print(model_files_ext_dict)
    for model_attr in model_attrs:
        model_name = model_attr[1:]
        print(model_name)
        if model_name not in model_files_ext_dict:
    	    raise Exception(f'Модель {model_name} отсутствует в файле {self.filename}.')
        ext = model_files_ext_dict[model_name]
        model_filename = model_name + ext
        full_path = os.path.join(dir_path, model_filename)
        print(full_path)
        try:
            if ext=='.pkl':
                with open(full_path, 'rb') as file:
                    model_val = pickle.load(file)
            elif ext=='.joblib':
                model_val = load(full_path)
            elif not ext:
                model_val = load_model(full_path)
            else:
                raise Exception(f'Неизвестное расширение {ext}.')
            setattr(self, model_attr, model_val)
            print(getattr(self, model_attr))
        except Exception:
            print(traceback.format_exc())
            shutil.rmtree(dir_path)
            raise Exception(f'Не удаётся создать модель {model_filename}.')
    shutil.rmtree(dir_path)


def load_VA_and_FACS_model(self, path):
    dir_path = self.unzip_model(path)
    model_attrs = self._get_model_attrs()
    try:
        for model_attr in model_attrs:
            model_filename = model_attr[1:] + '.tar.gz'
            full_path = os.path.join(dir_path, model_filename)
            print(full_path)
            model_type_file = tools.get_model_type(model_filename, full_path)
            print(model_type_file)
            model_attr_val = getattr(sys.modules[__name__],
                                     type_model_interface_dict[model_type_file])(model_filename, full_path)
            setattr(self, model_attr, model_attr_val)
            print(model_attr, model_attr_val)
    except Exception:
        print(traceback.format_exc())
        shutil.rmtree(dir_path)
        raise
    shutil.rmtree(dir_path)


class ModelVAClearNeural(AbstractModel):
    @classmethod
    @property
    def type_(cls):
        return '2->7 (Neural)'

    def loadmodel(self, path):
        load_models(self, path)

    def predict(self, df_VA):
        seven_vals = self._model.predict(df_VA.values)
        df_seven = pd.DataFrame(seven_vals, columns=tools.seven_fields)
        df_seven = tools.change_df_accuracy(df_seven)
        return df_seven


class ModelVAClearStat(AbstractModel):
    _model_neutral = None
    _model_happy = None
    _model_sad = None
    _model_angry = None
    _model_surprised = None
    _model_scared = None
    _model_disgusted = None
    _MODEL_ATTR_PREFIX = '_model_'

    @classmethod
    @property
    def type_(cls):
        return '2->7 (Stat)'

    def loadmodel(self, path):
        load_models(self, path)

    def predict(self, df_VA):
        model_attrs = self._get_model_attrs()
        seven_dict = {}
        for model_attr in model_attrs:
            field = model_attr[len(self._MODEL_ATTR_PREFIX):].capitalize()
            seven_dict[field] = getattr(self, model_attr).predict(df_VA.values)
        seven_vals = [[seven_dict[field][0][i] for i, field in enumerate(tools.seven_fields)]] # N без [0][i] для correct
        #seven_dict = {field: seven_dict[field][0][i] for i, field in enumerate(tools.seven_fields)} # по сути, не нужно уже
        df_seven = pd.DataFrame(seven_vals, columns=tools.seven_fields)
        df_seven = tools.change_df_accuracy(df_seven)
        return df_seven


class ModelClearVANeural(AbstractModel):
    @classmethod
    @property
    def type_(cls):
        return '7->2 (Neural)'

    def loadmodel(self, path):
        load_models(self, path)

    def predict(self, df_seven):
        VA_vals = self._model.predict(df_seven.values)
        df_VA = pd.DataFrame(VA_vals, columns=tools.pa_fields)
        df_VA = tools.change_df_accuracy(df_VA)
        return df_VA


class ModelClearVAStat(AbstractModel):
    _model_valence = None
    _model_arousal = None
    _MODEL_ATTR_PREFIX = '_model_'

    @classmethod
    @property
    def type_(cls):
        return '7->2 (Stat)'

    def loadmodel(self, path):
        load_models(self, path)

    def predict(self, df_seven):
        v = self._model_valence.predict(df_seven[df_seven.columns[:2]].values)[0][0] # N без [df_seven.columns[:2]] и мб без [0] для correct
        a = self._model_arousal.predict(df_seven[df_seven.columns[2:4]].values)[0][0] # аналогично
        df_VA = pd.DataFrame([[v, a]], columns=tools.pa_fields)
        df_VA = tools.change_df_accuracy(df_VA)
        return df_VA


class ModelClearFACSNeural(AbstractModel):
    _model_happy = None
    _model_sad = None
    _model_angry = None
    _model_surprised = None
    _model_scared = None
    _model_disgusted = None
    _model_contempt = None
    _model_other_facs = None
    _model_sum_01_02_26 = None
    _model_sum_04_05_07 = None
    _model_sum_12_15 = None
    _MODEL_ATTR_PREFIX = '_model_'

    @classmethod
    @property
    def type_(cls):
        return '7->42 (Neural)'

    def loadmodel(self, path):
        load_models(self, path)

    def predict(self, df_seven):
        happy_vals = self._model_happy.predict([list(df_seven.values[0][1:3])])[0][:6] # N просто df_seven.values + без [:6] для correct
        happy_vals = tools.cast_to_float(happy_vals)
        sad_vals = self._model_sad.predict([list(df_seven.values[0][1:3])])[0] # по аналогии
        sad_vals = list(sad_vals) # лишнее, для correct
        sad_vals += sad_vals[-2:] # чтобы dim совпадал
        sad_vals = tools.cast_to_float(sad_vals)
        surprised_vals = self._model_surprised.predict([list(df_seven.values[0][1:3])])[0]
        surprised_vals = list(surprised_vals)
        surprised_vals +=  surprised_vals[-3:]
        surprised_vals = tools.cast_to_float(surprised_vals)
        scared_vals = self._model_scared.predict([list(df_seven.values[0][1:3])])[0]
        scared_vals = list(scared_vals)
        scared_vals += scared_vals + scared_vals[-5:]
        scared_vals = tools.cast_to_float(scared_vals)
        angry_vals = self._model_angry.predict([list(df_seven.values[0][1:3])])[0]
        angry_vals = list(angry_vals)
        angry_vals += angry_vals[-3:]
        angry_vals = tools.cast_to_float(angry_vals)
        disgusted_vals = self._model_disgusted.predict([list(df_seven.values[0][1:3])])[0][-4:]
        disgusted_vals = tools.cast_to_float(disgusted_vals)
        contempt_vals = self._model_contempt.predict([list(df_seven.values[0][1:3])])[0][-6:]
        contempt_vals = tools.cast_to_float(contempt_vals)
        other_facs_vals = self._model_other_facs.predict([list(df_seven.values[0][1:3])])[0]
        other_facs_vals = list(other_facs_vals)
        other_facs_vals += other_facs_vals[-2:] # N до сюда - всё по аналогии, как выше
        other_facs_vals = tools.cast_to_float(other_facs_vals)

        # 3 модели - использовать
        # собираем по индексам значения
        _01_02_26_input = sad_vals[:3] + surprised_vals[:3] + scared_vals[:3] + \
                          surprised_vals[3:6] + scared_vals[3:6] + surprised_vals[-1:] +\
                          scared_vals[-1:]
        _04_05_07_input = sad_vals[3:6] + scared_vals[6:9] + angry_vals[:3] + \
                          surprised_vals[6:9] + scared_vals[9:12] + angry_vals[3:6] + \
                          scared_vals[12:15] + angry_vals[6:9]
        _12_15_input = happy_vals[3:6] + contempt_vals[:3] + sad_vals[6:9] + disgusted_vals[1:]

        _01_02_26_output = list(self._model_sum_01_02_26.predict([_01_02_26_input[1:3]])[0]) # N без [1:3]
        _04_05_07_output = list(self._model_sum_04_05_07.predict([_04_05_07_input[1:3]])[0]) # list() не нужен обычно
        _04_05_07_output += _04_05_07_output[-2:] # лишнее, для correct
        _12_15_output = list(self._model_sum_12_15.predict([_12_15_input[1:3]])[0][-6:]) # ещё без [-6:]

        # конечный сбор (42)
        res_42_vals = _01_02_26_output[:1] + _01_02_26_output[3:4] + _04_05_07_output[:1] + \
                      _04_05_07_output[3:4] + happy_vals[:1] + _04_05_07_output[6:7] + \
                      disgusted_vals[:1] + other_facs_vals[:1] + contempt_vals[:1] + \
                      contempt_vals[3:4] + _12_15_output[3:4] + other_facs_vals[1:3] + \
                      scared_vals[-4:-3] + angry_vals[-1:] + other_facs_vals[3:5] + \
                      _01_02_26_output[-1:] + other_facs_vals[5:7] + _01_02_26_output[1:2] + \
                      _01_02_26_output[4:5] + _04_05_07_output[1:2] + _04_05_07_output[4:5] + \
                      happy_vals[1:2] + _04_05_07_output[-2:-1] + _12_15_output[1:2] + \
                      contempt_vals[-2:-1] + _12_15_output[-2:-1] + scared_vals[-3:-2] + \
                      other_facs_vals[-2:-1] + _01_02_26_output[2:3] + _01_02_26_output[-2:-1] + \
                      _04_05_07_output[2:3] + _04_05_07_output[5:6] + happy_vals[2:3] + \
                      _04_05_07_output[-1:] + _12_15_output[2:3] + contempt_vals[-1:] + \
                      _12_15_output[-1:] + scared_vals[-2:-1] + other_facs_vals[-1:]
        df_42 = pd.DataFrame([res_42_vals], columns=tools.facs_fields)
        df_42 = tools.change_df_accuracy(df_42)
        return df_42


class ModelFACSClearStat(AbstractModel):
    _model_neutral = None
    _model_happy = None
    _model_sad = None
    _model_angry = None
    _model_surprised = None
    _model_scared = None
    _model_disgusted = None
    _MODEL_ATTR_PREFIX = '_model_'

    @classmethod
    @property
    def type_(cls):
        return '42->7 (Stat)'

    def loadmodel(self, path):
        load_models(self, path)

    def predict(self, df_42):
        happy_vals = df_42[tools.happy_fields].values
        sad_vals = df_42[tools.sad_fields].values
        surprised_vals = df_42[tools.surprised_fields].values
        scared_vals = df_42[tools.scared_fields].values
        angry_vals = df_42[tools.angry_fields].values
        disgusted_vals = df_42[tools.disgusted_fields].values
        all_unique_vals = df_42[tools.all_unique_fields].values

        neutral_res = self._model_neutral.predict([list(all_unique_vals[0][1:3])])[0][0] # N просто all_unique_vals для correct
        happy_res = self._model_happy.predict([list(happy_vals[0][1:3])])[0][0] # аналогично
        sad_res = self._model_sad.predict([list(sad_vals[0][1:3])])[0][0] # аналогично
        angry_res = self._model_angry.predict([list(angry_vals[0][1:3])])[0][0] # аналогично
        surprised_res = self._model_surprised.predict([list(surprised_vals[0][1:3])])[0][0] # аналогично
        scared_res = self._model_scared.predict([list(scared_vals[0][1:3])])[0][0] # аналогично
        disgusted_res = self._model_disgusted.predict([list(disgusted_vals[0][1:3])])[0][0] # аналогично

        df_seven = pd.DataFrame([[neutral_res, happy_res, sad_res,
                                  angry_res, surprised_res, scared_res,
                                  disgusted_res]], columns=tools.seven_fields)
        df_seven = tools.change_df_accuracy(df_seven)
        return df_seven


class ModelFACSEkmanOne(AbstractModel):
    @classmethod
    @property
    def type_(cls):
        return '42->7 (one)'

    def loadmodel(self, path):
        load_models(self, path)

    def predict(self, df_42):
        seven_vals = self._model.predict(df_42.values)
        df_seven = pd.DataFrame(seven_vals, columns=tools.seven_fields)
        df_seven = tools.change_df_accuracy(df_seven)
        return df_seven


class ModelVAFACS(AbstractModel):
    _model_va_clear = None
    _model_clear_facs = None
    _MODEL_ATTR_PREFIX = '_model_'

    @classmethod
    @property
    def type_(cls):
        return '2->42'

    def loadmodel(self, path):
        load_VA_and_FACS_model(self, path)

    def predict(self, df_VA):
        df_seven = self._model_va_clear.predict(df_VA)
        df_seven = pd.DataFrame([tools.cast_to_float(df_seven.values[0])], columns=tools.seven_fields)
        df_42 = self._model_clear_facs.predict(df_seven)
        return df_42


class ModelFACSVA(AbstractModel):
    _model_facs_clear = None
    _model_clear_va = None
    _MODEL_ATTR_PREFIX = '_model_'

    @classmethod
    @property
    def type_(cls):
        return '42->2'

    def loadmodel(self, path):
        load_VA_and_FACS_model(self, path)

    def predict(self, df_42):
        df_seven = self._model_facs_clear.predict(df_42)
        df_seven = pd.DataFrame([tools.cast_to_float(df_seven.values[0])], columns=tools.seven_fields)
        df_VA = self._model_clear_va.predict(df_seven)
        return df_VA


def create_abstract_model_inheritors_list():
    abstract_model_inheritors_list = [
        class_ for class_ in dir(sys.modules[__name__])
        if class_.startswith(MODEL_CLASS_PREFIX)
    ]
    i = abstract_model_inheritors_list.index(FACADE_CLASS_NAME)
    abstract_model_inheritors_list.pop(i)
    return abstract_model_inheritors_list

abstract_model_inheritors_list = create_abstract_model_inheritors_list()

type_model_interface_dict = {
    getattr(sys.modules[__name__], model).type_:
        model for model in abstract_model_inheritors_list
}
