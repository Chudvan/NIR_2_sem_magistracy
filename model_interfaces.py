from abc import ABC, abstractmethod
from GUI.tools import seven_fields, type_model_dict, change_df_accuracy, \
    type_model_interface_key_to_type_model_key, pa_fields
import tarfile
import os
import sys
import shutil
import pandas as pd
from tensorflow.keras.models import load_model
import pickle


DIR_PATH = '/tmp'
TYPE_FILENAME = 'type'

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
        return getattr(self, type_model_dict[type_]).predict(input_df)


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
            type_model = type_model_interface_key_to_type_model_key(cls.type_)
            model_attr = type_model_dict[type_model]
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


abstract_model_inheritors_list = ['ModelVAClearNeural', 'ModelVAClearStat',
                                  'ModelClearVANeural', 'ModelClearVAStat',
                                  'ModelClearFACSStat', 'ModelFACSClearStat',
                                  'ModelVAFACSStat', 'ModelFACSVAStat']


def load_neural_model(self, path):
    dir_path = self.unzip_model(path)
    if len(os.listdir(dir_path)) != 1:
        raise Exception('Число папок с моделями != 1.')
    full_path = os.path.join(dir_path, os.listdir(dir_path)[0])
    try:
        self._model = load_model(full_path)
    except Exception:
        shutil.rmtree(dir_path)
        raise
    shutil.rmtree(dir_path)


class ModelVAClearNeural(AbstractModel):
    @classmethod
    @property
    def type_(cls):
        return '2->7 (Neural)'

    def loadmodel(self, path):
        load_neural_model(self, path)

    def predict(self, df_VA):
        seven_vals = self._model.predict(df_VA.values)
        df_seven = pd.DataFrame(seven_vals, columns=seven_fields)
        df_seven = change_df_accuracy(df_seven)
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
        dir_path = self.unzip_model(path)
        model_attrs = self._get_model_attrs()
        if len(os.listdir(dir_path)) != len(model_attrs):
            raise Exception(f'Число моделей != {len(model_attrs)}.')
        for model_attr in model_attrs:
            model_filename = model_attr[1:] + '.pkl'
            full_path = os.path.join(dir_path, model_filename)
            if not os.path.exists(full_path):
                shutil.rmtree(dir_path)
                raise Exception(f'Модель {model_filename} отсутствует в файле {self.filename}.')
            with open(full_path, 'rb') as file:
                try:
                    pickle_model = pickle.load(file)
                    setattr(self, model_attr, pickle_model)
                except pickle.UnpicklingError:
                    shutil.rmtree(dir_path)
                    raise Exception(f'Не удаётся создать модель {model_filename}.')
        shutil.rmtree(dir_path)

    def predict(self, df_VA):
        model_attrs = self._get_model_attrs()
        seven_dict = {}
        for model_attr in model_attrs:
            field = model_attr[len(self._MODEL_ATTR_PREFIX):].capitalize()
            seven_dict[field] = getattr(self, model_attr).predict(df_VA.values)
        seven_vals = [[seven_dict[field][0][i] for i, field in enumerate(seven_fields)]] # N без [0][i] для correct
        #seven_dict = {field: seven_dict[field][0][i] for i, field in enumerate(seven_fields)} # по сути, не нужно уже
        df_seven = pd.DataFrame(seven_vals, columns=seven_fields)
        df_seven = change_df_accuracy(df_seven)
        return df_seven


class ModelClearVANeural(AbstractModel):
    @classmethod
    @property
    def type_(cls):
        return '7->2 (Neural)'

    def loadmodel(self, path):
        load_neural_model(self, path)

    def predict(self, df_seven):
        VA_vals = self._model.predict(df_seven[df_seven.columns[:2]].values) # N без [df_seven.columns[:2]] для correct
        df_VA = pd.DataFrame([VA_vals[0][:2]], columns=pa_fields) # N Просто VA_vals для correct
        df_VA = change_df_accuracy(df_VA)
        return df_VA


class ModelClearVAStat(AbstractModel):
    @classmethod
    @property
    def type_(cls):
        return '7->2 (Stat)'


class ModelClearFACSStat(AbstractModel):
    @classmethod
    @property
    def type_(cls):
        return '7->42 (Stat)'


class ModelFACSClearStat(AbstractModel):
    @classmethod
    @property
    def type_(cls):
        return '42->7 (Stat)'


class ModelVAFACSStat(AbstractModel):
    @classmethod
    @property
    def type_(cls):
        return '2->42 (Stat)'


class ModelFACSVAStat(AbstractModel):
    @classmethod
    @property
    def type_(cls):
        return '42->2 (Stat)'


type_model_interface_dict = {
    getattr(sys.modules[__name__], model).type_:
        model for model in abstract_model_inheritors_list
}
