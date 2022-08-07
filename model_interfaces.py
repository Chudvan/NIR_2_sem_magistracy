import traceback
from abc import ABC, abstractmethod
from GUI.tools import seven_fields, type_model_dict, change_df_accuracy, \
    type_model_interface_key_to_type_model_key, pa_fields, get_model_type, \
    happy_fields, sad_fields, surprised_fields, scared_fields, angry_fields, \
    disgusted_fields, contempt_fields, other_facs_fields, all_unique_fields
import tarfile
import os
import sys
import shutil
import pandas as pd
from tensorflow.keras.models import load_model
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


def load_models(self, path, neural=True, model_attrs=None):
    dir_path = self.unzip_model(path)
    if model_attrs is None:
        model_attrs = self._get_model_attrs()
    if len(os.listdir(dir_path)) < len(model_attrs):
        raise Exception(f'Число моделей < {len(model_attrs)}.')
    for model_attr in model_attrs:
        model_filename = model_attr[1:]
        if not neural:
            model_filename += '.pkl'
        full_path = os.path.join(dir_path, model_filename)
        print(full_path)
        if not os.path.exists(full_path):
            shutil.rmtree(dir_path)
            raise Exception(f'Модель {model_filename} отсутствует в файле {self.filename}.')
        try:
            if neural:
                model_val = load_model(full_path)
            else:
                with open(full_path, 'rb') as file:
                    model_val = pickle.load(file)
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
            model_type_file = get_model_type(model_filename, full_path)
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
        load_models(self, path, neural=False)

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
        load_models(self, path)

    def predict(self, df_seven):
        VA_vals = self._model.predict(df_seven[df_seven.columns[:2]].values) # N без [df_seven.columns[:2]] для correct
        df_VA = pd.DataFrame([VA_vals[0][:2]], columns=pa_fields) # N Просто VA_vals для correct
        df_VA = change_df_accuracy(df_VA)
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
        load_models(self, path, neural=False)

    def predict(self, df_seven):
        v = self._model_valence.predict(df_seven[df_seven.columns[:2]].values)[0][0] # N без [df_seven.columns[:2]] и мб без [0] для correct
        a = self._model_arousal.predict(df_seven[df_seven.columns[2:4]].values)[0][0] # аналогично
        df_VA = pd.DataFrame([[v, a]], columns=pa_fields)
        df_VA = change_df_accuracy(df_VA)
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
        pass


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
        load_models(self, path, neural=False)

    def predict(self, df_42):
        happy_vals = df_42[happy_fields].values
        sad_vals = df_42[sad_fields].values
        surprised_vals = df_42[surprised_fields].values
        scared_vals = df_42[scared_fields].values
        angry_vals = df_42[angry_fields].values
        disgusted_vals = df_42[disgusted_fields].values
        all_unique_vals = df_42[all_unique_fields].values

        neutral_res = self._model_neutral.predict([list(all_unique_vals[0][1:3])])[0][0] # N просто all_unique_vals для correct
        happy_res = self._model_happy.predict([list(happy_vals[0][1:3])])[0][0] # аналогично
        sad_res = self._model_sad.predict([list(sad_vals[0][1:3])])[0][0] # аналогично
        angry_res = self._model_angry.predict([list(angry_vals[0][1:3])])[0][0] # аналогично
        surprised_res = self._model_surprised.predict([list(surprised_vals[0][1:3])])[0][0] # аналогично
        scared_res = self._model_scared.predict([list(scared_vals[0][1:3])])[0][0] # аналогично
        disgusted_res = self._model_disgusted.predict([list(disgusted_vals[0][1:3])])[0][0] # аналогично

        df_seven = pd.DataFrame([[neutral_res, happy_res, sad_res,
                                  angry_res, surprised_res, scared_res,
                                  disgusted_res]], columns=seven_fields)
        df_seven = change_df_accuracy(df_seven)
        return df_seven


class ModelVAFACS(AbstractModel):
    _model_va_clear = None
    _model_clear_facs = None
    _MODEL_ATTR_PREFIX = '_model_'

    @classmethod
    @property
    def type_(cls):
        return '2->42 (Stat)'

    def loadmodel(self, path):
        load_VA_and_FACS_model(self, path)

    def predict(self, df_VA):
        pass


class ModelFACSVA(AbstractModel):
    _model_facs_clear = None
    _model_clear_va = None
    _MODEL_ATTR_PREFIX = '_model_'

    @classmethod
    @property
    def type_(cls):
        return '42->2 (Stat)'

    def loadmodel(self, path):
        load_VA_and_FACS_model(self, path)

    def predict(self, df_42):
        pass


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
