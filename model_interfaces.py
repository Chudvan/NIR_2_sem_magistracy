from abc import ABC, abstractmethod
from GUI.tools import seven_fields, type_model_dict, change_df_accuracy, \
    type_model_interface_key_to_type_model_key
import tarfile
import os
import sys
import shutil
import pandas as pd
from tensorflow.keras.models import load_model


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

    def __init__(self, filename, path):
        self.filename = filename
        self.loadmodel(path)

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

    @abstractmethod
    def predict(self, input_):
        pass


abstract_model_inheritors_list = ['ModelVAClearNeural', 'ModelVAClearStat',
                                  'ModelClearVANeural', 'ModelClearVAStat',
                                  'ModelClearFACSStat', 'ModelFACSClearStat',
                                  'ModelVAFACSStat', 'ModelFACSVAStat']


class ModelVAClearNeural(AbstractModel):
    @classmethod
    @property
    def type_(cls):
        return '2->7 (Neural)'

    def loadmodel(self, path):
        with tarfile.open(path, 'r:gz') as tar:
            type_model = type_model_interface_key_to_type_model_key(self.type_)
            model_attr = type_model_dict[type_model]
            dir_path = os.path.join(DIR_PATH, model_attr)
            for file in tar:
                if file.name != TYPE_FILENAME:
                    tar.extract(file.name, dir_path)
        if len(os.listdir(dir_path)) != 1:
            raise Exception('Число папок с моделями != 1.')
        full_path = os.path.join(dir_path, os.listdir(dir_path)[0])
        try:
            self._model = load_model(full_path)
        except Exception:
            shutil.rmtree(dir_path)
            raise
        shutil.rmtree(dir_path)

    def predict(self, df_VA):
        neural_vals = self._model.predict(df_VA.values)
        df_Neural = pd.DataFrame(neural_vals, columns=seven_fields)
        df_Neural = change_df_accuracy(df_Neural)
        return df_Neural


class ModelVAClearStat(AbstractModel):
    @classmethod
    @property
    def type_(cls):
        return '2->7 (Stat)'


class ModelClearVANeural(AbstractModel):
    @classmethod
    @property
    def type_(cls):
        return '7->2 (Neural)'


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
