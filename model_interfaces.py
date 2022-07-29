from abc import ABC, abstractmethod
from GUI.tools import seven_fields, type_model_dict
import tarfile
import os
import shutil


DIR_PATH = '/tmp'


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

    @property
    @abstractmethod
    def type_(self):
        pass

    @abstractmethod
    def loadmodel(self, path):
        pass

    @abstractmethod
    def predict(self, input_):
        pass


class ModelVAClearNeural(AbstractModel):
    def __init__(self, path):
        self.loadmodel(path)

    @property
    def type_(self):
        return '2->7 (Neutral)'

    def loadmodel(self, path):
        with tarfile.open(path, 'r:gz') as tar:
            dir_path = os.path.join(DIR_PATH, 'model_va_clear')
            for file in tar:
                if file.name != 'type':
                    tar.extract(file.name, dir_path)
        print(os.listdir(dir_path))
        if len(os.listdir(dir_path)) != 1:
            raise Exception('Число папок с моделями != 1.')
        full_path = os.path.join(dir_path, os.listdir(dir_path)[0])
        from tensorflow.keras.models import load_model
        try:
            print('full_path', full_path)
            self._model = load_model(full_path)
        except Exception:
            shutil.rmtree(dir_path)
            print('model?')
            raise
        shutil.rmtree(dir_path)

    def predict(self, df_VA):
        neural_vals = self._model.predict(df_VA.values)
        df_Neural = pd.DataFrame(neural_vals, columns=seven_fields)
        return df_Neural


class ModelVAClearStat(AbstractModel):
    pass


class ModelClearVANeural(AbstractModel):
    pass


class ModelClearVAStat(AbstractModel):
    pass


class ModelClearFACSStat(AbstractModel):
    pass


class ModelFACSClearStat(AbstractModel):
    pass


class ModelVAFACSStat(AbstractModel):
    pass


class ModelFACSVAStat(AbstractModel):
    pass
