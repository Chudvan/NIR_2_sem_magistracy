from abc import ABC, abstractmethod
from GUI.tools import seven_fields, type_model_dict


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
        from tensorflow.keras.models import load_model
        self._model = load_model(path)

    def predict(self, df_VA):
        neural_vals = self._model.predict(df_VA.values)
        df_Neural = pd.DataFrame(neural_vals, columns=seven_fields)
        return df_Neural

