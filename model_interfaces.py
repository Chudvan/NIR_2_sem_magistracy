from abc import ABC, abstractmethod
from GUI.tools import seven_fields


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

