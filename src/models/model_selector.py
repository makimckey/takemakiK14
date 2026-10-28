from src.models.base_model import BaseDLModel
from src.models.sleep_cam import SleepCAM
from src.models.customizable_sleep_cam import CustomizableSleepCAM

MODEL_DICT = {
    "sleep_cam": SleepCAM,
    "custom_sleep_cam": CustomizableSleepCAM,
}


class ModelSelector:
    @staticmethod
    def select(name: str) -> BaseDLModel:
        """
        Selects a model class by name.


        This method retrieves a model class from `MODEL_DICT` using the provided name.
        Args:
            name (str): The key corresponding to the desired model.
        Returns:
            nn.Module: The model class associated with the given name.
        Raises:
            ValueError: If the model name is not found in `MODEL_DICT`.
        Example:
            >>> ModelSelector.select("sleep_cam")
            <class 'models.sleep_cam.SleepCAM'>
        """

        model_class = MODEL_DICT.get(name)
        if model_class is None:
            raise ValueError(f"Model '{name}' is not defined in `MODEL_DICT`.")
        return model_class
