"""Package for DeepCrack model utilities."""

import importlib
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

def find_model_using_name(model_name):
    """Import the module 'models/[model_name]_model.py'."""
    try:
        model_filename = f"models.{model_name}_model"
        modellib = importlib.import_module(model_filename)
        model = None
        target_model_name = model_name.replace('_', '') + 'model'
        for name, cls in modellib.__dict__.items():
            if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
                model = cls
        if model is None:
            raise ValueError(f"No subclass of BaseModel found in {model_filename}")
        return model
    except Exception as e:
        logger.error(f"Error in find_model_using_name: {str(e)}")
        raise

def get_option_setter(model_name):
    """Return the static method modify_commandline_options of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options

def create_model(opt):
    """Create a model given the option."""
    try:
        model = find_model_using_name(opt.model)
        instance = model(opt)
        logger.info(f"Model [{type(instance).__name__}] was created")
        return instance
    except Exception as e:
        logger.error(f"Error in create_model: {str(e)}")
        raise