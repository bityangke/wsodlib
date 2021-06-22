import abc

from torch import nn


__all__ = ['DatasetAnnotationParser']


class DatasetAnnotationParser(nn.Module, metaclass=abc.ABCMeta):
    """
    Torchvision transforms that transform the annotations into the target format
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return callable(subclass)