from dataclasses import dataclass


@dataclass
class Frame(object):
    """
    C-like structure to hold our frame with all attributes for Video class
    """
    image: None
    heads: list
    one_face_detected: bool = False


@dataclass
class Head(object):
    """
    C-like structure to hold our detected heads
    """
    x: float
    y: float
    w: float
    h: float
    direction: str
    origin_xy: list
    destination_xy: list
    direction_image: None
