from enum import Enum


class Finger(Enum):
    THUMB = 0,
    INDEX = 1,
    MIDDLE = 2,
    RING = 3,
    PINKY = 4


class FingerJoint(Enum):
    PROXIMAL = 0,
    MEDIAL = 1,
    DISTAL = 2
