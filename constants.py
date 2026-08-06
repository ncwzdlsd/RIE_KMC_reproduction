from enum import IntEnum

class SiteType(IntEnum):
    M=0
    O=1

class Species(IntEnum):
    EMPTY=0
    CE=1
    O=2
    IR_ION=3
    IR=4

class Channel(IntEnum):
    CE_EXCHANGE=0
    O_EXCHANGE=1
    IR_RESERVOIR=2
    IR_REDUCTION=3
    IR_DIFFUSION=4

CHANNELS_PER_SITE=5
KB_EV_PER_K=8.617333262e-5