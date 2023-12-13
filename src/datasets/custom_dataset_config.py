import numpy as np
from pathlib import Path


########################################################################
#                                Labels                                #
########################################################################


raw_dir = Path("/path/to/raw/directory")

TILES = {
    "train": [
        str(_)
        for _ in (raw_dir / "train").iterdir()
        if _.suffix == ".las"
    ],
    "val": [
        str(_)
        for _ in (raw_dir / "val").iterdir()
        if _.suffix == ".las"
    ],
    "test": [
        str(_)
        for _ in (raw_dir / "test").iterdir()
        if _.suffix == ".las"
    ],
    "predict": [
        str(_)
        for _ in (raw_dir / "predict").iterdir()
        if _.suffix == ".las"
    ],
}

NUM_CLASSES = 6

ID2TRAINID = {0: 0,
              1: 0,
              2: 1,
              3: 1,
              9: 1,
              4: 2,
              5: 2,
              14: 3,
              15: 4,
              6: 5,
              7: 6,
}


CLASS_NAMES = [
    "Ground",
    "Vegetation",
    "Powerline",
    "Pole",
    "Building",
    "Noise",
]

CLASS_COLORS = np.asarray(
    [
        [243, 214, 171],  # sunset
        [70, 115, 66],  # fern green
        [233, 50, 239],
        [243, 238, 0],
        [190, 153, 153],
        [0, 233, 11],
        [214, 66, 54],  # vermillon
    ]
)
