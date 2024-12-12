import pyrootutils

root = str(
    pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", "README.md"],
        pythonpath=True,
        dotenv=True,
    )
)
import torch

from src.data.data import Data


def test_data():
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32, device="cuda")
    data = Data(x=x)
    try:
        print(data.keys())
    except Exception as e:
        print(e)
        print(
            " Beware of your pyg library, sometime data keys has to be data.keys() or data.keys according to your pyg version"
        )
        # * check for d1.keys , self.keys and data.keys


test_data()
