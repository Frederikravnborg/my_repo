from torch.utils.data import Dataset
from torchvision.datasets import MNIST

from my_project.data import MyDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("data/raw")
    assert isinstance(dataset, Dataset)

def test_data():
    dataset = MNIST(root="data", download=True)
    image, label = dataset[0]
    w, h = image.size
    assert w == 28 and h == 28
    assert len(dataset) > 0
