import pytest
from matplotlib.pyplot import plot
from splitfxm.visualize import draw, Domain


def test_draw():
    """
    Test the draw function with a real Domain instance.
    """
    # Create a Domain using from_size
    domain = Domain.from_size(nx=3, nb_left=1, nb_right=1, components=["u"])

    # Call the draw function
    draw(domain, "test_label")
