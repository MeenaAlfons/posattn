import numpy as np
from plotly import graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
from posattn import relative_unfold_flatten


def visualize_relative_1d(relative_positional_encoding, extract_size, name):
    """
    relative_positional_encoding: [<extract_size>, ModelDim]
    """
    assert len(extract_size) == 1
    assert np.allclose(
        np.array(relative_positional_encoding.shape[:-1]), extract_size
    )
    model_dim = relative_positional_encoding.shape[1]
    D1 = 1 + (extract_size[0] // 2)
    x = np.arange(-D1 + 1, D1)
    assert len(x) == extract_size[0]

    return go.Figure(
        data=go.Heatmap(
            x=x,
            y=np.arange(0, model_dim),
            z=relative_positional_encoding.T,
            colorscale="Viridis",
        ),
        layout=go.Layout(title=name)
    )
