import matplotlib.pyplot as plt
import numpy as np


def plot_data_with_pe(data, pe):
    assert len(data.shape) == 3, "data.shape {}".format(data.shape)
    assert data.shape == pe.shape
    num_examples = data.shape[0]
    num_tokens = data.shape[1]
    num_features_or_channels = data.shape[2]
    x = np.arange(num_tokens)

    fig, ax = plt.subplots(
        num_features_or_channels, 2 * num_examples, figsize=(10, 10)
    )
    if num_features_or_channels == 1:
        ax = ax[None, :]

    for i in range(num_examples):
        data_col = i * 2
        pe_col = data_col + 1
        ax[0][pe_col].set_title("PE")
        ax[0][data_col].set_title("Data")
        for j in range(num_features_or_channels):
            ax[j][data_col].plot(x, data[i, :, j])

            ax[j][pe_col].plot(x, pe[i, :, j])

    plt.tight_layout()
    plt.savefig('results/pe_plot.png')
    plt.show()


def plot_grid_data_with_pe(data, pe):
    assert len(data.shape) == 3, "data.shape {}".format(data.shape)
    assert data.shape == pe.shape
    num_examples = data.shape[0]

    fig, ax = plt.subplots(2, num_examples, figsize=(10, 10))
    if num_examples == 1:
        ax = ax[:, None]

    for i in range(num_examples):
        example_pe = pe[i, :, :].T
        example_data = data[i, :, :].T

        ax[0][i].imshow(example_data, aspect="auto")
        ax[1][i].imshow(example_pe, aspect="auto")
        ax[0][i].set_title("Data")
        ax[1][i].set_title("PE")

    plt.tight_layout()
    plt.savefig('results/pe_grid.png')
    plt.show()


# if __name__ == "__main__":
#     from models.self_attention import PositionalEncoding
#     from datasets.smnist import dataFactory, Config

#     config = Config(batch_size=2)
#     data = dataFactory(config)
#     trainloader, valloader = data.train_val()

#     inputs, labels = next(iter(trainloader))
#     print("inputs.shape", inputs.shape)

# pe_layer = PositionalEncoding(1, max_len=1000)
# pe_layer = ImplicitPositionalEncoding(1, activation='relu')

# pe_layer.eval()
# _, pe = pe_layer(inputs, with_pe=True)
# pe = pe.detach().numpy()
# print("pe.shape", pe.shape)

# plot_data_with_pe(inputs, pe)
