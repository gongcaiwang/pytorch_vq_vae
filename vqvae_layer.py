import torch

class Conv1d(torch.nn.Module):
    """ 1D Temporal Convolution """

    def __init__(
            self,
            in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
            dropout=0.5
    ):
        super(Conv1d, self).__init__()

        # Adds convolutional layers with spatial dropout.
        self.conv_layer = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode='zeros')
        self.dopout_layer = torch.nn.Dropout(dropout)

    def forward(self, inputs):
        x = self.conv_layer(inputs)
        x = self.dopout_layer(x)
        return x


class ConvTranspose1d(torch.nn.Module):
    """ 1D Transpose Temporal Convolution """

    def __init__(
            self,
            in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros',
            dropout=0.5
    ):
        super(ConvTranspose1d, self).__init__()

        # Adds convolutional layers with spatial dropout.
        self.conv_layer = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, output_padding=output_padding, padding_mode='zeros')
        self.dopout_layer = torch.nn.Dropout(dropout)
    def forward(self, inputs):
        x = self.conv_layer(inputs)
        x = self.dopout_layer(x)
        return x