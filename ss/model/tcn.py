import torch
import torch.nn as nn


class GlobalLayerNorm(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_channels, 1))
        self.beta = nn.Parameter(torch.zeros(n_channels, 1))

    def forward(self, x):
        # x: N x C x T
        mean = torch.mean(x, dim=(1, 2), keepdim=True)
        var = torch.var(x, dim=(1, 2), keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + 1e-6) + self.beta


class TCNBlock(nn.Module):
    def __init__(self, n_channel, kernel_size, dilation, speaker_dim=None, **kwargs):
        super().__init__()
        self.speaker_dim = speaker_dim

        if speaker_dim is None:
            self.conv1 = nn.Conv1d(n_channel, n_channel, kernel_size=1)
        else:
            self.conv1 = nn.Conv1d(n_channel + speaker_dim, n_channel, kernel_size=1)
        self.prelu1 = nn.PReLU()
        self.gln1 = GlobalLayerNorm(n_channel)

        self.depth_wise_conv = nn.Conv1d(n_channel, n_channel, kernel_size=kernel_size, groups=n_channel,
                                         padding=(dilation * (kernel_size - 1)) // 2, dilation=dilation)
        self.prelu2 = nn.PReLU()
        self.gln2 = GlobalLayerNorm(n_channel)

        self.conv3 = nn.Conv1d(n_channel, n_channel, kernel_size=1)

    def forward(self, x, speaker_emb=None):
        out = x
        if self.speaker_dim is not None:
            speaker_emb = speaker_emb.unsqueeze(-1).repeat(1, 1, out.size(-1))
            out = torch.cat((out, speaker_emb), dim=1)

        out = self.conv1(out)
        out = self.prelu1(out)
        out = self.gln1(out)

        out = self.depth_wise_conv(out)
        out = self.prelu2(out)
        out = self.gln2(out)

        out = self.conv3(out)
        return out + x


class StackedTCN(nn.Module):
    def __init__(self, n_blocks, n_channel, kernel_size, speaker_dim, **kwargs):
        super().__init__()

        self.tcn_blocks = nn.ModuleList(
            [TCNBlock(n_channel, kernel_size, dilation=2 ** b, speaker_dim=speaker_dim if b == 0 else None)
             for b in range(n_blocks)]
        )

    def forward(self, x, speaker_emb):
        out = x
        for i, module in enumerate(self.tcn_blocks):
            if i == 0:
                out = module(out, speaker_emb)
            else:
                out = module(out)
        return out
