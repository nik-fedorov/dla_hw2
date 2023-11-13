import torch
from torch import nn
import torch.nn.functional as F

from .tcn import StackedTCN


class ResNetBlock(nn.Module):
    def __init__(self, n_channel):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channel, n_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.prelu1 = nn.PReLU()

        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(n_channel)

        self.prelu2 = nn.PReLU()
        self.max_pool = nn.MaxPool1d(3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + x
        out = self.prelu2(out)
        out = self.max_pool(out)
        return out


class SpeechEncoder(nn.Module):
    def __init__(self, encoder_dim, L1, L2, L3):
        super().__init__()

        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.stride = L1 // 2

        self.short_encoder = nn.Conv1d(1, encoder_dim, kernel_size=L1, stride=self.stride)
        self.middle_encoder = nn.Conv1d(1, encoder_dim, kernel_size=L2, stride=self.stride)
        self.long_encoder = nn.Conv1d(1, encoder_dim, kernel_size=L3, stride=self.stride)

    def forward(self, x):
        w1 = F.relu(self.short_encoder(x.unsqueeze(1)))

        T = w1.shape[-1]
        pad_middle = (T - 1) * self.stride + self.L2 - x.shape[-1]
        pad_long = (T - 1) * self.stride + self.L3 - x.shape[-1]

        w2 = F.relu(self.middle_encoder(F.pad(x, [0, pad_middle], "constant", 0).unsqueeze(1)))
        w3 = F.relu(self.long_encoder(F.pad(x, [0, pad_long], "constant", 0).unsqueeze(1)))

        return torch.cat([w1, w2, w3], dim=1), w1, w2, w3


class SpeakerEncoder(nn.Module):
    def __init__(self, encoder_dim, resnet_block_dim, out_dim, n_resnet_blocks):
        super().__init__()

        self.ln = nn.LayerNorm(3 * encoder_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(3 * encoder_dim, resnet_block_dim, kernel_size=1),
            *[ResNetBlock(resnet_block_dim) for _ in range(n_resnet_blocks)],
            nn.Conv1d(resnet_block_dim, out_dim, kernel_size=1)
        )

    def forward(self, x):
        out = self.ln(x.transpose(1, 2)).transpose(1, 2)
        out = self.encoder(out)
        out = torch.mean(out, dim=-1)
        return out


class SpeechExtractor(nn.Module):
    def __init__(self, encoder_dim, n_stacked_tcn, n_tcn_blocks, kernel_size, tcn_dim, speaker_dim):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.ln = nn.LayerNorm(3 * encoder_dim)
        self.conv = nn.Conv1d(3 * encoder_dim, tcn_dim, kernel_size=1)

        self.tcn = nn.ModuleList(
            [StackedTCN(n_tcn_blocks, tcn_dim, kernel_size, speaker_dim) for _ in range(n_stacked_tcn)]
        )

        self.mask_convs = nn.ModuleList([nn.Conv1d(tcn_dim, encoder_dim, kernel_size=1) for _ in range(3)])

    def forward(self, x, speaker_emb):
        out = self.ln(x.transpose(1, 2)).transpose(1, 2)
        out = self.conv(out)
        for module in self.tcn:
            out = module(out, speaker_emb)
        return (F.relu(module(out)) for module in self.mask_convs)


class SpeechDecoder(nn.Module):
    def __init__(self, encoder_dim, L1, L2, L3):
        super().__init__()

        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.stride = L1 // 2

        self.short_decoder = nn.ConvTranspose1d(encoder_dim, 1, kernel_size=L1, stride=self.stride)
        self.middle_decoder = nn.ConvTranspose1d(encoder_dim, 1, kernel_size=L2, stride=self.stride)
        self.long_decoder = nn.ConvTranspose1d(encoder_dim, 1, kernel_size=L3, stride=self.stride)

    def forward(self, x1, x2, x3, len_mix):
        s1 = self.short_decoder(x1).squeeze(1)
        s1 = F.pad(s1, [0, len_mix - s1.shape[-1]], "constant", 0)
        s2 = self.middle_decoder(x2).squeeze(1)[:, :len_mix]
        s3 = self.long_decoder(x3).squeeze(1)[:, :len_mix]
        return s1, s2, s3


class SpExPlus(nn.Module):
    def __init__(self, tcn_params,
                 resnet_block_dim,
                 n_resnet_blocks,
                 speaker_dim,
                 num_speakers,
                 encoder_dim,
                 L1, L2, L3):
        super().__init__()

        self.speech_encoder = SpeechEncoder(encoder_dim, L1, L2, L3)
        self.speaker_encoder = SpeakerEncoder(encoder_dim, resnet_block_dim, speaker_dim, n_resnet_blocks)
        self.speech_extractor = SpeechExtractor(encoder_dim, speaker_dim=speaker_dim, **tcn_params)
        self.speech_decoder = SpeechDecoder(encoder_dim, L1, L2, L3)

        self.head = nn.Linear(speaker_dim, num_speakers)

    def forward(self, mix, ref, **kwargs):
        len_mix = mix.shape[-1]

        mix_enc, y1, y2, y3 = self.speech_encoder(mix)
        ref_enc, _, _, _ = self.speech_encoder(ref)
        speaker_emb = self.speaker_encoder(ref_enc)
        M1, M2, M3 = self.speech_extractor(mix_enc, speaker_emb)
        s1, s2, s3 = self.speech_decoder(y1 * M1, y2 * M2, y3 * M3, len_mix)

        return {"short": s1, "middle": s2, "long": s3, "speaker_logits": self.head(speaker_emb)}
