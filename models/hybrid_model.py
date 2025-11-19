"""
Hybrid Model combining Unsupervised VAE Encoder + Dual-Branch CNN
=================================================================

Architecture:
1. Unsupervised VAE Encoder: Pre-trained feature extractor
2. Frequency Branch: 3x1 kernels for vertical processing
3. Temporal Branch: 1x7 kernels for horizontal processing
4. Dual Classifiers: Sound type (4 classes) + Disease (5 classes)

Model Details:
- Input: Single-channel mel-spectrograms (1, 128, 157)
- Unsupervised features: VAE-learned representations
- Sound Classification: 4 classes (Crackles, Wheezes, Normal, Other)
- Disease Classification: 5 classes (Healthy, COPD, Asthma, Pneumonia, Bronchitis)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    """Variational Autoencoder Encoder for unsupervised feature learning"""

    def __init__(self, latent_dim=256):
        super(VAEEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # Input: (1, 128, 157)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 64, 78)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (128, 32, 39)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> (256, 16, 19)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # -> (512, 8, 9)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Adaptive pooling to get fixed size (4, 4)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Latent space parameters (512 x 4 x 4 = 8192)
        self.fc_mu = nn.Linear(8192, latent_dim)
        self.fc_logvar = nn.Linear(8192, latent_dim)

    def forward(self, x):
        """
        Forward pass through VAE encoder

        Args:
            x: Input tensor (batch, 1, 128, 157)

        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        h = self.encoder(x)
        h = self.pool(h)  # Pool to (4, 4)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class BasicBlock(nn.Module):
    """Basic residual block for dual branches"""

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                              stride=1, padding=padding, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.skip:
            out += identity

        out = self.relu(out)
        return out


class HybridModel(nn.Module):
    """
    Hybrid Model combining VAE encoder with Dual-Branch CNN

    This model uses:
    1. Pre-trained VAE encoder for unsupervised feature extraction
    2. Dual-branch CNN (frequency + temporal) for supervised learning
    3. Supervised conv layer for additional processing
    4. Feature fusion combining VAE and supervised features
    5. Two classifiers for multi-task learning (sound + disease)
    """

    def __init__(self, num_sound_classes=4, num_disease_classes=5, latent_dim=256):
        super(HybridModel, self).__init__()

        # Unsupervised VAE Encoder
        self.unsupervised_encoder = VAEEncoder(latent_dim=latent_dim)

        # Frequency Branch (3x1 kernels - vertical processing)
        self.freq_branch = nn.Sequential(
            BasicBlock(1, 32, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Identity(),
            BasicBlock(32, 64, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Identity(),
            BasicBlock(64, 128, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        )

        # Temporal Branch (1x7 kernels - horizontal processing)
        self.temporal_branch = nn.Sequential(
            BasicBlock(1, 32, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.Identity(),
            BasicBlock(32, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.Identity(),
            BasicBlock(64, 128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
        )

        # Supervised conv layer (processes concatenated dual-branch features)
        # Input: 256 channels (128 freq + 128 temporal)
        # Output: 256 channels
        self.supervised_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
        )

        # Adaptive pooling to reduce spatial dimensions to 4x4
        self.supervised_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Feature Fusion
        # Combines VAE features (256) + supervised features (256 x 4 x 4 = 4096) = 4352
        self.feature_fusion = nn.Sequential(
            nn.Linear(4352, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256, bias=True),
        )

        # Classifiers
        self.sound_classifier = nn.Linear(256, num_sound_classes, bias=True)
        self.disease_classifier = nn.Linear(256, num_disease_classes, bias=True)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input mel-spectrogram (batch, 1, 128, 157)

        Returns:
            sound_logits: Sound type predictions (batch, 4)
            disease_logits: Disease predictions (batch, 5)
        """
        # Extract VAE features
        mu, logvar = self.unsupervised_encoder(x)
        # Use mu as the latent representation (batch, 256)

        # Dual-branch feature extraction
        freq_features = self.freq_branch(x)
        temporal_features = self.temporal_branch(x)

        # Combine dual-branch features
        combined = torch.cat([freq_features, temporal_features], dim=1)  # (batch, 256, H, W)

        # Supervised conv processing
        supervised_features = self.supervised_conv(combined)  # (batch, 256, H, W)

        # Pool to fixed size
        pooled_supervised = self.supervised_pool(supervised_features)  # (batch, 256, 4, 4)

        # Flatten supervised features
        supervised_flat = pooled_supervised.view(pooled_supervised.size(0), -1)  # (batch, 4096)

        # Concatenate VAE and supervised features
        fused_features = torch.cat([mu, supervised_flat], dim=1)  # (batch, 4352)

        # Feature fusion
        final_features = self.feature_fusion(fused_features)  # (batch, 256)

        # Classification
        sound_logits = self.sound_classifier(final_features)
        disease_logits = self.disease_classifier(final_features)

        return sound_logits, disease_logits


def load_hybrid_model(checkpoint_path='outputs/hybrid_model.pth', device='cpu'):
    """
    Load the pre-trained hybrid model

    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on

    Returns:
        model: Loaded HybridModel
        checkpoint: Full checkpoint dictionary
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = HybridModel(num_sound_classes=4, num_disease_classes=5)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model, checkpoint


if __name__ == '__main__':
    # Test model creation
    print("Testing HybridModel...")

    model = HybridModel(num_sound_classes=4, num_disease_classes=5)
    print(f"Model created successfully")

    # Test forward pass
    x = torch.randn(2, 1, 128, 157)
    sound_logits, disease_logits = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Sound logits shape: {sound_logits.shape}")
    print(f"Disease logits shape: {disease_logits.shape}")

    print("\nModel architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
