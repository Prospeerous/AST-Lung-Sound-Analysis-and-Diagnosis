import torch
import torch.nn as nn
import timm  # For the Vision Transformer backbone, if not installed run: pip install timm

class ASTModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
        # Adapt the patch embedding for single-channel input
        orig = self.vit.patch_embed.proj
        self.vit.patch_embed.proj = nn.Conv2d(1, orig.out_channels, kernel_size=orig.kernel_size,
                                              stride=orig.stride, padding=orig.padding)
        with torch.no_grad():
            self.vit.patch_embed.proj.weight = nn.Parameter(orig.weight.mean(dim=1, keepdim=True))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((224, 224))

    def forward(self, x):
        x = self.adaptive_pool(x)
        return self.vit(x)
