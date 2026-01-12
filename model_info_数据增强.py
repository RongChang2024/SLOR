import torch
import random
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.drop = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)                 # dropout
        x = self.fc2(x)
        return x


class Classifier(nn.Module):
    """
    """
    def __init__(self, in_channels, num_classes=2, dropout_p=0.5):
        super(Classifier, self).__init__()
        self.drop = nn.Dropout(p=dropout_p)   # ✅dropout
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.drop(x)                      #  dropout in FC layers
        logits = self.fc(x)                   # raw logits
        return logits


class CombinedModel(nn.Module):
    def __init__(self, num_classes):
        super(CombinedModel, self).__init__()

        # 3D ResNet-18
        self.resnet_3d = r3d_18(pretrained=True)
        self.resnet_3d.fc = nn.Identity()

        # MC3-18
        self.i3d = mc3_18(pretrained=True)
        self.i3d.fc = nn.Identity()

        # R(2+1)D-18
        self.densenet_3d = r2plus1d_18(pretrained=True)
        self.densenet_3d.fc = nn.Identity()

        # 三个 backbone 输出一般都是 512 维
        self.fc1 = Classifier(512, num_classes=num_classes, dropout_p=0.5)
        self.fc2 = Classifier(512, num_classes=num_classes, dropout_p=0.5)
        self.fc3 = Classifier(512, num_classes=num_classes, dropout_p=0.5)
        self.fc4 = Classifier(512, num_classes=num_classes, dropout_p=0.5)

        # 512 + 512 + 512 = 1536
        self.conv1x1 = nn.Conv2d(512 + 512 + 512, 512, kernel_size=1)
        self.conv1x1_text = nn.Conv2d(512 + 768, 512, kernel_size=1)

    def forward(self, x, text_features=None):
        resnet_output = self.resnet_3d(x)          # [N, 512]
        i3d_output = self.i3d(x)                   # [N, 512]
        densenet_output = self.densenet_3d(x)      # [N, 512]

        aap_out_1 = resnet_output                  # 你原来就是这么用的
        aap_out_2 = aap(i3d_output)
        aap_out_3 = aap(densenet_output)

        logits_1 = self.fc1(aap_out_1)
        logits_2 = self.fc2(aap_out_2)
        logits_3 = self.fc3(aap_out_3)

        combined_output = torch.cat((resnet_output, i3d_output, densenet_output), dim=1)  # [N, 1536]
        feature_fused = self.conv1x1(combined_output.view(combined_output.size(0), -1, 1, 1))
        aap_out_4 = aap(feature_fused)  # [N, 512]

        if text_features is not None:
            combined_text = torch.cat((aap_out_4, text_features), dim=1)  # [N, 1280]
            feature_fused_info = self.conv1x1_text(
                combined_text.view(combined_text.size(0), -1, 1, 1)
            ).view(aap_out_4.size(0), -1)  # [N, 512]
        else:
            feature_fused_info = aap_out_4

        logits_4 = self.fc4(feature_fused_info)

        return logits_1, logits_2, logits_3, logits_4


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.densenet121 = models.densenet121(pretrained=True)
        self.efficientnetb2 = models.efficientnet_b2(pretrained=True)

        self.resnet18.fc = nn.Identity()
        self.densenet121.classifier = nn.Identity()
        self.efficientnetb2.classifier = nn.Identity()

    def forward(self, x):
        f11 = self.resnet18(x)
        f21 = self.densenet121(x)
        f34 = self.efficientnetb2(x)
        return f11, f21, f34


class FeatureFuser(nn.Module):
    def __init__(self, in_channels, out_channels, output_size):
        super(FeatureFuser, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.aap = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, f11, f12, f13, f14):
        f = torch.max(f11, f12)
        f = torch.max(f, f13)
        f = torch.max(f, f14)
        return f


def aap(fused):
    fused = fused.view(fused.size(0), -1, 1, 1)
    pool = nn.AdaptiveAvgPool2d((1, 1))
    out = pool(fused)
    out = out.view(out.size(0), -1)
    return out


def feature_max(feature_list):
    f = torch.max(feature_list[0], feature_list[1])
    f = torch.max(f, feature_list[2])
    f = torch.max(f, feature_list[3])
    return f


def random_feature(feature_list):
    return random.choice(feature_list)
