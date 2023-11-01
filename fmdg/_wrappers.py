'''
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research
and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
or recommendations expressed in this material are those of the author(s) and do not necessarily
reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2023 Massachusetts Institute of Technology.
'''

# CLIP wrappers

import torch.nn as nn

__all__ = [
    "CLIP_img",
    "CLIP",
    "OpenCLIP",
]


class CLIP_img(nn.Module):
    def __init__(self, clip_model, num_ftrs, n_classes, device):
        super(CLIP_img, self).__init__()
        self.clip_model = clip_model
        self.fc = nn.Linear(num_ftrs, n_classes).to(device)

    def forward(self, x):
        x1 = self.clip_model.encode_image(x)
        x1 = x1.float()
        out = self.fc(x1)
        return out


class CLIP(nn.Module):
    def __init__(self, clip_model):
        super(CLIP, self).__init__()
        self.clip_model = clip_model

    def forward(self, x, text):

        image_features = self.clip_model.encode_image(x)
        text_features = self.clip_model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = 100.0 * image_features @ text_features.T
        probs = similarity.softmax(dim=-1)

        return similarity, probs


class OpenCLIP(nn.Module):
    def __init__(self, clip_model):
        super(OpenCLIP, self).__init__()
        self.clip_model = clip_model

    def forward(self, x, text):
        img_f, txt_f, scale = self.clip_model(x, text)
        logits = scale * img_f @ txt_f.T
        probs = logits.softmax(dim=-1)
        return logits, probs
