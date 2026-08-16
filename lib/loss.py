import torch.nn.functional as F
import torch.nn as nn
import torch


class My_Loss(nn.Module):
    def __init__(self):
        super(My_Loss, self).__init__()
    def forward(self, pred, mask):
        loss = structure_loss(pred, mask)
        return loss

def structure_loss(pred, mask):
    weight = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weight * wbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    probability = torch.sigmoid(pred)
    intersection = ((probability * mask) * weight).sum(dim=(2, 3))
    union = ((probability + mask) * weight).sum(dim=(2, 3))
    weighted_iou = 1 - (intersection + 1) / (union - intersection + 1)
    return (wbce + weighted_iou).mean()

class Cross_Entropy(nn.Module):
    def __init__(self):
        super(Cross_Entropy, self).__init__()
    def forward(self, pred, labels):
        loss = cross_entropy_per_image(pred[0], labels)
        return loss

# Process every images
def cross_entropy_per_image(logits, labels):
    total_loss = 0
    for i, (_logit, _label) in enumerate(zip(logits, labels)):
        total_loss += cross_entropy_with_weight(_logit, _label)
    return total_loss / len(logits)

def cross_entropy_with_weight(logits, labels):
    # weight = 0.8
    logits = logits.contiguous().view(-1)  # contiguous() 开辟一块新的内存去存储logits
    labels = labels.view(-1)
    eps = 1e-6
    pred_pos = logits[labels > 0].clamp(eps, 1.0 - eps)
    pred_neg = logits[labels == 0].clamp(eps, 1.0 - eps)
    w_anotation = labels[labels > 0]
    cross_entropy = (-pred_pos.log() * w_anotation).mean() + \
                    (-(1.0 - pred_neg).log()).mean()
    return cross_entropy
