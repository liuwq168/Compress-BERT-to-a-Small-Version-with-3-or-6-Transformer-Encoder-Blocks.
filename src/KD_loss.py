import torch
import math
from torch import nn
import torch.nn.functional as F

def cross_entropy(y, labels):
    ce_loss = F.cross_entropy(y, labels, reduction='mean')
    return ce_loss
# def kl_loss(stu, tea):
#     kl_loss = nn.KLDivLoss(reduction='mean')(F.log_softmax(stu / T, dim=1),
#                                                       F.softmax(tea / T, dim=1)) * T * T
#     return kl_loss

def obo_distill_loss(y, teacher_scores, T,alpha):
    if teacher_scores is not None:
        kl_loss = nn.KLDivLoss(reduction='mean')(F.log_softmax(y/T, dim=1),
                                                F.softmax(teacher_scores/T,dim=1))*T*T
    else:
        assert alpha==0,'alpha cannot be {} when teacher scores are not provided'.format(alpha)
        kl_loss = 0.0
    return kl_loss

def sequence_loss(tea_sequence_feat, stu_sequence_output, normalized_sequence=False):
    if normalized_sequence:
        tea_sequence_feat = F.normalize(tea_sequence_feat, p=2, dim=2)
        stu_sequence_output = F.normalize(stu_sequence_output, p=2, dim=2)
    return F.mse_loss(tea_sequence_feat.float(), stu_sequence_output.float()).half()

# def total_loss(y, labels, teacher_scores, T, alpha):
#     ce_loss = cross_entropy(y, labels)
#     d_loss = obo_distill_loss(y, labels, teacher_scores, T,alpha)
#     total_loss = alpha * d_loss + (1.0 - alpha)*ce_loss
#     return total_loss


# loss_dl, kd_loss, ce_loss = distillation_loss(logits_pred_student, label_ids, teacher_pred, T=args.T, alpha=args.alpha)
def distillation_loss(y, labels, teacher_scores, T, alpha, reduction_kd='mean', reduction_nll='mean'):
    if teacher_scores is not None:#kd
        d_loss = nn.KLDivLoss(reduction=reduction_kd)(F.log_softmax(y / T, dim=1),
                                                      F.softmax(teacher_scores / T, dim=1)) * T * T
    else:# finetune teacher
        assert alpha == 0, 'alpha cannot be {} when teacher scores are not provided'.format(alpha)
        d_loss = 0.0
    nll_loss = F.cross_entropy(y, labels, reduction=reduction_nll)
    tol_loss = alpha * d_loss + (1.0 - alpha) * nll_loss
    return tol_loss, d_loss, nll_loss #loss_dl, kd_loss, ce_loss对应于蒸馏的损失+交叉熵损失，整流损失，交叉熵损失（）

def patience_loss(teacher_patience, student_patience, normalized_patience=False):
    if normalized_patience:
        teacher_patience = F.normalize(teacher_patience, p=2, dim=2)
        student_patience = F.normalize(student_patience, p=2, dim=2)
    return F.mse_loss(teacher_patience.float(), student_patience.float()).half()
