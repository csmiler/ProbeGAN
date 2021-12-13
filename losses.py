import torch
import torch.nn.functional as F

# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
    L1 = torch.mean(F.softplus(-dis_real))
    L2 = torch.mean(F.softplus(dis_fake))
    return L1, L2


def loss_dcgan_gen(dis_fake):
    loss = torch.mean(F.softplus(-dis_fake))
    return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
    # loss = torch.mean(F.relu(1. - dis_real))
    # loss += torch.mean(F.relu(1. + dis_fake))
    # return loss


def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


class ClippedCrossEntropyLoss(torch.nn.Module):

    def __init__(self, clip, device='cuda'):
        super(ClippedCrossEntropyLoss, self).__init__()
        self.clip = torch.log(torch.tensor(clip)).to(device=device)

    def forward(self, input, target):
        log_probs = F.log_softmax(input, dim=-1)
        nll = F.nll_loss(log_probs, target, reduction='none')
        mask = torch.gather(log_probs, dim=1, index=target.view(-1,1)).squeeze()
        nll = nll[mask < self.clip]
        if nll.size()[0]:
            return torch.mean(nll)
        else:
            return torch.tensor(0.0)


# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis

if __name__ == "__main__":
    loss = ClippedCrossEntropyLoss(0.9)
    input = torch.tensor([[1.0, 2.0, 30.0], [100.0, 0.1, 0.0]]).cuda()
    target = torch.tensor([2, 0]).cuda()
    l = loss(input, target)
    print(l)
    print(l.item())
