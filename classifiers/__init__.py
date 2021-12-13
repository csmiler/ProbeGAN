import os
import torch

from .robust_classifier import get_robust_classifier
from .densenet import densenet
from robustness.tools.helpers import InputNormalize

def get_classifier(arch, num_classes, path, parallel=True):
    assert os.path.isfile(path), 'Error: no checkpoint directory found!'

    if "densenet" in arch:
        classifier = densenet(num_classes=num_classes,
                              depth=190,
                              growthRate=40,
                              compressionRate=2,
                              dropRate=0,
                              )
        if parallel:
            classifier = torch.nn.DataParallel(classifier).cuda()
        else:
            classifier = classifier.cuda()
        checkpoint = torch.load(path)
        classifier.load_state_dict(checkpoint['state_dict'])
        
        normalizer = InputNormalize(torch.tensor([0.4914, 0.4822, 0.4465]), 
                                    torch.tensor([0.2023, 0.1994, 0.2010])
                                    ).cuda()
        classifier.module.set_normalizer(normalizer)

        classifier.eval()
    elif "waterbird" in arch:
        classifier = torch.load(path)
        if parallel:
            classifier = torch.nn.DataParallel(classifier).cuda()
        else:
            classifier = classifier.cuda()
        classifier.eval()
    elif "resnet50" in arch:
        if 'cifar' in path:
            classifier = get_robust_classifier('cifar10', path, parallel=parallel)
        elif 'Restricted' in path:
            classifier = get_robust_classifier('RestrictedImageNet', path, parallel=parallel)
        elif 'ImageNet' in path:
            classifier = get_robust_classifier('ImageNet', path, parallel=parallel)
        else:
            raise Exception('classifier not supported!')
    else:
        raise Exception("classifier not supported!")

    return classifier
