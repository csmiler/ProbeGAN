"""Implementation of Biggan-AM
"""

import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision
from torchvision.utils import save_image

# Import my stuff
import inception_utils
import utils
import losses
import train_fns

# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def run(config):

    # Update the config dict as necessary
    # This is for convenience, to add settings derived from the user-specified
    # configuration into the config-dict (e.g. inferring the number of classes
    # and size of the images from the dataset, passing in a pytorch object
    # for the activation specified as a string)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = 9
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    if config['new_class'] == -1:
        config['new_class'] = config['n_classes'] - 1

    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)
    device = 'cuda'

    # Seed RNG
    utils.seed_rng(config['seed'])

    # Prepare root folders if necessary
    utils.prepare_root(config)

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    # Import the model--this line allows us to dynamically select different files.
    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name']
                         else utils.name_from_config(config))
    print('Experiment name is %s' % experiment_name)

    # If using EMA, prepare it
    if config['ema']:
        print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
        G_ema = model.Generator(**{**config, 'skip_init':True, 
                                   'no_optim': True}).to(device)
        G, ema = None, None
    else:
        G = model.Generator(**config).to(device)
        G_ema, ema = None, None

    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}
    # If loading from a pre-trained model, load weights
    if config['resume']:
        print('Loading weights...')
        utils.load_weights(G, None, state_dict,
                           config['weights_root'], experiment_name, 
                           config['load_weights'] if config['load_weights'] else None,
                           G_ema if config['ema'] else None)
    G = G_ema if G is None else G
    #if config['parallel']:
    G.eval()

    # load in pretrained classifier
    if config['C_path']:
        from classifiers import get_classifier
        classifier = get_classifier(config['C_arch'], config['C_n_classes'], config['C_path'], parallel=config['parallel'])
        classifier.eval()
    else:
        raise


    # params for biggan-am
    lr = 0.1
    weight_decay = 0.9
    state_z = 0
    n_iters = 10
    steps_per_z = 20
    z_num = 10
    min_clamp, max_clamp = -2.0, 2.0
    radius = 0.1

    # class info and dirs
    target_class = config['new_class']
    labels = torch.LongTensor([target_class]*z_num).cuda()
    criterion = nn.CrossEntropyLoss()
    tdir = "%s/%s/am"%(config['samples_root'], config['experiment_name'])
    os.makedirs(tdir, exist_ok=True)

    # create initial embedding by taking the mean of existing classes
    print(G.shared.weight.min(), G.shared.weight.max(), G.shared.weight.mean(), G.shared.weight.shape)
    mean_class_embedding = torch.mean(G.shared.weight, dim=0, keepdim=False)
    print(mean_class_embedding.min(), mean_class_embedding.max(), mean_class_embedding.mean(), mean_class_embedding.shape)
    dim_z = mean_class_embedding.size(-1)
    init_embedding = (mean_class_embedding + torch.randn(mean_class_embedding.size()).cuda() * radius).detach()
    optim_embedding = init_embedding.clone()
    optim_embedding.requires_grad_()
    
    optimizer = optim.Adam([optim_embedding], lr=lr, weight_decay=weight_decay)

    #torch.set_rng_state(state_z)

    for epoch in range(n_iters):
        zs = torch.randn((z_num, dim_z), requires_grad=False).cuda()
        imgs = []
        for z_step in range(steps_per_z):
            optimizer.zero_grad()

            clamped_embedding = torch.clamp(optim_embedding, min_clamp, max_clamp)
            #print(clamped_embedding.min(), clamped_embedding.max(), clamped_embedding.mean(), clamped_embedding.shape)
            repeat_clamped_embedding = clamped_embedding.repeat(z_num, 1).cuda()
            G_z = G(zs, repeat_clamped_embedding)
            G_z = G_z * 0.5 + 0.5
            pred_logits = classifier(G_z)
            if isinstance(pred_logits, tuple):
                pred_logits, _ = pred_logits
            loss = criterion(pred_logits, labels)
            pred_probs = nn.functional.softmax(pred_logits, dim=-1)
            loss.backward()
            optimizer.step()

            avg_target_prob = pred_probs[:, target_class].mean().item()
            print(f"Epoch: {epoch:0=5d}\tStep: {z_step:0=5d}\tAverage Target Prob:{avg_target_prob:.4f}")
            print(f"Predicted classes = ", pred_probs.argmax(dim=-1))

            if (z_step + 1) % 10 == 0:
                imgs.append(G_z.detach())
            #global_step = epoch * steps_per_z + z_step
            #save_image(G_z, os.path.join(tdir, f"{global_step:0=7d}.jpg"), normalize=True, nrow=10)

            torch.cuda.empty_cache()
        save_image(torch.cat(imgs), os.path.join(tdir, f"{epoch:0=7d}.jpg"), normalize=True, range=(0,1), nrow=z_num)

    # save final embedding
    np.save(os.path.join(tdir, "final_embedding.npy"), 
            torch.clamp(optim_embedding, min_clamp, max_clamp).detach().cpu().numpy()
            )
    print(optim_embedding.min(), optim_embedding.max(), optim_embedding.mean(), optim_embedding.shape)
    print(optim_embedding)

    # sample and save
    N = 5000
    i, bsize = 0, 100
    imgs = []
    eb = torch.clamp(optim_embedding, min_clamp, max_clamp).repeat(bsize, 1)
    print(eb.shape)
    while i < N:
        print(i)
        step = min(bsize, N-i)
        zs = torch.randn(step, dim_z, requires_grad=False).cuda()
        G_z = G(zs, eb) * 0.5 + 0.5
        imgs.append(G_z.detach().cpu().numpy())
        i += step
    imgs = np.concatenate(imgs)
    np.save(os.path.join(tdir, "samples5k.npy"), imgs)




def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)

if __name__ == '__main__':
    main()
