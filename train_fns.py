''' train_fns.py
Functions for the main loop of training different conditional image models
'''
import torch
import torch.nn as nn
import torchvision
import os

import utils
import losses
from metrics import compute_IS_and_fid


# Dummy training function for debugging
def dummy_training_function():
    def train(x, y):
        return {}
    return train


def GAN_training_function(G, D, GD, z_, y_, ema, state_dict, config, C=None):
    def train(x, y):
        G.optim.zero_grad()
        D.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, config['batch_size'])
        y = torch.split(y, config['batch_size'])
        counter = 0

        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, True)
            utils.toggle_grad(G, False)

        # Determine type fo clasifier loss
        if config.get('prob_clip', 1.0) < 1.0:
            classifier_loss = losses.ClippedCrossEntropyLoss(config['prob_clip'])
        else:
            if config.get('C_new_coeff', 1.0) != 1.0:
                per_class_weight = torch.ones(config.get('C_n_classes', 10)).cuda()
                per_class_weight[config['new_class']] = config.get('C_new_coeff', 1.0)
                classifier_loss = nn.CrossEntropyLoss(weight=per_class_weight)
            else:
                classifier_loss = nn.CrossEntropyLoss()

        for step_index in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step
            D.optim.zero_grad()
            for accumulation_index in range(config['num_D_accumulations']):
                z_.sample_()
                y_.sample_()
                if D.D_second_type == "wgangp":
                    D_output, grad_penalty = GD(z_[:config['batch_size']], y_[:config['batch_size']], 
                                    x[counter], y[counter], train_G=False, 
                                    split_D=config['split_D'],
                                    grad_penalty=True)
                    D_fake, D_real = D_output
                else:
                    D_fake, D_real = GD(z_[:config['batch_size']], y_[:config['batch_size']], 
                                    x[counter], y[counter], train_G=False, 
                                    split_D=config['split_D'])

                # If there is a second D output, separate the two
                # and mask out the new classes for conditional-D
                if D.D_second and not D.D_only_uncond:
                    D_fake, D_fake2 = torch.chunk(D_fake, 2, dim=-1)
                    D_real, D_real2 = torch.chunk(D_real, 2, dim=-1)
                    D_fake = D_fake[y_[:config['batch_size']] != config['new_class']]
                    # Balance the number of fake vs real for discriminator
                    D_real = D_real[y_[:config['batch_size']] != config['new_class']]
                elif D.D_only_uncond:
                    D_fake2 = D_fake
                    D_real2 = D_real

                # Compute components of D's loss, average them, and divide by 
                # the number of gradient accumulations
                if not D.D_only_uncond:
                    D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
                    D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations']) * float(config['D_coeff'])
                    D_loss.backward(retain_graph=D.D_second)
                else:
                    # for book keeping purpose
                    D_loss_real, D_loss_fake = torch.tensor(0.0), torch.tensor(0.0)
                # Compute D_loss for the second D output
                if D.D_second:
                    if D.D_second_type == "wgangp":
                        # wgan-gp loss
                        if config['D2_reg_coeff'] > 0.0:
                            D_loss_real2 = -D_real2.mean() + config['D2_reg_coeff'] * (D_real2 ** 2).mean()
                            D_loss_fake2 = D_fake2.mean() + config['D2_reg_coeff'] * (D_fake2 ** 2).mean()
                        else:
                            D_loss_real2 = -(D_real2.mean() - 0.001 * (D_real2 ** 2).mean())
                            D_loss_fake2 = D_fake2.mean()
                        D_loss2 = (D_loss_real2 + D_loss_fake2) / float(config['num_D_accumulations']) * float(config['D2_coeff'])
                        D_loss2.backward(retain_graph=True)
                        grad_penalty = grad_penalty.mean()  / float(config['num_D_accumulations']) * float(config['D2_coeff'])
                        grad_penalty.backward()
                    else:
                        D_loss_real2, D_loss_fake2 = losses.discriminator_loss(D_fake2, D_real2)
                        D_loss2 = (D_loss_real2 + D_loss_fake2) / float(config['num_D_accumulations']) * float(config['D2_coeff'])
                        D_loss2.backward()
                counter += 1

            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D')
                utils.ortho(D, config['D_ortho'])

            D.optim.step()

        # Optionally toggle "requires_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(G, True)

        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()

        # If accumulating gradients, loop multiple times
        for accumulation_index in range(config['num_G_accumulations']):
            z_.sample_()
            y_.sample_()
            # If pretrained classifier is provided, fake images should also be returned
            # and compute classification loss
            if C is None:
                D_fake = GD(z_, y_, train_G=True, split_D=config['split_D'])
            else:
                D_fake, G_z = GD(z_, y_, train_G=True, split_D=config['split_D'], return_G_z=True)
                # Convert the range from -1~1 to 0~1
                G_z = G_z * 0.5 + 0.5
                if config.get('C_resize', 0) > 0:
                    G_z = nn.functional.interpolate(G_z, size=config['C_resize'])
                if config.get('C_imagenet_norm', False):
                    imagenet_mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float).view(1, -1, 1, 1).cuda()
                    imagenet_std = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float).view(1, -1, 1, 1).cuda()
                    G_z = G_z.clone()
                    G_z.sub_(imagenet_mean).div_(imagenet_std)
                C_logits = C(G_z)
                if isinstance(C_logits, tuple):
                    C_logits, _ = C_logits
                C_loss = classifier_loss(C_logits, y_) / float(config['num_G_accumulations']) * float(config['C_coeff'])
                if C_loss.item() != 0.0:
                    C_loss.backward(retain_graph=True)

            # If there is a second D output, separate the two
            # and mask out the new classes for conditional-D
            if D.D_second and not D.D_only_uncond:
                D_fake, D_fake2 = torch.chunk(D_fake, 2, dim=-1)
                D_fake = D_fake[y_ != config['new_class']]
            elif D.D_only_uncond:
                D_fake2 = D_fake
            # Compute loss of conditional D
            if not D.D_only_uncond:
                G_loss = losses.generator_loss(D_fake) / float(config['num_G_accumulations']) * float(config['D_coeff']) * float(config['G_coeff'])
                G_loss.backward(retain_graph=D.D_second)
            else:
                # for book keeping purpose
                G_loss = torch.tensor(0.0)
            # Compute loss of unconditional D
            if D.D_second:
                if config['G_new_coeff'] != 1.0:
                    D_fake2 = torch.where(y_ == config['new_class'], D_fake2 * config['G_new_coeff'], D_fake2)
                if D.D_second_type == "wgangp":
                    if config['D2_reg_coeff'] > 0.0:
                        G_loss2 = -D_fake2.mean() + config['D2_reg_coeff'] * (D_fake2 ** 2).mean()
                    else:
                        G_loss2 = -D_fake2.mean()
                    G_loss2 = G_loss2  / float(config['num_G_accumulations']) * float(config['D2_coeff']) * float(config['G_coeff'])
                else:
                    G_loss2 = losses.generator_loss(D_fake2) / float(config['num_G_accumulations']) * float(config['D2_coeff']) * float(config['G_coeff'])
                G_loss2.backward()

        # Optionally apply modified ortho reg in G
        if config['G_ortho'] > 0.0:
            print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(G, config['G_ortho'],
                        blacklist=[param for param in G.shared.parameters()])
        G.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])

        out = {'G_loss': float(G_loss.item()),
                'D_loss_real': float(D_loss_real.item()),
                'D_loss_fake': float(D_loss_fake.item())}
        if D.D_second:
            out.update({
                'G_loss2': float(G_loss2.item()),
                'D_loss_r2': float(D_loss_real2.item()),
                'D_loss_f2': float(D_loss_fake2.item())})
            if D.D_second_type == "wgangp":
                out.update({'D_gp': float(grad_penalty.item())})
        if C is not None:
            out.update({'C_loss': float(C_loss.item())})
        # Return G's loss and the components of D's loss.
        return out
    return train

''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''
def save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, 
                    state_dict, config, experiment_name):
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name, None, G_ema if config['ema'] else None)
    # Save an additional copy to mitigate accidental corruption if process
    # is killed during a save (it's happened to me before -.-)
    if config['num_save_copies'] > 0:
        utils.save_weights(G, D, state_dict, config['weights_root'],
                           experiment_name,
                           'copy%d' %  state_dict['save_num'],
                           G_ema if config['ema'] else None)
        state_dict['save_num'] = (state_dict['save_num'] + 1 ) % config['num_save_copies']

    # Use EMA G for samples or non-EMA?
    which_G = G_ema if config['ema'] and config['use_ema'] else G

    # Accumulate standing statistics?
    if config['accumulate_stats']:
        utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                               z_, y_, config['n_classes'],
                               config['num_standing_accumulations'])

    # Save a random sample sheet with fixed z and y      
    with torch.no_grad():
        if config['parallel']:
            fixed_Gz =  nn.parallel.data_parallel(which_G, (fixed_z, which_G.shared(fixed_y)))
        else:
            fixed_Gz = which_G(fixed_z, which_G.shared(fixed_y))
    if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
        os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
    image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'], 
                                                    experiment_name,
                                                    state_dict['itr'])
    torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                               nrow=int(fixed_Gz.shape[0] **0.5), normalize=True)
    # For now, every time we save, also save sample sheets
    utils.sample_sheet(which_G,
                       classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
                       num_classes=config['n_classes'],
                       samples_per_class=10, parallel=config['parallel'],
                       samples_root=config['samples_root'],
                       experiment_name=experiment_name,
                       folder_number=state_dict['itr'],
                       z_=z_)
#    # Also save interp sheets
#    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
#        utils.interp_sheet(which_G,
#                           num_per_sheet=16,
#                           num_midpoints=8,
#                           num_classes=config['n_classes'],
#                           parallel=config['parallel'],
#                           samples_root=config['samples_root'],
#                           experiment_name=experiment_name,
#                           folder_number=state_dict['itr'],
#                           sheet_number=0,
#                           fix_z=fix_z, fix_y=fix_y, device='cuda')



''' This function runs the inception metrics code, checks if the results
    are an improvement over the previous best (either in IS or FID, 
    user-specified), logs the results, and saves a best_ copy if it's an 
    improvement. '''
def test(G, D, G_ema, z_, y_, state_dict, config, sample, get_inception_metrics,
         experiment_name, test_log):
    print('Gathering inception metrics...')
    if config['accumulate_stats']:
        utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                               z_, y_, config['n_classes'],
                               config['num_standing_accumulations'])
    if config['test_with_tf_inception']:
        metrics = test_with_tf_inception(G_ema if config['ema'] and config['use_ema'] else G,
                                         config)
        IS_mean, IS_std, FID, FID_all = 0.0, 0.0, metrics['FID']['per_class'][config['new_class']], metrics['FID']['all']
        print('Itr %d: Inception Score is %3.3f +/- %3.3f, FID is %5.4f, intra-fid is %5.4f' % (state_dict['itr'], IS_mean, IS_std, FID_all, FID))
    else:
        IS_mean, IS_std, FID = get_inception_metrics(sample, 
                                                 config['num_inception_images'],
                                                 num_splits=10)
        FID_all = FID
        print('Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (state_dict['itr'], IS_mean, IS_std, FID))
    # If improved over previous best metric, save approrpiate copy
    if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
      or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])):
        print('%s improved over previous best, saving checkpoint...' % config['which_best'])
        utils.save_weights(G, D, state_dict, config['weights_root'],
                           experiment_name, 'best%d' % state_dict['save_best_num'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num'] = (state_dict['save_best_num'] + 1 ) % config['num_best_copies']
    state_dict['best_IS'] = max(state_dict['best_IS'], IS_mean)
    state_dict['best_FID'] = min(state_dict['best_FID'], FID)
    # Log results to file
    test_log.log(itr=int(state_dict['itr']), IS_mean=float(IS_mean),
                 IS_std=float(IS_std), FID=float(FID), FID_all=float(FID_all))

'''This function runs the inception metrics with corrected pytorch model loading original tf weights
'''
def test_with_tf_inception(G, config, sample_per_class=5000):
    # sample images
    y_ = torch.cuda.LongTensor([[i]*sample_per_class for i in range(config['n_classes'])]).reshape(-1)
    G_z_list = []
    idx, total = 0, y_.size(0)
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    with torch.no_grad():
        while idx < total:
            b_size = min(G_batch_size, total - idx)
            z_ = torch.randn(b_size, G.dim_z, device='cuda')
            G_z = G(z_, G.shared(y_[idx:idx+b_size]))
            G_z_list.append(G_z)
            idx += b_size
    G_z_list = torch.cat(G_z_list)

    # correct for normalization
    G_z_list = G_z_list * 0.5 + 0.5

    # compute FID and intra-FID score
    metrics = compute_IS_and_fid(G_z_list.cpu().numpy(), labels=y_.cpu().numpy(), compute_IS=False)
    return metrics
