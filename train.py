import os
import random
import torch
import torch.optim as optim
import customAudioDataset as data
from customAudioDataset import collate_fn
from utils import set_seed, get_dataset_filelist
from tqdm import tqdm
from model import EncodecModel
from msstftd import MultiScaleSTFTDiscriminator
from losses import total_loss, disc_loss
from scheduler import WarmupCosineLrScheduler
import torch.distributed as dist
import torch.multiprocessing as mp
import hydra
import logging
import warnings
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from utils import AttrDict, build_env
from utils import  get_dataset_filelist, mel_spectrogram
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
import time
from data import SoundDataset, get_dataloader
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def train_encodec(rank, config):
    if config.num_gpus > 1:
        init_process_group(backend=config.dist_config['dist_backend'], init_method=config.dist_config['dist_url'],
                           world_size=config.dist_config['world_size'] * config.num_gpus, rank=rank)
    torch.cuda.manual_seed(config.seed)
    device = torch.device('cuda:{:d}'.format(rank))
    # set encodec model and discriminator model
    model = EncodecModel._get_model(
        config.model.target_bandwidths,
        config.model.sample_rate,
        config.model.channels,
        causal=False, model_norm='none',
        audio_normalize=config.model.audio_normalize,
        segment=1., name='my_encodec')
    disc_model = MultiScaleSTFTDiscriminator(filters=config.model.filters)

    model.cuda()
    disc_model.cuda()

    model = model.to(device)
    disc_model = disc_model.to(device)

    if rank == 0:
        os.makedirs(config.checkpoint.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", config.checkpoint.checkpoint_path)
    if os.path.isdir(config.checkpoint.checkpoint_path):
        cp_g = scan_checkpoint(config.checkpoint.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(config.checkpoint.checkpoint_path, 'do_')
    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        model.load_state_dict(state_dict_g['generator'])
        disc_model.load_state_dict(state_dict_do['discriminator'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
    if config.num_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[rank]).to(device)
        disc_model = DistributedDataParallel(disc_model, device_ids=[rank]).to(device)
        # msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)
        # set optimizer and scheduler, warmup scheduler

    params = [p for p in model.parameters() if p.requires_grad]
    disc_params = [p for p in disc_model.parameters() if p.requires_grad]
    optimizer = optim.Adam([{'params': params, 'lr': config.optimization.lr}], betas=(0.5, 0.9))
    optimizer_disc = optim.Adam([{'params': disc_params, 'lr': config.optimization.disc_lr}], betas=(0.5, 0.9))
    if state_dict_do is not None:
        optimizer.load_state_dict(state_dict_do['optim_g'])
        optimizer_disc.load_state_dict(state_dict_do['optim_d'])

    # get train files
    training_filelist, validation_filelist = get_dataset_filelist(config.datasets)
    # set train dataset
    trainset = data.CustomAudioDataset(config, training_filelist)
    testset = data.CustomAudioDataset(config, validation_filelist, split=False)

    train_sampler = DistributedSampler(trainset) if config.num_gpus > 1 else None
    test_sampler =  None

    
    trainset = SoundDataset(
        training_filelist,
        split=True,
        shuffle=True,
        segment_size=config.segment_size,
        max_length=config.data_max_length
    )

    # train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    trainloader = DataLoader(trainset, num_workers=config.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=config.datasets.batch_size,
                              pin_memory=True,
                              drop_last=True)
    if rank == 0:
        validset = SoundDataset(
            validation_filelist,
            split=False,
            shuffle=True,
            segment_size=config.segment_size,
            max_length=config.data_max_length
        )
        testloader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

    sw = SummaryWriter(os.path.join(config.checkpoint.checkpoint_path, 'logs'))

    scheduler = WarmupCosineLrScheduler(optimizer, max_iter=config.common.max_epoch * len(trainloader), eta_ratio=0.1,
                                        warmup_iter=config.lr_scheduler.warmup_epoch * len(trainloader),
                                        warmup_ratio=1e-4)
    disc_scheduler = WarmupCosineLrScheduler(optimizer_disc, max_iter=config.common.max_epoch * len(trainloader),
                                             eta_ratio=0.1,
                                             warmup_iter=config.lr_scheduler.warmup_epoch * len(trainloader),
                                             warmup_ratio=1e-4)
    model.train()
    disc_model.train()
    logs = {}
    for epoch in range(max(0, last_epoch), config.common.max_epoch):
        # model.train()
        # disc_model.train()
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))
        if config.num_gpus > 1:
            train_sampler.set_epoch(epoch)
        for i, input_wav in enumerate(trainloader):
            if rank == 0:
                start_b = time.time()
            # warmup learning rate, warmup_epoch is defined in config file,default is 5
            input_wav = input_wav.cuda()  # [B, 1, T]: eg. [2, 1, 203760]
            input_wav = torch.autograd.Variable(input_wav.to(device, non_blocking=True))
            input_wav = input_wav.unsqueeze(1)
            
            
            
            output, loss_w, _ = model(input_wav)  # output: [B, 1, T]: eg. [2, 1, 203760] | loss_w: [1]
            logits_real, fmap_real = disc_model(input_wav)

            # update discrminator
            optimizer_disc.zero_grad()
            logits_fake, fmap_fake = disc_model(output.detach())
            loss_disc = disc_loss(logits_real, logits_fake)
            loss_disc.backward(retain_graph=True)
            optimizer_disc.step()


            #update generator
            optimizer.zero_grad()
            logits_real, fmap_real = disc_model(input_wav)
            logits_fake, fmap_fake = disc_model(output)
            loss_g = total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output)
            loss = loss_g + loss_w
            # input_wav_mel = input_wav.unsqueeze(1)
            input_wav_mel =  mel_spectrogram(input_wav.squeeze(1), config.n_fft, config.num_mels, config.sampling_rate, config.hop_size, config.win_size,
                                          config.fmin, config.fmax_for_loss)
            output_wav_mel = mel_spectrogram(output.squeeze(1), config.n_fft, config.num_mels, config.sampling_rate, config.hop_size, config.win_size,
                                          config.fmin, config.fmax_for_loss)
            loss_mel = F.l1_loss(input_wav_mel, output_wav_mel) * 45
            loss.backward()
            optimizer.step()

            

            scheduler.step()
            disc_scheduler.step()
            if rank == 0:
                if steps % config.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(input_wav_mel, output_wav_mel).item()
                        # if config.model.train_discriminator and epoch > config.lr_scheduler.warmup_epoch:
                    print('Steps : {:d}, Gen Loss: {:.3f}, VQ. Loss : {:.3f},  Mel-Spec. Error : {:4.3f}, Disc. Error : {:4.3f}, s/b : {:4.3f}'.
                                format(steps, loss_g.item(), loss_w.item(),
                                    mel_error, loss_disc.item(), time.time() - start_b))
                        
                        # checkpointing
                if steps % config.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(config.checkpoint.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (
                                        model.module if config.num_gpus > 1 else model).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(config.checkpoint.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {
                                        'discriminator': (
                                            disc_model.module if config.num_gpus > 1 else disc_model).state_dict(),
                                        'optim_g': optimizer.state_dict(), 'optim_d': optimizer_disc.state_dict(),
                                        'steps': steps,
                                        'epoch': epoch})
                # Tensorboard summary logging
                if steps % config.summary_interval == 0:
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    # print(disc_loss)

                    sw.add_scalar("training/disc_loss", loss_disc, steps)
                    sw.add_scalar("training/loss_g", loss_g, steps)

                    sw.add_scalar("training/all_commit_loss", loss_w, steps)
                    # sw.add_scalar("training/adversarial_loss", adversarial_loss, steps)
                # Validation
                if steps % config.validation_interval == 0:  # and steps != 0:
                    model.eval()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, input_wav in enumerate(testloader):
                            input_wav = input_wav.to(device)  # [B, 1, T]: eg. [2, 1, 203760]
                            input_wav = input_wav.unsqueeze(1)

                            output = model(input_wav)  # output: [B, 1, T]: eg. [2, 1, 203760] | loss_w: [1]
                            logits_real, fmap_real = disc_model(input_wav)
                            logits_fake, fmap_fake = disc_model(output)
                            loss_disc = disc_loss(logits_real, logits_fake)  # compute discriminator loss

                            loss_g = total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output)
                            input_wav_mel = mel_spectrogram(input_wav.squeeze(1), config.n_fft, config.num_mels,
                                                            config.sampling_rate, config.hop_size, config.win_size,
                                                            config.fmin, config.fmax_for_loss)
                            output_wav_mel = mel_spectrogram(output.squeeze(1), config.n_fft, config.num_mels,
                                                             config.sampling_rate, config.hop_size, config.win_size,
                                                             config.fmin, config.fmax_for_loss)
                            val_err_tot += F.l1_loss(input_wav_mel, output_wav_mel).item()
                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), input_wav[0], steps, config.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j),
                                                  plot_spectrogram(input_wav_mel.squeeze(0).cpu().numpy()), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), output[0], steps, config.sampling_rate)
                                y_hat_spec = mel_spectrogram(output.squeeze(1), config.n_fft, config.num_mels,
                                                             config.sampling_rate, config.hop_size, config.win_size,
                                                             config.fmin, config.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                            # val_err = val_err_tot / (j + 1)
                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)
                    model.train()
            steps += 1

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))




@hydra.main(config_path='config', config_name='config')
def main(config):
    if config.distributed.torch_distributed_debug:  # set distributed debug, if you encouter some multi gpu bug, please set torch_distributed_debug=True
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    if not os.path.exists(config.checkpoint.save_folder):
        os.makedirs(config.checkpoint.save_folder)
    

    train_encodec(0, config)

if __name__ == '__main__':
    main()