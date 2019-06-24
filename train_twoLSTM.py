import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models.TwoLSTM import DecoderWithAttention
from models.models import Encoder
from data.datasets import *
from utils.utils import *
from tensorboardX import SummaryWriter
from nltk.translate.bleu_score import corpus_bleu
# Data parameters
# folder with data files saved by create_input_files.py
data_folder = '../../datasets/coco2014/'
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
# sets device for model and PyTorch tensors
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.benchmark = True

# Training parameters
start_epoch = 0
# number of epochs to train for (if early stopping is not triggered)
epochs = 30
# keeps track of number of epochs since there's been an improvement in validation BLEU
epochs_since_improvement = 0
batch_size = 8
workers = 0  # for data-loading; right now, only 1 works with h5py
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 1  # print training/validation stats every __ batches
# path to checkpoint, None if none
checkpoint = './BEST_two_lstmcheckpoint_two_lstmcoco_5_cap_per_img_5_min_word_freq.pth.tar'
fine_tune_encoder = True
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, data_name, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)

        decoder_optimizer = torch.optim.Adam(params=filter(
            lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
        encoder = encoder.to(device)
        decoder = decoder.to(device)
    else:
        checkpoint = torch.load(checkpoint, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder = decoder.to(device)
        decoder.load_state_dict(checkpoint['decoder'])
        decoder_optimizer = torch.optim.Adam(params=filter(
            lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr)
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        encoder = Encoder()
        encoder = encoder.to(device)
        encoder.fine_tune(fine_tune_encoder)
        encoder.load_state_dict(checkpoint['encoder'])
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])

    # Loss functions
    criterion_ce = nn.CrossEntropyLoss().to(device)
    # criterion_dis = nn.MultiLabelMarginLoss().to(device)

    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    writer = SummaryWriter(log_dir='./log_two')

    # for group in encoder_optimizer.param_groups:
    #     for param in group['params']:
    #         print('eL2 :{}, max :{} , min :{}, mean: {}'.format(
    #             param.data.norm().item(), param.data.max().item(), param.data.min().item(),
    #             param.data.mean().item()))

    # for group in decoder_optimizer.param_groups:
    #     for param in group['params']:
    #         print('dL2 :{}, max :{} , min :{}, mean: {}'.format(
    #             param.data.norm().item(), param.data.max().item(), param.data.min().item(),
    #             param.data.mean().item()))

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 4 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion_ce=criterion_ce,
              # criterion_dis=criterion_dis,
              decoder_optimizer=decoder_optimizer,
              encoder_optimizer=encoder_optimizer,
              epoch=epoch,
              writer=writer)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion_ce=criterion_ce,
                                # criterion_dis=criterion_dis,
                                epoch=epoch,
                                writer=writer)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" %
                  (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint

        saveatwoLSTM_checkpoint(data_name, epoch, epochs_since_improvement,
                                encoder, decoder, encoder_optimizer, decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion_ce, encoder_optimizer, decoder_optimizer, epoch, writer):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """
    encoder.train()
    decoder.train()  # train mode (dropout and batchnorm is used)
    total_step = len(train_loader)
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        features = encoder(imgs)
        # Forward prop.
        scores, scores_d, caps_sorted, decode_lengths, sort_ind = decoder(
            features, caps, caplens)

        # Max-pooling across predicted words across time steps for discriminative supervision
        scores_d = scores_d.max(1)[0]

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        targets_d = torch.zeros(scores_d.size(0), scores_d.size(1)).to(device)
        targets_d.fill_(-1)

        for length in decode_lengths:
            targets_d[:, :length-1] = targets[:, :length-1]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _ = pack_padded_sequence(
            scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(
            targets, decode_lengths, batch_first=True)

        # Calculate loss
        #loss_d = criterion_dis(scores_d, targets_d.long())
        loss_g = criterion_ce(scores, targets)
        loss = loss_g  # + (10 * loss_d)

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()

        loss.backward()

        # Clip gradients when they are getting too large
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, decoder.parameters()), 0.4)
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, encoder.parameters()), 0.4)

        # Update weights
        if encoder_optimizer is not None:
            encoder_optimizer.step()
        decoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            writer.add_scalars(
                'train: ', {'loss': loss.item(), 'mAp': top5accs.val}, epoch*total_step+i)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion_ce, epoch, writer):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :return: BLEU-4 score
    """
    encoder.eval()
    decoder.eval()  # eval mode (no dropout or batchnorm)
    total_step = len(val_loader)
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            features = encoder(imgs)
            scores, scores_d, caps_sorted, decode_lengths, sort_ind = decoder(
                features, caps, caplens)

            # Max-pooling across predicted words across time steps for discriminative supervision
            scores_d = scores_d.max(1)[0]

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]
            targets_d = torch.zeros(scores_d.size(
                0), scores_d.size(1)).to(device)
            targets_d.fill_(-1)

            for length in decode_lengths:
                targets_d[:, :length-1] = targets[:, :length-1]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(
                scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(
                targets, decode_lengths, batch_first=True)

            # Calculate loss
            # loss_d = criterion_dis(scores_d, targets_d.long())
            loss_g = criterion_ce(scores, targets)
            loss = loss_g  # + (10 * loss_d)

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                writer.add_scalars(
                    'val: ', {'loss': loss.item(), 'mAp': top5accs.val}, epoch*total_step+i)
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            # because images were sorted in the decoder
            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        weights = (1.0 / 1.0,)
        bleu1 = corpus_bleu(references, hypotheses, weights)

        weights = (1.0 / 2.0, 1.0 / 2.0,)
        bleu2 = corpus_bleu(references, hypotheses, weights)

        weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,)
        bleu3 = corpus_bleu(references, hypotheses, weights)
        bleu4 = corpus_bleu(references, hypotheses)
        writer.add_scalars(
            'Bleu: ', {'Bleu1': bleu1, 'Bleu2': bleu2, 'Bleu3': bleu3, 'Bleu4': bleu4}, epoch)
        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
