# -----------------------------
# -*- coding:utf-8 -*-
# author:kangkang
# datetime:2019/4/22 11:42
# -----------------------------

import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
from torch.nn.utils.weight_norm import weight_norm

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, basemodel='res101', encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        self.basemodel = basemodel
        if self.basemodel == 'vgg16':
            # pretrained ImageNet ResNet-101
            model = torchvision.models.vgg16(pretrained=True)
            encoded_image_size = 28
            # Remove linear and pool layers (since we're not doing classification)
            modules = list(model.features)[:-1]
            self.model = nn.Sequential(*modules)
            # self.model输出大小是[-1, 512, 32, 32]
            # Resize image to fixed size to allow input images of variable size
            self.adaptive_pool = nn.AdaptiveAvgPool2d(
                (encoded_image_size, encoded_image_size))

            self.fine_tune()
        else:
            model = torchvision.models.resnet101(
                pretrained=True)  # pretrained ImageNet ResNet-101

            # Remove linear and pool layers (since we're not doing classification)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            # self.model输出大小是[-1, 2048, 16, 16]
            # Resize image to fixed size to allow input images of variable size
            self.adaptive_pool = nn.AdaptiveAvgPool2d(
                (encoded_image_size, encoded_image_size))

            self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.model(
            images)  # (batch_size, 2048, image_size/32, image_size/32)
        # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = self.adaptive_pool(out)
        # (batch_size, encoded_image_size, encoded_image_size, 2048)
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        if self.basemodel == 'vgg16':
            layer = -6
        else:
            layer = -7

        for p in self.model.parameters():
            p.requires_grad = False
            # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.model.children())[layer:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim, dropout=0.5):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = weight_norm(nn.Linear(encoder_dim, attention_dim))  # linear layer to transform encoded image
        self.decoder_att = weight_norm(nn.Linear(decoder_dim, attention_dim))  # linear layer to transform decoder's output
        self.full_att = weight_norm(nn.Linear(attention_dim, 1))  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(
            encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        # (batch_size, num_pixels)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (
            encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(
            encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        # top_down and language LSTM
        self.top_down_attention = nn.LSTMCell(
            embed_dim + encoder_dim + decoder_dim, decoder_dim, bias=True)
        self.language_model = nn.LSTMCell(
            encoder_dim + decoder_dim, decoder_dim, bias=True)

        # linear layer to find scores over vocabulary
        self.fc1 = weight_norm(nn.Linear(decoder_dim, vocab_size))
        # linear layer to find scores over vocabulary
        self.fc = weight_norm(nn.Linear(decoder_dim, vocab_size))
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, batch_size):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        h = torch.zeros(batch_size, self.decoder_dim).to(
            device)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size, self.decoder_dim).to(device)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        # 获取像素数14*14， 和维度2048
        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size
        encoder_dim = encoder_out.size(-1)
        # Flatten image  encoder_out：[batch_size, 14, 14, 2048]
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        encoder_out_mean = encoder_out.mean(1).to(device)

        # Sort input data by decreasing lengths; why? apparent below
        # 为了torch中的pack_padded_sequence，需要降序排列
        caption_lengths, sort_ind = caption_lengths.squeeze(
            1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoder_out_mean = encoder_out_mean[sort_ind]

        # global_encoder = encoder_out.sum(dim=2)
        # [32, 52]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        # (batch_size, max_caption_length, embed_dim)
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM state
        h1, c1 = self.init_hidden_state(
            batch_size)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(batch_size)
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(
            decode_lengths), vocab_size).to(device)
        predictions1 = torch.zeros(batch_size, max(
            decode_lengths), vocab_size).to(device)

        # old single LSTM
        # alphas = torch.zeros(batch_size, max(
        #     decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            h1, c1 = self.top_down_attention(
                torch.cat([h2[:batch_size_t], encoder_out_mean[:batch_size_t], embeddings[:batch_size_t, t, :]], dim=1), (h1[:batch_size_t], c1[:batch_size_t]))
            attention_weighted_encoding = self.attention(encoder_out[:batch_size_t],
                                                         h1[:batch_size_t])

            preds1 = self.fc1(self.dropout(h1))

            h2, c2 = self.language_model(
                torch.cat([attention_weighted_encoding[:batch_size_t],
                           h1[:batch_size_t]], dim=1),
                (h2[:batch_size_t], c2[:batch_size_t]))
            # embeddings[:batch_size_t, t, :]逐个取出单词
            preds = self.fc(self.dropout(h2))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            predictions1[:batch_size_t, t, :] = preds1

        return predictions, predictions1, encoded_captions, decode_lengths, sort_ind
