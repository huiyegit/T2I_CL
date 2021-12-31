import cog
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import tempfile
from torch.autograd import Variable
from nltk.tokenize import RegexpTokenizer
from AttnGAN_CL.code.miscc.config import cfg, cfg_from_file
from AttnGAN_CL.code.datasets import TextDataset
from AttnGAN_CL.code.trainer import condGANTrainer as trainer
from AttnGAN_CL.code.model import RNN_ENCODER
from AttnGAN_CL.code.model import G_DCGAN, G_NET
from AttnGAN_CL.code.miscc.utils import build_super_images, build_super_images2


class Predictor(cog.Predictor):
    def setup(self):
        config()

    @cog.input("sentence", type=str, help="text for image generation, describe an image with a bird")
    def predict(self, sentence):
        algo, data_dic = gen_image(sentence)
        output = gen_example(algo, data_dic)
        return output


def config():
    cfg_file = 'AttnGAN_CL/code/cfg/eval_bird.yml'
    cfg_from_file(cfg_file)
    if torch.cuda.is_available():
        cfg.GPU_ID = 0
    else:
        cfg.CUDA = False
        cfg.GPU_ID = -1

    manual_seed = 100
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(manual_seed)


def gen_image(sentence):
    split_dir = 'test'
    bshuffle = True
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)

    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    output_dir = './output'
    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword, dataset)
    data_dic = gen_data_dic(dataset.wordtoix, sentence)
    return algo, data_dic


def gen_data_dic(wordtoix, sentence):
    data_dic = {}
    sentences = ['b', sentence]
    # a list of indices for a sentence
    captions = []
    cap_lens = []
    for sent in sentences:
        if len(sent) == 0:
            continue
        sent = sent.replace("\ufffd\ufffd", " ")
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sent.lower())
        if len(tokens) == 0:
            continue

        rev = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0 and t in wordtoix:
                rev.append(wordtoix[t])
        captions.append(rev)
        cap_lens.append(len(rev))

    max_len = np.max(cap_lens)

    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype='int64')
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
    data_dic['current'] = [cap_array, cap_lens, sorted_indices]
    return data_dic


def gen_example(algo, data_dic):

    if cfg.TRAIN.NET_G == '':
        print('Error: the path for morels is not found!')
    else:
        # Build and load the generator
        text_encoder = \
            RNN_ENCODER(algo.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
        text_encoder.eval()

        if cfg.GAN.B_DCGAN:
            netG = G_DCGAN()
        else:
            netG = G_NET()
        model_dir = cfg.TRAIN.NET_G
        state_dict = \
            torch.load(model_dir, map_location=lambda storage, loc: storage)
        netG.load_state_dict(state_dict)
        print('Load G from: ', model_dir)
        if cfg.CUDA:
            netG.cuda()
        netG.eval()
        for key in data_dic:
            captions, cap_lens, sorted_indices = data_dic[key]
            batch_size = captions.shape[0]
            nz = cfg.GAN.Z_DIM
            captions = Variable(torch.from_numpy(captions), volatile=True)
            cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)
            if cfg.CUDA:
                captions = captions.cuda()
                cap_lens = cap_lens.cuda()

            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            if cfg.CUDA:
                noise = noise.cuda()
            #######################################################
            # (1) Extract text embeddings
            ######################################################
            hidden = text_encoder.init_hidden(batch_size)
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            mask = (captions == 0)
            #######################################################
            # (2) Generate fake images
            ######################################################
            noise.data.normal_(0, 1)
            fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
            im = fake_imgs[-1][0].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            out_path = Path(tempfile.mkdtemp()) / "out.png"
            im.save(str(out_path))

    return out_path
