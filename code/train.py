import argparse
import os
from dataset import *
from models import *
import torch
from utils import *
from sample import *
from torch.optim import lr_scheduler
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np

class UnconditionalClassEmbedding(nn.Module):
    def __init__(self, embed_size=1):
        super(UnconditionalClassEmbedding, self).__init__()
        self.embed_size = embed_size

    def forward(self, class_labels, captions):
        zero = torch.zeros(class_labels.size(0), 1).to(class_labels.device)
        return zero

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument('--lr_decay', type=float, default=0.99,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
parser.add_argument("--use_cuda", type=int, default=1, help="use cuda if available")
parser.add_argument("--output_dir", type=str, default="outputs/transformer", help="directory to store the sampled outputs")
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--train_on_val", type=int, default=0, help="train on val set, useful for debugging")
parser.add_argument("--train", type=int, default=0, help="0 = eval, 1=train") # use eval
parser.add_argument("--model_checkpoint", type=str, default='./outputs/transformer/models/epoch_best_8layer_2d_local.pt',
                    help="load model from checkpoint, model_checkpoint = path_to_your_pixel_cnn_model.pt")
parser.add_argument("--print_every", type=int, default=10)
parser.add_argument("--dataset", type=str, default="cifar10", choices=["imagenet32", "cifar10"])
parser.add_argument("--conditioning", type=str, default="unconditional", choices=["unconditional", "one-hot", "bert"])
parser.add_argument("--tier", type=int, default=1)
parser.add_argument("--gradient_accumulation", type=int, default=1, help="number of batches to accumulate before gradient")

parser.add_argument("--nlayers", type=int, default=8, help="number of layers for transformer")
parser.add_argument("--nhead", type=int, default=4, help="number of heads for transformer")
parser.add_argument("--d_model", type=int, default=256, help="number of dims of embeddings for transformer")
# These parameters are the maximum we can use for K80 memory


def get_lr(step):
    warmup_steps = 4000
    lr_base = 0.1 * 0.002 # for Adam correction
    ret = 5000. * 256 ** (-0.5) * np.min([(step + 1) * warmup_steps ** (-1.5), (step + 1) ** (-0.5)])
    return ret * lr_base

def train(model, embedder, optimizer, scheduler,
          train_loader, val_loader, opt):
    print(f"Batch size is {opt.batch_size} and gradient accumulation {opt.gradient_accumulation}"
          f" so actual batch size is {opt.batch_size*opt.gradient_accumulation}")
    print("TRAINING STARTS")
    best_bpd = 1000.
    for epoch in range(opt.n_epochs):
        model = model.train()
        loss_to_log = 0.0
        for i, (imgs, labels, captions) in enumerate(train_loader):
            start_batch = time.time()
            imgs = imgs.to(device)
            labels = labels.to(device)
            # 

            with torch.no_grad():
                condition_embd = embedder(labels, captions)

            outputs = model.forward(imgs, condition_embd)
            loss = outputs['loss'].mean() / opt.gradient_accumulation
            loss_b = 32*32*3*loss  # 
            loss_b.backward()

            if (i + 1) % opt.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()
            

            batches_done = epoch * len(train_loader) + i
            writer.add_scalar('train/bpd', loss / np.log(2), batches_done)
            loss_to_log += loss.item() * opt.gradient_accumulation
            if (i + 1) % opt.print_every == 0:
                loss_to_log = loss_to_log / (np.log(2) * opt.print_every)
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [bpd: %f] [Time/batch %.3f] [Lr %.3E]"
                    % (epoch + 1, opt.n_epochs, i + 1, len(train_loader), loss_to_log, time.time() - start_batch, scheduler.get_lr()[0])
                )
                loss_to_log = 0.0

            # if (batches_done + 1) % opt.sample_interval == 0:
            #     print("sampling_images")
            #     model = model.eval()
            #     sample_image(model, embedder, opt.output_dir, n_row=4,
            #                  batches_done=batches_done,
            #                  dataloader=val_loader, device=device)
            #     model = model.train()
                
        val_bpd = eval(model, embedder, val_loader)
        writer.add_scalar("val/bpd", val_bpd, (epoch + 1) * len(train_loader))

        # updated ! 
        torch.save(model.state_dict(),
                   os.path.join(opt.output_dir, 'models', 'epoch_latest_8layer_2d_local.pt'))
        if val_bpd < best_bpd:
            best_bpd = val_bpd
            torch.save(model.state_dict(),
                   os.path.join(opt.output_dir, 'models', 'epoch_best_8layer_2d_local.pt'))

def eval(model, embedder, test_loader):
    print("EVALUATING ON VAL")
    model = model.eval()
    bpd = 0.0
    sample = True
    for i, (imgs, labels, captions) in tqdm(enumerate(test_loader)):
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            condition_embd = embedder(labels, captions)
            outputs = model.forward(imgs, condition_embd)
            loss = outputs['loss'].mean()
            bpd += loss / np.log(2)


            # sample at eval func
            if sample == True:
                # sample image
                sample_image(model, embedder, opt.output_dir, n_row=4,
                             batches_done=0,
                             dataloader=test_loader, device=device)
                sample = False
                break
            
    bpd /= len(test_loader)
    print("VAL bpd : {}".format(bpd))
    return bpd

if __name__ == "__main__":
    # opt = parser.parse_args()
    opt, unknown = parser.parse_known_args()
    print(opt)

    # vocab_file = "map_clsloc.txt"
    # if opt.tier == 2:
    #     vocab_file = "map_clsloc2.txt"
    #     print("Categories: 2")
    # if opt.tier == 3:
    #     vocab_file = "map_clsloc3.txt"
    #     print("Categories: 3")

    print("loading dataset")
    if opt.dataset == "imagenet32":
        train_dataset = Imagenet32Dataset(train=not opt.train_on_val, max_size=1 if opt.debug else -1, vocab_file=vocab_file)
        val_dataset = Imagenet32Dataset(train=0, max_size=1 if opt.debug else -1, vocab_file=vocab_file)
    else:
        assert opt.dataset == "cifar10"
        train_dataset = CIFAR10Dataset(train=not opt.train_on_val)
        val_dataset = CIFAR10Dataset(train=0)


    print("creating dataloaders")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
    )



    print("Len train : {}, val : {}".format(len(train_dataloader), len(val_dataloader)))

    device = torch.device("cuda") if (torch.cuda.is_available() and opt.use_cuda) else torch.device("cpu")
    print("Device is {}".format(device))

    print("Loading models on device...")

    # Initialize embedder
    if opt.conditioning == 'unconditional':
        encoder = UnconditionalClassEmbedding()
    elif opt.conditioning == "bert":
        encoder = BERTEncoder()
    else:
        assert opt.conditioning == "one-hot"
        encoder = OneHotClassEmbedding(train_dataset.n_classes)

    generative_model = GenerativeTransformer(embd_size=encoder.embed_size,)  # use default params

    generative_model = generative_model.to(device)
    encoder = encoder.to(device)
    print("Models loaded on device")

    print("dataloaders loaded")

    print("Conditioning initial dimension:", encoder.embed_size)

    print("Initializing model parameters...")
    gain = 0.2
    for name, p in generative_model.named_parameters():
        if "layernorm" in name:
            continue
        # This is from a pytorch implementation of the language transformer, but is not needed/in TF code.
        # if "attn" in name and "output" not in name:
        #     nn.init.xavier_normal_(p)
        if p.dim() > 1:
            nn.init.xavier_uniform_(p, gain=np.sqrt(gain)) # Need sqrt for inconsistency between pytorch / TF
        else:
            a =  np.sqrt(3. * gain / p.shape[0])
            nn.init.uniform_(p, -a, a)

    # Optimizers
    optimizer = torch.optim.Adam(generative_model.parameters(), lr=1., betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: get_lr(step))
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: get_lr(step))
    # create output directory
    # optimizer_a = torch.optim.Adam(generative_model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

    os.makedirs(os.path.join(opt.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, "tensorboard"), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, "tensorboard"))

    # ----------
    #  Training
    # ----------
    if opt.train:
        # print("Loading model from state dict...")
        # load_model(opt.model_checkpoint, generative_model)
        # print("Model loaded.")
        train(model=generative_model, embedder=encoder, optimizer=optimizer, scheduler=scheduler,
              train_loader=train_dataloader, val_loader=val_dataloader, opt=opt) # use a small dataset to overfit
    else:
        assert opt.model_checkpoint is not None, 'no model checkpoint specified'
        print("Loading model from state dict...")
        load_model(opt.model_checkpoint, generative_model)
        print("Model loaded.")
        eval(model=generative_model, embedder=encoder, test_loader=train_dataloader)
        
