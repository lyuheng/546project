import pdb
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DecoderLayer(nn.Module):
    """Implements a single layer of an unconditional ImageTransformer"""
    def __init__(self, hidden_size=256, filter_size=512, num_heads=8, dropout=0.1, block_length=256):
        super().__init__()
        self.attn = Attn(hidden_size, num_heads, dropout, block_length)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm_attn = nn.LayerNorm([hidden_size], eps=1e-6, elementwise_affine=True)
        self.layernorm_ffn = nn.LayerNorm([hidden_size], eps=1e-6, elementwise_affine=True)
        self.ffn = nn.Sequential(nn.Linear(hidden_size, filter_size, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(filter_size, hidden_size, bias=True))

    # Takes care of the "postprocessing" from tensorflow code with the layernorm and dropout
    def forward(self, X):
        y = self.attn(X)
        X = self.layernorm_attn(self.dropout(y) + X)
        y = self.ffn(X)
        X = self.layernorm_ffn(self.dropout(y) + X)
        return X



class Attn(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, block_length):
        super().__init__()
        self.kd = hidden_size
        self.vd = hidden_size
        self.q_dense = nn.Linear(hidden_size, self.kd, bias=False)
        self.k_dense = nn.Linear(hidden_size, self.kd, bias=False)
        self.v_dense = nn.Linear(hidden_size, self.vd, bias=False)
        self.output_dense = nn.Linear(self.vd, hidden_size, bias=False)
        assert self.kd % num_heads == 0
        assert self.vd % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout

        self.attn_type = 'local_2d'
        self.block_length = block_length

        self.local_mask_2d = None

    def dot_product_attention(self, q, k, v, bias=None):
        logits = torch.einsum("...kd,...qd->...qk", k, q)
        if bias is not None:
            logits += bias
        weights = F.softmax(logits, dim=-1)
        return weights @ v

    def forward(self, X):
        """
        x: (bs,len,hidden_size=256)
        Return:(bs,len,256)
        """
        q = self.q_dense(X)
        k = self.k_dense(X)
        v = self.v_dense(X)
        # Split to shape (bs,len,8,32)->[bs, 8, len, 32]
        q = q.view(q.shape[:-1] + (self.num_heads, self.kd // self.num_heads)).permute([0, 2, 1, 3])
        k = k.view(k.shape[:-1] + (self.num_heads, self.kd // self.num_heads)).permute([0, 2, 1, 3])
        v = v.view(v.shape[:-1] + (self.num_heads, self.vd // self.num_heads)).permute([0, 2, 1, 3])
        q *= (self.kd // self.num_heads) ** (-0.5)

        if self.attn_type == "global":
            bias = -1e9 * torch.triu(torch.ones(X.shape[1], X.shape[1]), 1).to(X.device)
            result = self.dot_product_attention(q, k, v, bias=bias)
        elif self.attn_type == "local_1d":
            len = X.shape[1]
            blen = self.block_length
            pad = (0, 0, 0, (-len) % self.block_length) # Append to multiple of block length
            q = F.pad(q, pad)
            k = F.pad(k, pad)
            v = F.pad(v, pad)

            
            bias = -1e9 * torch.triu(torch.ones(blen, blen), 1).to(X.device)
            first_output = self.dot_product_attention(
                q[:,:,:blen,:], k[:,:,:blen,:], v[:,:,:blen,:], bias=bias) # (bs,8,blen,32)

            if q.shape[2] > blen:
                q = q.view(q.shape[0], q.shape[1], -1, blen, q.shape[3]) # (bs,8,nblocks,block_len,32)
                k = k.view(k.shape[0], k.shape[1], -1, blen, k.shape[3]) # (bs,8,nblocks,block_len,32)
                v = v.view(v.shape[0], v.shape[1], -1, blen, v.shape[3]) # (bs,8,nblocks,block_len,32)
                local_k = torch.cat([k[:,:,:-1], k[:,:,1:]], 3) # [batch,8,nblocks-1,blen*2,32]
                local_v = torch.cat([v[:,:,:-1], v[:,:,1:]], 3) # [batch,8,nblocks-1,blen*2,32]
                tail_q = q[:,:,1:] # (bs,8,nblock-1,blen, 32)
                bias = -1e9 * torch.triu(torch.ones(blen, 2 * blen), blen + 1).to(X.device)
                tail_output = self.dot_product_attention(tail_q, local_k, local_v, bias=bias) # (bs,8,nblocks-1,blen,32)
                tail_output = tail_output.view(tail_output.shape[0], tail_output.shape[1], -1, tail_output.shape[4]) # (bs,8,(nblocks-1)*blen,32)
                result = torch.cat([first_output, tail_output], 2) # (bs,8,len,32)
                result = result[:,:,:X.shape[1],:] # remove padding
            else:
                result = first_output[:,:,:X.shape[1],:]

        elif self.attn_type == "local_2d":
            len = X.shape[1]
            blen = self.block_length
            pad = (0, 0, 0, (-len) % self.block_length) # Append to multiple of block length
            q = F.pad(q, pad)
            k = F.pad(k, pad)
            v = F.pad(v, pad)

            if self.local_mask_2d == None:
                self.local_mask_2d = -1e9 * torch.ones(blen, 2 * blen).to(X.device)
                h = 32
                for current_pos in range(blen, 2*blen):
                    current_h, current_w = current_pos//h, current_pos%h
                    for other_pixel_pos in range(current_pos + 1):
                        other_pixel_h, other_pixel_w = other_pixel_pos//h, other_pixel_pos%h
                        if abs(current_h - other_pixel_h) <= 6 and abs(current_w - other_pixel_w) <= 6: # set kernel_sz = 6
                            self.local_mask_2d[current_pos-blen, other_pixel_pos] = 0

            bias = -1e9 * torch.triu(torch.ones(blen, blen), 1).to(X.device)
            first_output = self.dot_product_attention(
                q[:,:,:blen,:], k[:,:,:blen,:], v[:,:,:blen,:], bias=bias) # (bs,8,blen,32)

            if q.shape[2] > blen:
                q = q.view(q.shape[0], q.shape[1], -1, blen, q.shape[3]) # (bs,8,nblocks,block_len,32)
                k = k.view(k.shape[0], k.shape[1], -1, blen, k.shape[3]) # (bs,8,nblocks,block_len,32)
                v = v.view(v.shape[0], v.shape[1], -1, blen, v.shape[3]) # (bs,8,nblocks,block_len,32)
                local_k = torch.cat([k[:,:,:-1], k[:,:,1:]], 3) # [batch,8,nblocks-1,blen*2,32]
                local_v = torch.cat([v[:,:,:-1], v[:,:,1:]], 3) # [batch,8,nblocks-1,blen*2,32]
                tail_q = q[:,:,1:] # (bs,8,nblock-1,blen, 32)
                # bias = -1e9 * torch.triu(torch.ones(blen, 2 * blen), blen + 1).to(X.device)
                tail_output = self.dot_product_attention(tail_q, local_k, local_v, bias=self.local_mask_2d) # (bs,8,nblocks-1,blen,32)
                tail_output = tail_output.view(tail_output.shape[0], tail_output.shape[1], -1, tail_output.shape[4]) # (bs,8,(nblocks-1)*blen,32)
                result = torch.cat([first_output, tail_output], 2) # (bs,8,len,32)
                result = result[:,:,:X.shape[1],:] # remove padding
            else:
                result = first_output[:,:,:X.shape[1],:]
            
            

        result = result.permute([0, 2, 1, 3]).contiguous() # (bs,len,8,32)
        result = result.view(result.shape[0:2] + (-1,)) # (bs,len,256)
        result = self.output_dense(result)  # # (bs,len,256)
        return result



class Transformer(nn.Module):
    """ImageTransformer with DMOL or categorical distribution."""
    def __init__(self, nlayers=6, dropout=0.1, distr='dmol', hidden_size=256, num_mixtures=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.distr = distr
        self.num_mixtures = num_mixtures
        self.channel = 3
        self.image_size = 32


        self.layers = nn.ModuleList([DecoderLayer() for _ in range(nlayers)]) # all use default values
        self.input_dropout = nn.Dropout(p=dropout)
        if self.distr == "dmol": # Discretized mixture of logistic, for ordinal valued inputs
            # assert self.hparams.channels == 3, "Only supports 3 channels for DML"
            size = (1, 3)
            self.embedding_conv = nn.Conv2d(1, hidden_size,
                                            kernel_size=size, stride=size)
            # 10 = 1 + 2c + c(c-1)/2; if only 1 channel, then 3 total
            depth = num_mixtures * 10
            self.output_dense = nn.Linear(self.hidden_size, depth, bias=False)

        elif self.distr == "cat": # Categorical
            self.embeds = nn.Embedding(NUM_PIXELS * self.channels, self.hidden_size)
            self.output_dense = nn.Linear(self.hidden_size, NUM_PIXELS, bias=True)
        else:
            raise ValueError("Only dmol or categorical distributions")

    # TODO: can probably cache this computation. (Be wary of shapes for train vs. predict)
    def add_timing_signal(self, X, min_timescale=1.0, max_timescale=1.0e4):
        """
        X:(bs,h,w,256)
        """
        num_dims = len(X.shape) - 2 # 2 corresponds to batch and hidden_size dimensions
        num_timescales = self.hidden_size // (num_dims * 2)
        log_timescale_increment = np.log(max_timescale / min_timescale) / (num_timescales - 1)
        inv_timescales = min_timescale * torch.exp((torch.arange(num_timescales).float() * -log_timescale_increment))
        inv_timescales = inv_timescales.to(X.device)
        total_signal = torch.zeros_like(X) # Only for debugging purposes
        for dim in range(num_dims):
            length = X.shape[dim + 1] # add 1 to exclude batch dim
            position = torch.arange(length).float().to(X.device)
            scaled_time = position.view(-1, 1) * inv_timescales.view(1, -1)
            signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
            prepad = dim * 2 * num_timescales
            postpad = self.hidden_size - (dim + 1) * 2 * num_timescales
            signal = F.pad(signal, (prepad, postpad))
            for _ in range(1 + dim):
                signal = signal.unsqueeze(0)
            for _ in range(num_dims - 1 - dim):
                signal = signal.unsqueeze(-2)
            X += signal
            total_signal += signal
        return X

    def shift_and_pad_(self, X):
        """
        X:(bs,h,w,256)
        """
        # Shift inputs over by 1 and pad
        shape = X.shape
        X = X.view(shape[0], shape[1] * shape[2], shape[3]) # (bs,h*w,256)
        X = X[:,:-1,:] # (bs,h*w-1,256)
        X = F.pad(X, (0, 0, 1, 0)) # Pad second to last dimension
        X = X.view(shape) # (bs,h,w,256)
        return X

    def forward(self, X, sampling=False):
        # Reshape inputs
        """
        X:(bs,3,h,w)
        """
        if sampling:
            curr_infer_length = X.shape[1] 
            row_size = self.image_size * self.channels 
            nrows = curr_infer_length // row_size + 1
            X = F.pad(X, (0, nrows * row_size - curr_infer_length))
            X = X.view(X.shape[0], -1, row_size)
        else:
            X = X.permute([0, 2, 3, 1]).contiguous()  #(bs,h,w,3)
            X = X.view(X.shape[0], X.shape[1], X.shape[2] * X.shape[3]) # Flatten channels into width # (bs,h,w*3)

        # Inputs -> embeddings
        if self.distr == "dmol":
            # Create a "channel" dimension for the 1x3 convolution
            # (NOTE: can apply a 1x1 convolution and not reshape, this is for consistency)
            X = X.unsqueeze(1) # (bs,1,h,w*3)
            X = F.relu(self.embedding_conv(X)) # (bs,256,h,w)
            X = X.permute([0, 2, 3, 1]) # move channels to the end # (bs,h,w,256)

        elif self.distr == "cat":
            # Convert to indexes, and use separate embeddings for different channels
            X = (X * (NUM_PIXELS - 1)).long()
            channel_addition = (torch.tensor([0, 1, 2]) * NUM_PIXELS).to(X.device).repeat(X.shape[2] // 3).view(1, 1, -1)
            X += channel_addition
            X = self.embeds(X) * (self.hidden_size ** 0.5)

        X = self.shift_and_pad_(X) 
        X = self.add_timing_signal(X) # (bs,h,w,256)
        shape = X.shape
        X = X.view(shape[0], -1, shape[3]) # (bs,h*w,256)

        X = self.input_dropout(X)
        for layer in self.layers:
            X = layer(X)
        # X = self.layers[-1].preprocess_(X) # NOTE: this is identity (exists to replicate tensorflow code)
        X = self.output_dense(X).view(shape[:3] + (-1,)) # (bs,h,w,100)

        if not sampling and self.distr == "cat": # Unpack the channels
            X = X.view(X.shape[0], X.shape[1], X.shape[2] // self.hparams.channels, self.hparams.channels, X.shape[3])
            X = X.permute([0, 3, 1, 2, 4])

        return X


class GenerativeTransformer(nn.Module):
    ''' Code partly taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html '''

    def __init__(self, embd_size, d_model=256, nhead=8, nlayers=8, dropout=0.1):
        '''
        :param embd_size: int, dimension of the conditional embedding
        '''

        super(GenerativeTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.transformer = Transformer() # use default value
        self.loss = discretized_mix_logistic_loss

        self.h = 32
        self.w = 32

    def forward(self, imgs, condition_embd):
        '''
        :param imgs: torch.FloatTensor bsize * c * h * w
        :param condition_embd: torch.FloatTensor bsize * embd_size
        :return: outputs : dict of ouputs, this can be {"d_loss" : d_loss, "g_loss" : g_loss"} for a gan
        '''
        initial_imgs = imgs
        bsize, c, h, w = imgs.shape
        model_output = self.transformer(imgs) # (bs,h,w,100)
        model_output = model_output.permute(0,3,1,2) #bsize, 100, h, w
        
        loss = self.loss(initial_imgs, model_output, nmix=10) / (
                    h * w * bsize * c)
        outputs = {"loss": loss, "log_likelihood": None,
                   "log_probs": model_output}  
        return outputs

    def likelihood(self, imgs, condition_embd):
        '''
        :param imgs: torch.FloatTensor bsize * c * h * w
        :param condition_embd: torch.FloatTensor bsize * embd_size
        :return: likelihoods : torch.FloatTensor of size bSize, likelihoods of the images conditioned on the captions
        '''
        return None # As in pixelcnnpp

    def sample_half(self, img, captions_embd):
        '''
        :param captions_embd: torch.FloatTensor bsize * embd_size
        :return: imgs : torch.FloatTensor of size n_imgs * c * h * w
        '''
        self.eval()
        bsize, channels, h, w = [captions_embd.size(0)] + [3, self.h, self.w]
        data = torch.zeros((bsize, channels, h, w), dtype=torch.float32, device=captions_embd.device,
                           requires_grad=False)
        # print(img.shape, data.shape)
        data[:,:,:h//2,:] = img[:,:,:h//2,:]
        with torch.no_grad():
            for i in tqdm(range(h//2,h,1)):
                for j in range(w):
                    out = self.forward(data, captions_embd)
                    out_sample = sample_from_discretized_mix_logistic(out['log_probs'], 10)
                    data[:, :, i, j] = out_sample[:, :, i, j]
        return data

    def sample(self, img, captions_embd):

        self.eval()
        bsize, channels, h, w = [captions_embd.size(0)] + [3, self.h, self.w]
        data = torch.zeros((bsize, channels, h, w), dtype=torch.float32, device=captions_embd.device,
                           requires_grad=False)
        with torch.no_grad():
            for i in tqdm(range(h)):
                for j in range(w):
                    out = self.forward(data, captions_embd)
                    out_sample = sample_from_discretized_mix_logistic(out['log_probs'], 10)
                    data[:, :, i, j] = out_sample[:, :, i, j]
        return data

