import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import torch


def sample_image(model, encoder, output_image_dir, n_row, batches_done, dataloader, device):
    """Saves a grid of generated imagenet pictures with captions"""
    target_dir = os.path.join(output_image_dir, "samples/")
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    captions = []
    gen_imgs = []
    ori_imgs = []
    # get sample captions
    done = False
    while not done:
        for (img, labels_batch, captions_batch) in dataloader:
            captions += captions_batch
            conditional_embeddings = encoder(labels_batch.to(device), captions)
            imgs = model.sample(img, conditional_embeddings).cpu()
            gen_imgs.append(imgs)
            ori_imgs.append(img)

            if len(captions) > n_row ** 2:
                done = True
                break

    gen_imgs = torch.cat(gen_imgs).numpy()/2. + .5
    gen_imgs = np.clip(gen_imgs, 0, 1)
    ori_imgs = torch.cat(ori_imgs).numpy()/2. + .5
    ori_imgs = np.clip(ori_imgs, 0, 1)
    gen_imgs_half = ori_imgs.copy()
    gen_imgs_half[:, :, 32//2:, :] = 0.5


    fig = plt.figure(figsize=((8, 8)))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_row, n_row), axes_pad=0.05)

    # save imgs for generated
    for i in range(n_row ** 2):
        grid[i].imshow(gen_imgs[i].transpose([1, 2, 0]))
        # grid[i].set_title(captions[i])
        grid[i].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labeltop=False,labelbottom=False,labelleft=False,labelright=False)

    save_file = os.path.join(target_dir, "{:013d}.png".format(batches_done))
    plt.savefig(save_file, bbox_inches='tight')
    print("saved  {}".format(save_file))

    # save imgs for original
    fig = plt.figure(figsize=((8, 8)))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_row, n_row), axes_pad=0.05)

    for i in range(n_row ** 2):
        grid[i].imshow(ori_imgs[i].transpose([1, 2, 0]))
        # grid[i].set_title(captions[i])
        grid[i].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labeltop=False,labelbottom=False,labelleft=False,labelright=False)

    save_file = os.path.join(target_dir, "{:013d}_ori.png".format(batches_done))
    plt.savefig(save_file, bbox_inches='tight')
    print("saved  {}".format(save_file))
    plt.close()

    ###########################################

    # fig = plt.figure(figsize=((8, 12)))
    # grid = ImageGrid(fig, 111, nrows_ncols=(n_row, n_row+2), axes_pad=0.05)

    # # save imgs for generated
    # for i in range(n_row * (n_row+2) ):
    #     x = i//3
    #     if i % 3 == 0:
    #         grid[i].imshow(gen_imgs_half[x].transpose([1, 2, 0]))
    #     elif i % 3 == 1:
    #         grid[i].imshow(gen_imgs[x].transpose([1,2,0]))
    #     else:
    #         grid[i].imshow(ori_imgs[x].transpose([1,2,0]))
        
    #     if i == 1 or i == 4: 
    #         grid[i].set_title('gen')
    #     if i == 2 or i == 5:
    #         grid[i].set_title('gt')
    #     grid[i].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labeltop=False,labelbottom=False,labelleft=False,labelright=False)

    # save_file = os.path.join(target_dir, "{:013d}.png".format(batches_done))
    # plt.savefig(save_file, bbox_inches='tight')
    # print("saved  {}".format(save_file))
    # plt.close()

    ###########################################


def load_model(file_path, generative_model):
    dict = torch.load(file_path)
    generative_model.load_state_dict(dict)