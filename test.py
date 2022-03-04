"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import os
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from utils import sample, trim_tokens
import argparse
from sdataset import  CSVLayout
from model import GPT, GPTConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Layout Transformer')
    parser.add_argument("--ckpt", default='/Users/wangyanqing/Downloads/checkpoint.pth', help="path to checkpoint")
    parser.add_argument("--dir", default='./outputs', help="path to output")
    parser.add_argument("--train_csv", default="/Users/wangyanqing/Downloads/synz-master/data/rico_train.csv", help="/path/to/train/csv")
    parser.add_argument("--val_csv", default="./testfile5.csv", help="/path/to/val/csv")
    # Architecture/training options
    parser.add_argument('--n_layer', default=6, type=int)
    parser.add_argument('--n_embd', default=512, type=int)
    parser.add_argument('--n_head', default=8, type=int)

    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")


    train_dataset = CSVLayout(args.train_csv)
    valid_dataset = CSVLayout(args.val_csv, max_length=train_dataset.max_length)



    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_length,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)  # a GPT-1
    model = GPT(mconf)
    if args.ckpt :
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(checkpoint)
        print(f"loaded checkpoint from {args.ckpt}")

        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).to(device)

    pad_token = train_dataset.vocab_size - 1
    eos_token = train_dataset.vocab_size - 2

    

    # best_loss = float('inf')

    with torch.no_grad():
        model.train(False)
        data = valid_dataset
        loader = DataLoader(data, shuffle=True, pin_memory=True,
                            batch_size=64,
                            num_workers=8)

        # pbar = tqdm(enumerate(loader), total=len(loader)) 
        for x, y, all in loader:

            # place data on the correct device
            x = x.to(device)

            layouts = x.detach().cpu().numpy()
            for i, layout in enumerate(layouts):
                layout = train_dataset.render(layout)
                layout.save(os.path.join(args.dir, f'input_{i:02d}.png'))

            # reconstruction
            # x_cond = x.to(device)
            # print(x_cond)
            # logits, _ = model(x_cond)
            # probs = F.softmax(logits, dim=-1)
            # _, y = torch.topk(probs, k=1, dim=-1)
            # layouts = torch.cat((x_cond[:, :1], y[:, :, 0]), dim=1).detach().cpu().numpy()
            # print('reconstruction')
            # for i, layout in enumerate(layouts):
            #     layout = train_dataset.render(layout)
            #     layout.save(os.path.join(args.dir, f'recon_{i:02d}.png'))

            # # x_cond[x_cond==eos_token]=pad_token
            # # x_cond = x_cond.squeeze()
            # # x_cond = x_cond[:torch.where(x_cond==eos_token)[0]]
            # # x_cond = x_cond.view(1,-1)
            # input_items=12
            # x_cond = x_cond[:,:1+5*input_items]
            # print(x_cond)

            # samples - random
            # layouts = sample(model, x.to(device), steps=train_dataset.max_length,
            #                     temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
            # print('random')

            # for i, layout in enumerate(layouts):
            #     layout = train_dataset.render(layout)
            #     layout.save(os.path.join(args.dir, f'sample_random_{i:02d}.png'))

            # samples - deterministic
            x_cond = trim_tokens(x[0], train_dataset.eos_token, train_dataset.pad_token).unsqueeze(0).to(device)
            layouts = sample(model, x_cond, steps=train_dataset.max_item,
                                temperature=1.0, sample=False, top_k=None).detach().cpu().numpy()
            print('deterministic')

            for i, layout in enumerate(layouts):
                print(layouts)
                layout = train_dataset.render(layout)
                layout.save(os.path.join(args.dir, f'sample_det_{i:02d}.png'))







