import os
import argparse
from tabnanny import check
import torch
from sdataset import MNISTLayout, JSONLayout, CSVLayout
from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from utils import set_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Layout Transformer')
    parser.add_argument("--exp", default="cpu_test", help="experiment name")
    parser.add_argument("--log_dir", default="./logs", help="/path/to/logs/dir")
    parser.add_argument("--ckpt", default=None, help="path to checkpoint")

    # MNIST options
    parser.add_argument("--data_dir", default=None, help="/path/to/mnist/data")
    parser.add_argument("--threshold", type=int, default=16, help="threshold for grayscale values")

    # COCO/PubLayNet options
    parser.add_argument("--train_json", default=None, help="/path/to/train/json")
    parser.add_argument("--val_json", default=None, help="/path/to/val/json")

    # CSV options
    parser.add_argument("--train_csv", default="/Users/wangyanqing/Downloads/data/rico_anno_tiny.csv", help="/path/to/train/json")
    parser.add_argument("--val_csv", default="/Users/wangyanqing/Downloads/data/rico_anno_tiny.csv", help="/path/to/val/json")

    # Layout options
    parser.add_argument("--max_length", type=int, default=128, help="batch size")
    parser.add_argument('--precision', default=8, type=int)
    parser.add_argument('--element_order', default='raster')
    parser.add_argument('--attribute_order', default='cxywh')

    # Architecture/training options
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=4.5e-06, help="learning rate")
    parser.add_argument('--n_layer', default=6, type=int)
    parser.add_argument('--n_embd', default=512, type=int)
    parser.add_argument('--n_head', default=8, type=int)
    # parser.add_argument('--evaluate', action='store_true', help="evaluate only")
    parser.add_argument('--lr_decay', action='store_true', help="use learning rate decay")
    parser.add_argument('--warmup_iters', type=int, default=0, help="linear lr warmup iters")
    parser.add_argument('--final_iters', type=int, default=0, help="cosine lr final iters")
    parser.add_argument('--sample_every', type=int, default=1, help="sample every epoch")

    args = parser.parse_args()

    log_dir = os.path.join(args.log_dir, args.exp)
    samples_dir = os.path.join(log_dir, "samples")
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # MNIST Testing
    if args.data_dir is not None:
        train_dataset = MNISTLayout(args.log_dir, train=True, threshold=args.threshold)
        valid_dataset = MNISTLayout(args.log_dir, train=False, threshold=args.threshold,
                                    max_length=train_dataset.max_length)
    #CSV 
    elif args.train_csv is not None:
        train_dataset = CSVLayout(args.train_csv)
        valid_dataset = CSVLayout(args.val_csv, max_length=train_dataset.max_length)
    # COCO and PubLayNet
    else:
        train_dataset = JSONLayout(args.train_json)
        valid_dataset = JSONLayout(args.val_json, max_length=train_dataset.max_length)

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_length,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)  # a GPT-1
    model = GPT(mconf)
    if args.ckpt :
        checkpoint = torch.load(args.ckpt)
        model = model.load_state_dict(checkpoint)
        print(f"loaded checkpoint from {args.ckpt}")
    tconf = TrainerConfig(max_epochs=args.epochs,
                          batch_size=args.batch_size,
                          lr_decay=args.lr_decay,
                          learning_rate=args.lr * args.batch_size,
                          warmup_iters=args.warmup_iters,
                          final_iters=args.final_iters,
                          ckpt_dir=ckpt_dir,
                          samples_dir=samples_dir,
                          sample_every=args.sample_every)
    trainer = Trainer(model, train_dataset, valid_dataset, tconf, args)
    trainer.train()
