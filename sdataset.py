import numpy as np
import torch
from torchvision.datasets.mnist import MNIST
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageDraw, ImageOps, ImageFont
import pandas as pd
import random

from utils import trim_tokens, gen_colors


class Padding(object):
    def __init__(self, max_length, vocab_size):
        self.max_length = max_length
        # self.bos_token = vocab_size - 3
        self.eos_token = vocab_size - 2
        self.pad_token = vocab_size - 1

    def __call__(self, layout):
        # grab a chunk of (max_length + 1) from the layout
        chunk = torch.zeros(self.max_length+1, dtype=torch.long) + self.pad_token
        # Assume len(item) will always be <= self.max_length:
        # chunk[0] = self.bos_token
        chunk[0:len(layout)] = layout
        chunk[len(layout)+1] = self.eos_token

        x = chunk[:-1]
        y = chunk[1:]
        return {'x': x, 'y': y}

class CSVLayout(Dataset):
    def __init__(self, csv_path, max_length=None,max_item=46, precision=8):
        df = pd.read_csv(csv_path)
        self.df = df
    #     self.categories = ['Icon', 'Text', 'Advertisement', 'Web View', 'Text Button',
    #    'Slider', 'Image', 'Toolbar', 'List Item', 'Multi-Tab', 'Card',
    #    'Pager Indicator', 'Button Bar', 'Input', 'Modal', 'Date Picker',
    #    'Drawer', 'Background Image', 'On/Off Switch', 'Map View',
    #    'Bottom Navigation', 'Video', 'Radio Button', 'Checkbox',
    #    'Number Stepper']
        self.categories = ['Icon', 'Text',  'Text Button',
       'Slider', 'Image',
       'Pager Indicator','Input', 
        'On/Off Switch', 'Map View',
       'Video', 'Radio Button', 'Checkbox']
        self.width=1440
        self.height = 2560
        self.size = pow(2, precision)

        self.colors = gen_colors(len(self.categories))

        self.csv_category_to_contiguous_id = {
            v: i + self.size for i, v in enumerate(self.categories)
        }

        self.contiguous_category_id_to_csv = {
            v: k for k, v in self.csv_category_to_contiguous_id.items()
        }

        self.vocab_size = self.size + len(self.categories) + 2  # bos, eos, pad tokens
        # self.bos_token = self.vocab_size - 2
        self.eos_token = self.vocab_size - 2
        self.pad_token = self.vocab_size - 1

        self.data = df.filename.unique()
        self.max_item = max_item
        self.max_length = max_length
        if self.max_length is None:
            # self.max_length = min(df.groupby('filename').count().xmin.max(),self.max_item)*5 + 2  # bos, eos tokens
            self.max_length = self.max_item*5 + 1  # bos, eos tokens
        self.transform = Padding(self.max_length, self.vocab_size)

    def quantize_box(self, boxes, width, height):

        # range of xy is [0, large_side-1]
        # range of wh is [1, large_side]
        # bring xywh to [0, 1]
        boxes = np.array(boxes, dtype=float)
        boxes[:, [2, 3]] = boxes[:, [2, 3]] - 1
        boxes[:, [0, 2]] /= (width - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / (height - 1)
        boxes = np.clip(boxes, 0, 1)

        # next take xywh to [0, size-1]
        boxes = (boxes * (self.size - 1)).round()

        return boxes.astype(np.int32)

    def __len__(self):
        return len(self.data)

    def render(self, layout):
        img = Image.new('RGB', (self.width, self.height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        layout = layout.reshape(-1)
        layout = trim_tokens(layout,  self.eos_token, self.pad_token)
        layout = layout[: len(layout) // 5 * 5].reshape(-1, 5)
        box = layout[:, 1:].astype(np.float32)
        box[:, [0, 2]] = box[:, [0, 2]] / (self.size - 1) * (self.width-1)
        box[:, [1, 3]] = box[:, [1, 3]] / self.size * self.height
        box[:, [2, 3]] = box[:, [0, 1]] + box[:, [2, 3]]

        for i in range(len(layout)):
            x1, y1, x2, y2 = box[i]
            cat = layout[i][0]
            if  0 <= cat-self.size < len(self.colors) :
                # col = self.colors[cat-self.size] if 0 <= cat-self.size < len(self.colors) else [0, 0, 0]
                col = self.colors[cat-self.size] 
                draw.rectangle([x1, y1, x2, y2],
                           outline=tuple(col) + (200,),
                           fill=tuple(col) + (64,),
                           width=2)

                font_path = 'font/Lantinghei.ttc'
                font = ImageFont.truetype(font_path, 40)
                draw.text((x1+5,y1+5),self.contiguous_category_id_to_csv[cat], tuple(col),font)
        # Add border around image
        img = ImageOps.expand(img, border=2)
        return img

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        filename = self.data[idx]
        # print(filename)
        image = self.df.query(f"filename == '{filename}'")
        ann_box = []
        ann_cat = []
        if len(image) > self.max_item:
            image = image[:self.max_item]
        for ann in range(len(image)):
            ann = image.iloc[ann]
            x, y, w, h = ann['xmin'], ann['ymin'], ann['width'], ann['height']
            ann_box.append([x, y, w, h])
            ann_cat.append(self.csv_category_to_contiguous_id[ann['category']])

        # Sort boxes
        ann_box = np.array(ann_box,dtype=float)
        ind = np.lexsort((ann_box[:, 0], ann_box[:, 1]))
        # ind = np.argsort(ann_box[:, 2] * ann_box[:, 3], )[::-1]
        ann_box = ann_box[ind]
    
        ann_cat = np.array(ann_cat)
        ann_cat = ann_cat[ind]

        # Discretize boxes
        ann_box = self.quantize_box(ann_box, self.width, self.height)

        # Append the categories
        layout = np.concatenate([ann_cat.reshape(-1, 1), ann_box], axis=1)
        layout = torch.tensor(layout, dtype=torch.long)
        # print(layout)
        x_num = random.randint(1,len(image))
        # x_num = max(len(image) -1, 1)
        slice = random.sample(list(range(len(image))), x_num)
        layout_x = torch.zeros(self.max_length, dtype=torch.long) + self.pad_token
        layout_x[:x_num*5] = layout[slice].view(-1)
        # layout_x[x_num*5] = self.eos_token
        layout_y = torch.zeros(5, dtype=torch.long) + self.pad_token
        if x_num == len(image):
            layout_y[0] = self.eos_token
        else:
            for i in range(len(image)):
                if i not in slice:
                    layout_y = layout[i]
                    break

        # layout = self.transform(layout)
        layout_all = torch.zeros(self.max_length, dtype=torch.long) + self.pad_token
        layout_all[:x_num*5] = layout[slice].view(-1)
        ids = [False if i in slice else True for i in range(len(image))]
        layout = layout[ids]
        
        if x_num*5 < len(image)*5 :
            layout_all[x_num*5:len(image)*5] = layout.view(-1)
        layout_all[len(image)*5] = self.eos_token
        # print(layout_x,layout_all)
        return layout_x, layout_y, layout_all

if __name__ == '__main__':
    layout_all = CSVLayout('testfile5.csv')
    sample_xy = layout_all[0]
    layout_all.render(np.array(sample_xy[0])).show()
    # layout_all.render(np.array(sample_xy[1])).show()
    layout_all.render(np.array(sample_xy[2])).show()
    print(sample_xy)
