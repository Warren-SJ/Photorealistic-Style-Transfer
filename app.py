"""
Copyright (c) 2019 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# Modifications by Warren Jayakumar, 2025

import os

import torch
from torchvision.utils import save_image

from model import WaveEncoder, WaveDecoder

from utils.core import feature_wct
from utils.io import open_image

import gradio as gr
from PIL import Image
import numpy as np


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class WCT2:
    def __init__(self, transfer_at=['encoder', 'skip', 'decoder'], device='cuda:0', verbose=False):

        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'models')
        self.device = device
        self.transfer_at = set(transfer_at)
        assert not(self.transfer_at - set(['encoder', 'decoder', 'skip'])), 'invalid transfer_at: {}'.format(transfer_at)
        assert self.transfer_at, 'empty transfer_at'

        self.device = torch.device(device)
        self.verbose = verbose
        self.encoder = WaveEncoder('cat5').to(self.device)
        self.decoder = WaveDecoder('cat5').to(self.device)
        encoder_ckpt = torch.load(os.path.join(model_path, 'wave_encoder.pth'), map_location=self.device)
        for k, v in encoder_ckpt.items():
            encoder_ckpt[k] = v.clone() if isinstance(v, torch.Tensor) else v
        self.encoder.load_state_dict(encoder_ckpt)
        decoder_ckpt = torch.load(os.path.join(model_path, 'wave_decoder.pth'), map_location=self.device)
        for k, v in decoder_ckpt.items():
            decoder_ckpt[k] = v.clone() if isinstance(v, torch.Tensor) else v
        self.decoder.load_state_dict(decoder_ckpt)

    def print_(self, msg):
        if self.verbose:
            print(msg)

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)

    def get_all_feature(self, x):
        skips = {}
        feats = {'encoder': {}, 'decoder': {}}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            if 'encoder' in self.transfer_at:
                feats['encoder'][level] = x

        if 'encoder' not in self.transfer_at:
            feats['decoder'][4] = x
        for level in [4, 3, 2]:
            x = self.decode(x, skips, level)
            if 'decoder' in self.transfer_at:
                feats['decoder'][level - 1] = x
        return feats, skips

    def transfer(self, content, style, alpha=0.5):
        content_feat, content_skips = content, {}
        style_feats, style_skips = self.get_all_feature(style)

        wct2_enc_level = [1, 2, 3, 4]
        wct2_dec_level = [1, 2, 3, 4]
        wct2_skip_level = ['pool1', 'pool2', 'pool3']

        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            if 'encoder' in self.transfer_at and level in wct2_enc_level:
                content_feat = feature_wct(content_feat, style_feats['encoder'][level],
                                           alpha=alpha, device=self.device).clone()
                self.print_('transfer at encoder {}'.format(level))
        if 'skip' in self.transfer_at:
            for skip_level in wct2_skip_level:
                new_skip = []
                for component in [0, 1, 2]:
                    new_component = feature_wct(
                        content_skips[skip_level][component].clone(),
                        style_skips[skip_level][component].clone(),
                        alpha=alpha, device=self.device
                    ).clone()
                    new_skip.append(new_component)
                content_skips[skip_level] = new_skip

                self.print_('transfer at skip {}'.format(skip_level))

        for level in [4, 3, 2, 1]:
            if 'decoder' in self.transfer_at and level in style_feats['decoder'] and level in wct2_dec_level:
                content_feat = feature_wct(content_feat, style_feats['decoder'][level],
                                           alpha=alpha, device=self.device).clone()
                self.print_('transfer at decoder {}'.format(level))
            content_feat = self.decode(content_feat, content_skips, level)
        return content_feat


def get_all_transfer():
    ret = []
    for e in ['encoder', None]:
        for d in ['decoder', None]:
            for s in ['skip', None]:
                _ret = set([e, d, s]) & set(['encoder', 'decoder', 'skip'])
                if _ret:
                    ret.append(_ret)
    return ret

def stylize(content_path, style_path, output_path):
    device = "cpu"
    device = torch.device(device)
    content = open_image(content_path).to(device)
    style   = open_image(style_path).to(device)
    wct2 = WCT2(device=device, verbose=False)
    with torch.no_grad():
        out = wct2.transfer(content, style, alpha=0.5)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image(out.clamp_(0,1), output_path, padding=0)
    print(f"Stylized image saved to {output_path}")


def stylize(content_img, style_img, alpha=0.5):
    device =  "cpu"
    device = torch.device(device)
    if content_img is None or style_img is None:
        raise ValueError("Both content and style images must be provided.")
    content_img = Image.fromarray(np.array(content_img).astype(np.uint8))
    style_img = Image.fromarray(np.array(style_img).astype(np.uint8))
    content = open_image(content_img).to(device)
    style = open_image(style_img).to(device)
    wct2 = WCT2(device=device, verbose=False)
    with torch.no_grad():
        out = wct2.transfer(content, style, alpha=alpha)
    out_img = out.clamp_(0,1).cpu().squeeze(0)
    out_img = out_img.permute(1,2,0).numpy()
    out_img = (out_img * 255).astype(np.uint8)
    return Image.fromarray(out_img)


def main():
    demo = gr.Interface(
        fn=stylize,
        inputs=[
            gr.Image(type="pil", label="Content Image", elem_id="content_img"),
            gr.Image(type="pil", label="Style Image", elem_id="style_img"),
            gr.Slider(0, 1, value=0.5, step=0.01, label="Alpha (Style Strength)")
        ],
        outputs=gr.Image(type="pil", label="Stylized Output"),
        title="Photorealistic Style Transfer",
        description="Upload a content and a style image to perform photorealistic style transfer. Adjust the alpha slider to control style strength.",
        allow_flagging="never",
    )
    demo.launch()

if __name__ == '__main__':
   main()
