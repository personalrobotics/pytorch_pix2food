#!/usr/bin/env python3

import os
import sys
import numpy as np

from PIL import Image, ImageDraw, ImageFont

import rospkg

rospack = rospkg.RosPack()
biteselection_base = rospack.get_path('bite_selection_package')

def draw_point(draw, point, psize, fill, width):
    draw.line(
        [point[0] - psize, point[1], point[0] + psize, point[1]],
        fill=fill, width=width)
    draw.line(
        [point[0], point[1] - psize, point[0], point[1] + psize],
        fill=fill, width=width)


def draw_axis(draw, p1, p2, fill=(0, 0, 200, 150), width=4,
              psize=5, pfill=(255, 0, 0, 200), pwidth=3):
    draw.line(p1 + p2, fill=fill, width=width)
    draw_point(draw, p1, psize=psize, fill=pfill, width=pwidth)
    draw_point(draw, p2, psize=psize, fill=pfill, width=pwidth)
    cp = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
    draw_point(draw, cp, psize=psize + 3, fill=pfill, width=pwidth)


def draw_vectors(draw, pred):
    draw_axis(
        draw, pred[:2], pred[2:],
        fill=(200, 0, 0, 150), pfill=(255, 0, 0, 200))


def draw_scores(draw, img_size, margin, pred):
    fnt_title = ImageFont.truetype(
        os.path.join(biteselection_base,
            'resource/Roboto-Regular.ttf'), 15)
    fnt = ImageFont.truetype(
        os.path.join(biteselection_base,
            'resource/Roboto-Regular.ttf'), 20)
    lr_margin = 5
    tb_margin = 3
    cell_offset = (img_size[0] - lr_margin * 2) / (len(pred) + 1)
    vertical_offset = (margin - tb_margin * 2) / 3.0

    title_top = img_size[1] + tb_margin
    pred_top = img_size[1] + vertical_offset
    gt_top = img_size[1] + vertical_offset * 2

    titles = ['', 'v0', 'v90', 'tv0', 'tv90', 'ta0', 'ta90', 'scoop']
    pred_strs = ['pred'] + list(map(str, np.around(pred,2)))
    for idx in range(len(titles)):
        col_x = lr_margin + cell_offset * idx
        draw.text(
            (col_x, title_top),
            titles[idx], font=fnt_title, fill=(15, 15, 15, 255))
        if idx > 0:
            draw.rectangle(
                (col_x - 7, pred_top, col_x + cell_offset - 10, pred_top + 23),
                fill=(250, 0, 0, int(float(pred_strs[idx]) * 150)),
                outline=None)

        draw.text(
            (col_x, pred_top),
            pred_strs[idx], font=fnt, fill=(0, 0, 0, 255))


def draw_image(img_org, pred_vector):
    new_size = 600
    img_org = img_org.resize((new_size, new_size), resample=Image.BILINEAR)
    img_size = img_org.size

    margin = 80
    img = Image.new('RGBA', (img_size[0], img_size[1] + margin),
                    color=(255, 255, 255))
    img.paste(img_org)

    img_draw = ImageDraw.Draw(img)
    pred_points = tuple(np.asarray(pred_vector[:4]) * new_size)
    draw_vectors(img_draw, pred_points)

    pred_scores = pred_vector[4:]
    draw_scores(img_draw, img_size, margin, pred_scores)

    img_noalpha = Image.new("RGB", img.size, (255, 255, 255))
    img_noalpha.paste(img, mask=img.split()[3])

    del img_draw
    return img_noalpha
