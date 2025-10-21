"""
synthetic_generator.py
Generates synthetic images with chess piece sprites composited on board backgrounds.
Produces YOLO-format .txt labels for each image.

Usage:
    python synthetic_generator.py --out dataset --n_images 3000 --img_size 640 --min_pieces 6 --max_pieces 16
"""

import os
import random
import argparse
from PIL import Image, ImageEnhance, ImageFilter
import math

# Classes order must match data.yaml
CLASSES = [
 "w_king","w_queen","w_rook","w_bishop","w_knight","w_pawn",
 "b_king","b_queen","b_rook","b_bishop","b_knight","b_pawn"
]

def load_sprites(sprite_dir):
    sprites = {}
    for cls in CLASSES:
        p = os.path.join(sprite_dir, f"{cls}.png")
        if os.path.isfile(p):
            sprites[cls] = Image.open(p).convert("RGBA")
        else:
            raise FileNotFoundError(f"Missing sprite: {p}")
    return sprites

def random_transform(sprite, max_rotate=10, scale_range=(0.4, 1.0)):
    # rotate + scale + optional blur
    scale = random.uniform(*scale_range)
    w = max(8, int(sprite.width * scale))
    h = max(8, int(sprite.height * scale))
    s = sprite.resize((w,h), Image.LANCZOS)
    angle = random.uniform(-max_rotate, max_rotate)
    s = s.rotate(angle, expand=True)
    # optional slight blur
    if random.random() < 0.05:
        s = s.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5,1.5)))
    return s

def place_pieces_on_board(bg, sprites, min_pieces, max_pieces, allow_overlap=False):
    img_w, img_h = bg.size
    n_pieces = random.randint(min_pieces, max_pieces)
    labels = []
    canvas = bg.copy().convert("RGBA")

    # Simple occupancy grid to reduce heavy overlap
    occupancy = []

    attempts = 0
    for _ in range(n_pieces):
        if attempts > n_pieces * 50:
            break
        attempts += 1
        cls_name = random.choice(list(sprites.keys()))
        sprite = sprites[cls_name]
        s = random_transform(sprite, max_rotate=8, scale_range=(0.35, 0.85))
        sw, sh = s.size
        # choose position
        x = random.randint(0, max(0, img_w - sw))
        y = random.randint(0, max(0, img_h - sh))
        box = (x, y, x + sw, y + sh)
        # check overlap
        good = True
        if not allow_overlap:
            for ox in occupancy:
                # IoU-ish simple overlap check
                inter_w = max(0, min(box[2], ox[2]) - max(box[0], ox[0]))
                inter_h = max(0, min(box[3], ox[3]) - max(box[1], ox[1]))
                inter_area = inter_w * inter_h
                if inter_area > 0:
                    # disallow too big overlap
                    if inter_area > 0.35 * (sw * sh):
                        good = False
                        break
        if not good:
            continue
        # composite
        canvas.alpha_composite(s, (x, y))
        occupancy.append(box)
        # optionally add small shadow
        if random.random() < 0.6:
            shadow = Image.new("RGBA", (sw, sh), (0,0,0,0))
            # slight offset and low alpha ellipse as shadow
            shad = Image.new("RGBA", (sw, sh), (0,0,0,0))
            # draw simple shadow using transformed sprite mask
            mask = s.split()[-1].point(lambda p: int(p*0.25))
            canvas.paste((0,0,0,80), (x+int(sw*0.06), y+int(sh*0.06)), mask)

        # compute normalized bbox center,w,h
        xc = (x + sw/2) / img_w
        yc = (y + sh/2) / img_h
        w_norm = sw / img_w
        h_norm = sh / img_h
        cls_idx = CLASSES.index(cls_name)
        labels.append((cls_idx, xc, yc, w_norm, h_norm))

    # final color adjustments to simulate varied lighting
    canvas_rgb = canvas.convert("RGB")
    if random.random() < 0.5:
        enh = ImageEnhance.Brightness(canvas_rgb)
        canvas_rgb = enh.enhance(random.uniform(0.8, 1.15))
    if random.random() < 0.5:
        enh = ImageEnhance.Contrast(canvas_rgb)
        canvas_rgb = enh.enhance(random.uniform(0.9, 1.2))
    return canvas_rgb, labels

def main(args):
    random.seed(args.seed)
    os.makedirs(os.path.join(args.out, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "labels", "val"), exist_ok=True)

    sprites = load_sprites(args.sprite_dir)
    boards = []
    for f in os.listdir(args.board_dir):
        path = os.path.join(args.board_dir, f)
        if os.path.isfile(path):
            boards.append(Image.open(path).convert("RGBA"))
    if not boards:
        raise FileNotFoundError("No board images found in board_dir")

    n_val = max(1, int(args.n_images * args.val_ratio))
    n_train = args.n_images - n_val

    def gen_and_save(idx, subset):
        bg = random.choice(boards).resize((args.img_size, args.img_size), Image.LANCZOS)
        img, labels = place_pieces_on_board(bg, sprites, args.min_pieces, args.max_pieces, allow_overlap=args.allow_overlap)
        name = f"{subset}_{idx:06d}.jpg"
        img_path = os.path.join(args.out, "images", subset, name)
        label_path = os.path.join(args.out, "labels", subset, name.replace(".jpg", ".txt"))
        img.save(img_path, quality=90)
        with open(label_path, "w") as f:
            for c, xc, yc, w, h in labels:
                f.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    print(f"Generating {n_train} train and {n_val} val images to {args.out}")
    for i in range(n_train):
        gen_and_save(i, "train")
    for i in range(n_val):
        gen_and_save(i, "val")
    print("Done!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sprite_dir", default="sprites", help="folder with 12 piece pngs named like w_king.png")
    p.add_argument("--board_dir", default="boards", help="folder with board background images")
    p.add_argument("--out", default="dataset", help="output dataset folder")
    p.add_argument("--n_images", type=int, default=3000)
    p.add_argument("--img_size", type=int, default=640)
    p.add_argument("--min_pieces", type=int, default=6)
    p.add_argument("--max_pieces", type=int, default=16)
    p.add_argument("--val_ratio", type=float, default=0.12)
    p.add_argument("--allow_overlap", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
