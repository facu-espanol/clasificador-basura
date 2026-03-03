import random, shutil
from pathlib import Path

src_root = Path("TrashType_Image_Dataset")
dst_root = Path("dataset_split")

train_ratio, val_ratio, test_ratio = 0.80, 0.10, 0.10
seed = 42
move_files = True 

random.seed(seed)
classes = [d for d in src_root.iterdir() if d.is_dir()]

def list_images(folder: Path):
    exts = {".jpg",".jpeg",".png",".webp",".bmp"}
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]

# Crear estructura
for split in ["train", "val", "test"]:
    for c in classes:
        (dst_root / split / c.name).mkdir(parents=True, exist_ok=True)

total_before = 0
total_after = 0

for c in classes:
    imgs = list_images(c)
    total_before += len(imgs)
    if not imgs:
        print(f"[WARN] clase vacía: {c.name}")
        continue

    random.shuffle(imgs)
    n = len(imgs)

    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val  # asegura suma exacta

    splits = {
        "train": imgs[:n_train],
        "val":   imgs[n_train:n_train+n_val],
        "test":  imgs[n_train+n_val:],
    }

    for split, files in splits.items():
        for p in files:
            dst = dst_root / split / c.name / p.name
            shutil.move(str(p), str(dst))

    print(f"{c.name:10s} total={n:4d}  train={len(splits['train']):4d}  val={len(splits['val']):4d}  test={len(splits['test']):4d}")

for p in dst_root.rglob("*"):
    if p.is_file():
        total_after += 1

print(f"\nTotal antes: {total_before}")
print(f"Total después: {total_after}")
print("OK" if total_before == total_after else "OJO: no coincide el total")
print("Ruta:", dst_root.resolve())