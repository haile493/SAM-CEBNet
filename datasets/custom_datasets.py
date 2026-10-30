"""
Custom Dataset Loaders for Remote Sensing Building Detection, Bi-Temporal Change Detection,
Building Damage Assessment, and Bi-Temporal Change Captioning.

Supports:
1. `SingleImageBuildingDataset` (for `ImprovedMaskRCNNBuilding` [2022] and `CBRNet` [2026])
   Directory layout:
       <data_dir>/
         images/   (*.png, *.jpg, *.tif, *.bmp)
         masks/    (*.png, *.jpg, *.tif, *.bmp)  [optional in inference mode]

2. `BiTemporalChangeDataset` (for `BiSAMCD` [2025], `SAM2CD` [2025], `GMFIMamba` [2026],
   `SemiBCD` [2026], `CGCCENet` [2026], and `EdgeRefNet` [2026])
   Directory layout:
       <data_dir>/
         A/ (or t1/, pre/)       (*.png, *.jpg, *.tif, *.bmp)
         B/ (or t2/, post/)      (*.png, *.jpg, *.tif, *.bmp)
         label/ (or masks/, gt/) (*.png, *.jpg, *.tif, *.bmp) [optional in inference mode]
         vlm_pseudo/             [optional VLM pseudo-labels for SemiBCD]
         loc_label/              [optional pre-event building mask for GMFIMamba BDA]

3. `BiTemporalCaptionDataset` (for `KGBDCNet` [2026])
   Directory layout:
       <data_dir>/
         A/ (or t1/, pre/)
         B/ (or t2/, post/)
         captions.json           [{"id": "0001.png", "caption": "...", "keywords": ["collapsed", "roof"]}, ...]
"""

import json
import os
from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from ag_models.losses import CBRNetCompositeLoss, EdgeGuidedChangeLoss

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _list_image_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files: List[str] = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(VALID_EXTENSIONS):
            files.append(os.path.join(folder, fname))
    return files


def discover_hue_geotiff_tiles(data_dir: str) -> List[Dict[str, Optional[str]]]:
    """
    Discovers nested Hue GeoTIFF + GeoJSON tile folders (`dataset/image-hue_RR_CC/`).
    Each entry contains:
      - `id`: tile identifier (e.g. `'image-hue_01_01.png'`)
      - `stem`: tile stem (e.g. `'image-hue_01_01'`)
      - `tif_path`: path to `image-hue_RR_CC.tif`
      - `geojson_path`: path to `image-hue_RR_CC.geojson` (or `None` if tile has 0 buildings)
    """
    if not os.path.isdir(data_dir):
        return []
    tiles: List[Dict[str, Optional[str]]] = []
    for entry in sorted(os.listdir(data_dir)):
        sub = os.path.join(data_dir, entry)
        if not os.path.isdir(sub):
            continue
        tif_candidates = [
            f for f in sorted(os.listdir(sub)) if f.lower().endswith((".tif", ".tiff"))
        ]
        if not tif_candidates:
            continue
        tif_path = os.path.join(sub, tif_candidates[0])
        stem = os.path.splitext(tif_candidates[0])[0]
        gj_candidate = os.path.join(sub, stem + ".geojson")
        if not os.path.isfile(gj_candidate):
            gj_files = [f for f in sorted(os.listdir(sub)) if f.lower().endswith(".geojson")]
            gj_candidate = os.path.join(sub, gj_files[0]) if gj_files else None
        tiles.append(
            {
                "id": stem + ".png",
                "stem": stem,
                "tif_path": tif_path,
                "geojson_path": gj_candidate if (gj_candidate and os.path.isfile(gj_candidate)) else None,
            }
        )
    return tiles


def create_or_load_hue_splits(
    data_dir: str,
    strategy: str = "spatial_block",
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    save_manifest: bool = True,
) -> Dict[str, Any]:
    """
    Creates or loads a zero-leakage `train` / `val` / `unseen_test` split manifest
    (`<data_dir>/hue_splits.json`) for the Hue GeoTIFF+GeoJSON grid or flat custom datasets.

    Strategies:
    1. `'spatial_block'` (Default & Recommended for Remote Sensing):
       Partitions the 11x14 Hue geographic grid by East-West columns:
         - `train`:       Columns `01..10` (109 tiles, 72.7%, Western & Central Hue)
         - `val`:         Columns `11..12` (20 tiles, 13.3%, Buffer zone between train & unseen_test)
         - `unseen_test`: Columns `13..14` (21 tiles, 14.0%, Eastern Hue holdout — zero shared borders with `train`!)
    2. `'stratified'`:
       Sorts tiles by building footprint count from `.geojson` and assigns every 20-tile block
       proportionally (70% train / 15% val / 15% unseen_test) so building density distributions match.
    3. `'random'`:
       Deterministic seeded permutation (`seed=42`) split by `ratios`.
    """
    manifest_path = os.path.join(data_dir, "hue_splits.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("strategy") == strategy:
            return cached

    hue_tiles = discover_hue_geotiff_tiles(data_dir)
    if hue_tiles:
        stems = [str(t["stem"]) for t in hue_tiles]
    else:
        img_dir = (
            _resolve_subfolder(data_dir, ("images", "image", "img", "A", "t1", "T1", "pre"))
            or data_dir
        )
        stems = [os.path.splitext(os.path.basename(p))[0] for p in _list_image_files(img_dir)]

    train_stems: List[str] = []
    val_stems: List[str] = []
    unseen_stems: List[str] = []

    if strategy == "spatial_block" and hue_tiles and all("_" in s for s in stems):
        for s in stems:
            parts = s.split("_")
            col = int(parts[-1]) if parts[-1].isdigit() else 1
            if col <= 10:
                train_stems.append(s)
            elif col <= 12:
                val_stems.append(s)
            else:
                unseen_stems.append(s)
    elif strategy == "stratified" and hue_tiles:
        # Sort by building polygon count and stratify deterministically
        counts = []
        for t in hue_tiles:
            gj_p = t["geojson_path"]
            c = 0
            if gj_p and os.path.isfile(gj_p):
                with open(gj_p, "r", encoding="utf-8") as f:
                    c = len(json.load(f).get("features", []))
            counts.append((c, str(t["stem"])))
        counts.sort(key=lambda x: (x[0], x[1]))
        for idx, (_, s) in enumerate(counts):
            mod = idx % 20
            if mod in (19, 13, 6):
                unseen_stems.append(s)
            elif mod in (18, 12, 5):
                val_stems.append(s)
            else:
                train_stems.append(s)
        train_stems.sort()
        val_stems.sort()
        unseen_stems.sort()
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(stems)).tolist()
        n_total = len(stems)
        n_unseen = max(1, int(round(n_total * ratios[2]))) if n_total >= 3 else 0
        n_val = max(1, int(round(n_total * ratios[1]))) if n_total >= 3 else 0
        n_train = max(1, n_total - n_val - n_unseen)
        train_stems = sorted(stems[i] for i in perm[:n_train])
        val_stems = sorted(stems[i] for i in perm[n_train : n_train + n_val])
        unseen_stems = sorted(stems[i] for i in perm[n_train + n_val :])

    # Strict zero-leakage verification
    set_tr, set_va, set_te = set(train_stems), set(val_stems), set(unseen_stems)
    assert set_tr.isdisjoint(set_va), "Data leakage detected between train and val!"
    assert set_tr.isdisjoint(set_te), "Data leakage detected between train and unseen_test!"
    assert set_va.isdisjoint(set_te), "Data leakage detected between val and unseen_test!"

    payload: Dict[str, Any] = {
        "strategy": strategy,
        "seed": seed,
        "total_tiles": len(stems),
        "counts": {
            "train": len(train_stems),
            "val": len(val_stems),
            "unseen_test": len(unseen_stems),
        },
        "zero_leakage_verified": True,
        "splits": {
            "train": train_stems,
            "val": val_stems,
            "unseen_test": unseen_stems,
        },
    }
    if save_manifest and os.path.isdir(data_dir) and len(stems) > 0:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return payload


def _normalize_split_name(split: str) -> str:
    s = split.lower().strip()
    if s in ("test", "unseen", "unseen_test", "holdout"):
        return "unseen_test"
    if s in ("val", "valid", "validation"):
        return "val"
    if s in ("train", "training"):
        return "train"
    return "all"


_MERCATOR_ORIGIN_SHIFT = 20037508.342789244


def _project_pt_to_tiepoint_crs(pt: Sequence[float], x0: float) -> Tuple[float, float]:
    """
    Projects a GeoJSON vertex `(x, y)` to EPSG:3857 meters if the vertex is stored in
    WGS84 geographic degrees (`EPSG:4326`, `|lon| <= 180`, `|lat| <= 90`) while the
    GeoTIFF tiepoint `x0` is in EPSG:3857 meters (`|x0| > 1000`).
    """
    x, y = float(pt[0]), float(pt[1])
    if abs(x) <= 180.0 and abs(y) <= 90.0 and abs(x0) > 1000.0:
        import math

        mx = x * _MERCATOR_ORIGIN_SHIFT / 180.0
        my = math.log(math.tan((90.0 + y) * math.pi / 360.0)) * (_MERCATOR_ORIGIN_SHIFT / math.pi)
        return mx, my
    return x, y


def rasterize_hue_geojson(
    tif_path: str,
    geojson_path: Optional[str],
    out_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, List[List[Tuple[float, float]]]]:
    """
    Reads GeoTIFF tiepoint (tag 33922) and pixel scale (tag 33550) in EPSG:3857 and
    rasterizes all building footprint polygons (including interior courtyard holes, and
    auto-projecting both EPSG:3857 and EPSG:4326 GeoJSON coordinates) from `geojson_path`
    into a binary uint8 mask (0 or 255) aligned with the GeoTIFF.

    Also returns the list of exterior pixel polygon rings in resized `(col, row)` coordinates.
    """
    with Image.open(tif_path) as im:
        orig_w, orig_h = im.size
        tags = getattr(im, "tag_v2", {}) or {}
        scale = tags.get(33550, (1.0, 1.0, 0.0))
        tiepoint = tags.get(33922, (0.0, 0.0, 0.0, 0.0, float(orig_h), 0.0))

    sx, sy = float(scale[0]), float(scale[1])
    x0, y0 = float(tiepoint[3]), float(tiepoint[4])

    target_h, target_w = out_size if out_size is not None else (orig_h, orig_w)
    scale_x_res = float(target_w) / max(float(orig_w), 1.0)
    scale_y_res = float(target_h) / max(float(orig_h), 1.0)

    mask_im = Image.new("L", (target_w, target_h), 0)
    draw = ImageDraw.Draw(mask_im)
    pixel_polygons: List[List[Tuple[float, float]]] = []

    if geojson_path is not None and os.path.isfile(geojson_path):
        with open(geojson_path, "r", encoding="utf-8") as f:
            gj = json.load(f)
        for feat in gj.get("features", []):
            geom = feat.get("geometry") or {}
            gtype = geom.get("type")
            coords = geom.get("coordinates") or []
            polys = [coords] if gtype == "Polygon" else (coords if gtype == "MultiPolygon" else [])
            for poly in polys:
                if not poly:
                    continue
                ext = []
                for pt in poly[0]:
                    mx, my = _project_pt_to_tiepoint_crs(pt, x0)
                    ext.append((((mx - x0) / sx) * scale_x_res, ((y0 - my) / sy) * scale_y_res))
                if len(ext) >= 3:
                    draw.polygon(ext, fill=255)
                    pixel_polygons.append(ext)
                for hole in poly[1:]:
                    h_pts = []
                    for pt in hole:
                        mx, my = _project_pt_to_tiepoint_crs(pt, x0)
                        h_pts.append((((mx - x0) / sx) * scale_x_res, ((y0 - my) / sy) * scale_y_res))
                    if len(h_pts) >= 3:
                        draw.polygon(h_pts, fill=0)

    return np.asarray(mask_im, dtype=np.uint8), pixel_polygons


def synthesize_bitemporal_from_hue_tile(
    img_rgb: torch.Tensor,
    pixel_polygons: List[List[Tuple[float, float]]],
    img_size: Tuple[int, int],
    stem: str,
    multi_class_damage: bool = False,
    single_image_mode: bool = False,
    full_building_mask_np: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Constructs a pair `(img_t1, img_t2, mask_tensor, loc_tensor, meta)` from a Hue GeoTIFF tile:
      - If `single_image_mode=True`:
          * Zero ground-truth information is used to construct `img_t1`.
          * `img_t2` is the raw single Hue RGB image (`img_rgb`).
          * `img_t1` is a purely image-derived multi-scale low-pass local background reference of `img_rgb`.
          * `mask_tensor` is the 100% all-building footprint mask (identical to `SingleImageBuildingDataset`).
      - If `single_image_mode=False`:
          * Constructs a polygon-level bi-temporal change pair where ~35% of polygons are masked in `img_t1`.
    """
    h, w = img_size
    img_t2 = img_rgb.clone()

    if single_image_mode:
        # Purely image-derived multi-scale low-pass background context (zero GT leakage)
        x_in = img_rgb.unsqueeze(0)
        lp_15 = F.avg_pool2d(F.pad(x_in, (7, 7, 7, 7), mode="reflect"), kernel_size=15, stride=1)
        lp_31 = F.avg_pool2d(F.pad(x_in, (15, 15, 15, 15), mode="reflect"), kernel_size=31, stride=1)
        img_t1 = (0.5 * lp_15 + 0.5 * lp_31).squeeze(0).clamp(0.0, 1.0)

        if full_building_mask_np is not None:
            bld_bin = (full_building_mask_np > 0).astype(np.float32)
        else:
            all_im = Image.new("L", (w, h), 0)
            draw_all = ImageDraw.Draw(all_im)
            for poly in pixel_polygons:
                draw_all.polygon(poly, fill=255)
            bld_bin = (np.asarray(all_im, dtype=np.uint8) > 0).astype(np.float32)

        if multi_class_damage:
            mask_tensor = torch.from_numpy(bld_bin.astype(np.int64)).long()
        else:
            mask_tensor = torch.from_numpy(bld_bin).unsqueeze(0)
        loc_tensor = torch.from_numpy(bld_bin).unsqueeze(0)
        meta = {
            "total_buildings": len(pixel_polygons),
            "changed_buildings": len(pixel_polygons),
            "intact_buildings": 0,
        }
        return img_t1, img_t2, mask_tensor, loc_tensor, meta

    img_t1 = img_rgb.clone()

    loc_im = Image.new("L", (w, h), 0)
    chg_im = Image.new("L", (w, h), 0)
    draw_loc = ImageDraw.Draw(loc_im)
    draw_chg = ImageDraw.Draw(chg_im)

    seed = sum(ord(c) * (i + 1) for i, c in enumerate(stem))
    rng = np.random.default_rng(seed)

    num_changed = 0
    num_intact = 0
    for idx, poly in enumerate(pixel_polygons):
        is_changed = (idx % 3 == 0)
        if is_changed:
            num_changed += 1
            if multi_class_damage:
                dmg_cls = int((idx % 3) + 2)  # 2: minor, 3: major, 4: destroyed
                draw_chg.polygon(poly, fill=dmg_cls)
                draw_loc.polygon(poly, fill=255)
            else:
                draw_chg.polygon(poly, fill=255)
        else:
            num_intact += 1
            draw_loc.polygon(poly, fill=255)

    chg_np = np.asarray(chg_im, dtype=np.uint8)
    loc_np = np.asarray(loc_im, dtype=np.uint8)

    # Compute non-building background color statistics from the real Hue tile
    all_bld_mask = (chg_np > 0) | (loc_np > 0)
    bg_pixels = img_rgb[:, ~all_bld_mask]
    if bg_pixels.numel() > 0:
        bg_mean = bg_pixels.mean(dim=1, keepdim=True).unsqueeze(-1)
        bg_std = bg_pixels.std(dim=1, keepdim=True).unsqueeze(-1).clamp(min=0.02, max=0.08)
    else:
        bg_mean = torch.tensor([0.28, 0.34, 0.25]).view(3, 1, 1)
        bg_std = torch.tensor([0.04, 0.04, 0.04]).view(3, 1, 1)

    chg_bool = torch.from_numpy(chg_np > 0)
    noise = torch.from_numpy(rng.normal(0.0, 1.0, size=(3, h, w)).astype(np.float32))
    bg_synth = (bg_mean + noise * bg_std).clamp(0.0, 1.0)
    # Smooth synthetic pre-construction ground texture
    bg_synth = F.avg_pool2d(bg_synth.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)

    for c in range(3):
        img_t1[c][chg_bool] = bg_synth[c][chg_bool]

    if multi_class_damage:
        # Include intact buildings as class 1 in BDA mask
        bda_np = np.zeros((h, w), dtype=np.int64)
        bda_np[loc_np > 0] = 1
        bda_np[chg_np > 0] = chg_np[chg_np > 0].astype(np.int64)
        mask_tensor = torch.from_numpy(bda_np).long()
    else:
        mask_tensor = torch.from_numpy((chg_np > 0).astype(np.float32)).unsqueeze(0)

    loc_tensor = torch.from_numpy((loc_np > 0).astype(np.float32)).unsqueeze(0)
    meta = {
        "total_buildings": len(pixel_polygons),
        "changed_buildings": num_changed,
        "intact_buildings": num_intact,
    }
    return img_t1, img_t2, mask_tensor, loc_tensor, meta


def _load_rgb_tensor(path: str, img_size: Tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as img:
        img_rgb = img.convert("RGB").resize((img_size[1], img_size[0]), Image.Resampling.BILINEAR)
        arr = np.asarray(img_rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


def _load_mask_tensor(path: str, img_size: Tuple[int, int], multi_class: bool = False) -> torch.Tensor:
    with Image.open(path) as img:
        img_l = img.convert("L").resize((img_size[1], img_size[0]), Image.Resampling.NEAREST)
        arr = np.asarray(img_l, dtype=np.float32)
    if multi_class:
        if arr.max() > 10:
            arr = (arr >= 128).astype(np.int64)
        else:
            arr = arr.astype(np.int64)
        return torch.from_numpy(arr).long()
    else:
        if arr.max() > 1.0:
            arr = (arr >= 128.0).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0).float()


def _resolve_subfolder(root: str, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        p = os.path.join(root, c)
        if os.path.isdir(p):
            return p
    return None


class SingleImageBuildingDataset(Dataset):
    """
    Custom dataset for single-image building extraction and instance/semantic detection
    (`ImprovedMaskRCNNBuilding` and `CBRNet`).

    Natively supports:
    1. Hue GeoTIFF + EPSG:3857 GeoJSON nested tile directories (`dataset/image-hue_RR_CC/`).
    2. Flat `images/` + `masks/` directory layouts.
    3. Zero-leakage `split` filtering (`'train'`, `'val'`, `'unseen_test'`, or `'all'`).
    """

    def __init__(
        self,
        data_dir: str,
        img_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
        max_samples: Optional[int] = None,
        split: str = "all",
        split_strategy: str = "spatial_block",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment
        self.split = _normalize_split_name(split)
        self.split_strategy = split_strategy

        self.hue_tiles = discover_hue_geotiff_tiles(data_dir)
        self.image_paths: List[str] = []
        self.mask_dir: Optional[str] = None

        split_manifest = (
            create_or_load_hue_splits(data_dir, strategy=split_strategy)
            if self.split != "all"
            else None
        )
        allowed_stems = (
            set(split_manifest["splits"][self.split]) if split_manifest is not None else None
        )

        if self.hue_tiles:
            if allowed_stems is not None:
                self.hue_tiles = [t for t in self.hue_tiles if t["stem"] in allowed_stems]
            if max_samples is not None and max_samples > 0:
                self.hue_tiles = self.hue_tiles[:max_samples]
        else:
            img_dir = _resolve_subfolder(data_dir, ("images", "image", "img", "A", "t1")) or data_dir
            self.mask_dir = _resolve_subfolder(data_dir, ("masks", "mask", "label", "labels", "gt"))
            self.image_paths = _list_image_files(img_dir)
            if allowed_stems is not None:
                self.image_paths = [
                    p
                    for p in self.image_paths
                    if os.path.splitext(os.path.basename(p))[0] in allowed_stems
                ]
            if max_samples is not None and max_samples > 0:
                self.image_paths = self.image_paths[:max_samples]
            if not self.image_paths:
                raise FileNotFoundError(
                    f"No Hue GeoTIFF tiles or image files found in {data_dir} for split='{self.split}'"
                )

        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, str, bool]] = {}

    def __len__(self) -> int:
        return len(self.hue_tiles) if self.hue_tiles else len(self.image_paths)

    def _find_matching_mask(self, img_path: str) -> Optional[str]:
        if self.mask_dir is None:
            return None
        stem = os.path.splitext(os.path.basename(img_path))[0]
        for ext in VALID_EXTENSIONS:
            candidate = os.path.join(self.mask_dir, stem + ext)
            if os.path.isfile(candidate):
                return candidate
        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx in self._cache:
            base_image, base_mask, sample_id, has_label = self._cache[idx]
            image = base_image.clone()
            mask = base_mask.clone()
        else:
            if self.hue_tiles:
                tile = self.hue_tiles[idx]
                tif_path = tile["tif_path"]
                gj_path = tile["geojson_path"]
                assert tif_path is not None
                image = _load_rgb_tensor(tif_path, self.img_size)
                mask_np, _ = rasterize_hue_geojson(tif_path, gj_path, out_size=self.img_size)
                mask = torch.from_numpy((mask_np > 0).astype(np.float32)).unsqueeze(0)
                sample_id = str(tile["id"])
                has_label = True
            else:
                img_path = self.image_paths[idx]
                image = _load_rgb_tensor(img_path, self.img_size)
                mask_path = self._find_matching_mask(img_path)
                if mask_path is not None:
                    mask = _load_mask_tensor(mask_path, self.img_size, multi_class=False)
                else:
                    mask = torch.zeros((1, self.img_size[0], self.img_size[1]), dtype=torch.float32)
                sample_id = os.path.basename(img_path)
                has_label = mask_path is not None
            self._cache[idx] = (image.clone(), mask.clone(), sample_id, has_label)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[-1])
                mask = torch.flip(mask, dims=[-1])
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[-2])
                mask = torch.flip(mask, dims=[-2])
            if self.img_size[0] == self.img_size[1]:
                k = int(torch.randint(0, 4, (1,)).item())
                if k > 0:
                    image = torch.rot90(image, k, dims=[-2, -1])
                    mask = torch.rot90(mask, k, dims=[-2, -1])

        bnd_gt, cor_gt = CBRNetCompositeLoss.extract_boundary_and_corner_gt(mask.unsqueeze(0))
        return {
            "id": sample_id,
            "image": image,
            "mask": mask,
            "boundary_gt": bnd_gt.squeeze(0),
            "corner_gt": cor_gt.squeeze(0),
            "has_label": has_label,
        }


class BiTemporalChangeDataset(Dataset):
    """
    Custom dataset for bi-temporal change detection (`BiSAMCD`, `SAM2CD`, `GMFIMamba`,
    `SemiBCD`, `CGCCENet`, `EdgeRefNet`).

    Natively supports:
    1. Standard bi-temporal folders (`A/`, `B/`, `label/`, optional `vlm_pseudo/` and `loc_label/`).
    2. Nested Hue GeoTIFF + EPSG:3857 GeoJSON tiles (`dataset/image-hue_RR_CC/`), constructing
       polygon-level paired bi-temporal views directly from the Hue building footprints.
    3. Zero-leakage `split` filtering (`'train'`, `'val'`, `'unseen_test'`, or `'all'`).
    """

    def __init__(
        self,
        data_dir: str,
        img_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
        multi_class_damage: bool = False,
        return_strong_aug: bool = False,
        max_samples: Optional[int] = None,
        split: str = "all",
        split_strategy: str = "spatial_block",
        single_image_mode: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment
        self.multi_class_damage = multi_class_damage
        self.return_strong_aug = return_strong_aug
        self.split = _normalize_split_name(split)
        self.split_strategy = split_strategy
        self.single_image_mode = single_image_mode

        dir_a = _resolve_subfolder(data_dir, ("A", "t1", "T1", "pre", "before", "images_A"))
        dir_b = _resolve_subfolder(data_dir, ("B", "t2", "T2", "post", "after", "images_B"))

        self.hue_tiles: List[Dict[str, Optional[str]]] = []
        self.paths_a: List[str] = []
        self.dir_a = dir_a
        self.dir_b = dir_b
        self.dir_label = _resolve_subfolder(data_dir, ("label", "labels", "mask", "masks", "gt", "OUT"))
        self.dir_vlm = _resolve_subfolder(data_dir, ("vlm_pseudo", "pseudo", "vlm_labels"))
        self.dir_loc = _resolve_subfolder(data_dir, ("loc_label", "building_pre", "loc"))

        split_manifest = (
            create_or_load_hue_splits(data_dir, strategy=split_strategy)
            if self.split != "all"
            else None
        )
        allowed_stems = (
            set(split_manifest["splits"][self.split]) if split_manifest is not None else None
        )

        if dir_a is not None and dir_b is not None:
            self.paths_a = _list_image_files(dir_a)
            if allowed_stems is not None:
                self.paths_a = [
                    p
                    for p in self.paths_a
                    if os.path.splitext(os.path.basename(p))[0] in allowed_stems
                ]
            if max_samples is not None and max_samples > 0:
                self.paths_a = self.paths_a[:max_samples]
            if not self.paths_a:
                raise FileNotFoundError(f"No images found in {dir_a} for split='{self.split}'")
        else:
            self.hue_tiles = discover_hue_geotiff_tiles(data_dir)
            if allowed_stems is not None:
                self.hue_tiles = [t for t in self.hue_tiles if t["stem"] in allowed_stems]
            if max_samples is not None and max_samples > 0:
                self.hue_tiles = self.hue_tiles[:max_samples]
            if not self.hue_tiles:
                raise FileNotFoundError(
                    f"BiTemporalChangeDataset expects either 'A/' & 'B/' folders or Hue GeoTIFF tiles inside {data_dir}"
                )

        self._cache: Dict[
            int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, bool]
        ] = {}

    def __len__(self) -> int:
        return len(self.hue_tiles) if self.hue_tiles else len(self.paths_a)

    @staticmethod
    def _find_by_stem(folder: Optional[str], stem: str) -> Optional[str]:
        if folder is None:
            return None
        for ext in VALID_EXTENSIONS:
            p = os.path.join(folder, stem + ext)
            if os.path.isfile(p):
                return p
        return None

    @staticmethod
    def _strong_photometric_aug(img: torch.Tensor) -> torch.Tensor:
        scale = 0.8 + 0.4 * torch.rand(1, 1, 1)
        shift = -0.1 + 0.2 * torch.rand(1, 1, 1)
        out = (img * scale + shift).clamp(0.0, 1.0)
        if torch.rand(1).item() > 0.5:
            out = F.avg_pool2d(out.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx in self._cache:
            b_a, b_b, b_mask, b_vlm, b_loc, sample_id, has_label = self._cache[idx]
            img_a = b_a.clone()
            img_b = b_b.clone()
            mask = b_mask.clone()
            vlm_mask = b_vlm.clone()
            loc_mask = b_loc.clone()
        else:
            if self.hue_tiles:
                tile = self.hue_tiles[idx]
                tif_path = tile["tif_path"]
                gj_path = tile["geojson_path"]
                stem = tile["stem"] or f"tile_{idx:03d}"
                assert tif_path is not None
                img_rgb = _load_rgb_tensor(tif_path, self.img_size)
                mask_np, polys = rasterize_hue_geojson(tif_path, gj_path, out_size=self.img_size)
                img_a, img_b, mask, loc_mask, _ = synthesize_bitemporal_from_tile_or_hue(
                    img_rgb,
                    polys,
                    self.img_size,
                    stem,
                    multi_class_damage=self.multi_class_damage,
                    single_image_mode=self.single_image_mode,
                    full_building_mask_np=mask_np,
                )
                vlm_mask = mask.float() if not self.multi_class_damage else (mask > 1).unsqueeze(0).float()
                sample_id = str(tile["id"])
                has_label = True
            else:
                path_a = self.paths_a[idx]
                stem = os.path.splitext(os.path.basename(path_a))[0]
                path_b = self._find_by_stem(self.dir_b, stem)
                if path_b is None:
                    raise FileNotFoundError(f"Matching post-event image for '{stem}' not found in {self.dir_b}")

                img_a = _load_rgb_tensor(path_a, self.img_size)
                img_b = _load_rgb_tensor(path_b, self.img_size)

                label_path = self._find_by_stem(self.dir_label, stem)
                if label_path is not None:
                    mask = _load_mask_tensor(label_path, self.img_size, multi_class=self.multi_class_damage)
                else:
                    if self.multi_class_damage:
                        mask = torch.zeros((self.img_size[0], self.img_size[1]), dtype=torch.long)
                    else:
                        mask = torch.zeros((1, self.img_size[0], self.img_size[1]), dtype=torch.float32)

                vlm_path = self._find_by_stem(self.dir_vlm, stem)
                vlm_mask = (
                    _load_mask_tensor(vlm_path, self.img_size, multi_class=False)
                    if vlm_path is not None
                    else (mask.float() if not self.multi_class_damage else (mask > 0).unsqueeze(0).float())
                )

                loc_path = self._find_by_stem(self.dir_loc, stem)
                loc_mask = (
                    _load_mask_tensor(loc_path, self.img_size, multi_class=False)
                    if loc_path is not None
                    else (mask.float() if not self.multi_class_damage else (mask > 0).unsqueeze(0).float())
                )
                sample_id = os.path.basename(path_a)
                has_label = label_path is not None

            self._cache[idx] = (
                img_a.clone(),
                img_b.clone(),
                mask.clone(),
                vlm_mask.clone(),
                loc_mask.clone(),
                sample_id,
                has_label,
            )

        if self.augment:
            if torch.rand(1).item() > 0.5:
                img_a = torch.flip(img_a, dims=[-1])
                img_b = torch.flip(img_b, dims=[-1])
                mask = torch.flip(mask, dims=[-1])
                vlm_mask = torch.flip(vlm_mask, dims=[-1])
                loc_mask = torch.flip(loc_mask, dims=[-1])
            if torch.rand(1).item() > 0.5:
                img_a = torch.flip(img_a, dims=[-2])
                img_b = torch.flip(img_b, dims=[-2])
                mask = torch.flip(mask, dims=[-2])
                vlm_mask = torch.flip(vlm_mask, dims=[-2])
                loc_mask = torch.flip(loc_mask, dims=[-2])
            if self.img_size[0] == self.img_size[1]:
                k = int(torch.randint(0, 4, (1,)).item())
                if k > 0:
                    img_a = torch.rot90(img_a, k, dims=[-2, -1])
                    img_b = torch.rot90(img_b, k, dims=[-2, -1])
                    mask = torch.rot90(mask, k, dims=[-2, -1])
                    vlm_mask = torch.rot90(vlm_mask, k, dims=[-2, -1])
                    loc_mask = torch.rot90(loc_mask, k, dims=[-2, -1])

        edge_gt = EdgeGuidedChangeLoss.extract_edge_gt(
            (mask > 0).unsqueeze(0).unsqueeze(0).float() if mask.ndim == 2 else mask.unsqueeze(0)
        ).squeeze(0)

        item: Dict[str, Any] = {
            "id": sample_id,
            "img_t1": img_a,
            "img_t2": img_b,
            "mask": mask,
            "vlm_pseudo": vlm_mask,
            "loc_mask": loc_mask,
            "edge_gt": edge_gt,
            "boundary_gt": edge_gt,
            "has_label": has_label,
        }
        if self.return_strong_aug:
            item["img_t1_strong"] = self._strong_photometric_aug(img_a)
            item["img_t2_strong"] = self._strong_photometric_aug(img_b)
        return item


synthesize_bitemporal_from_tile_or_hue = synthesize_bitemporal_from_hue_tile


class SimpleWordTokenizer:
    """
    Lightweight, zero-dependency word-level tokenizer for `KGBDCNet` building damage captions
    and hard damage keywords.
    """

    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
    DEFAULT_DAMAGE_KEYWORDS = [
        "no",
        "damage",
        "minor",
        "major",
        "destroyed",
        "collapsed",
        "roof",
        "debris",
        "intact",
        "flooded",
        "burned",
        "buildings",
        "residential",
        "industrial",
        "road",
        "trees",
        "dense",
        "moderate",
        "sparse",
        "urban",
        "sector",
        "hue",
        "riverfront",
    ]

    def __init__(self, texts: Optional[Sequence[str]] = None, max_vocab_size: int = 1000):
        self.word2idx: Dict[str, int] = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        for kw in self.DEFAULT_DAMAGE_KEYWORDS:
            if kw not in self.word2idx:
                self.word2idx[kw] = len(self.word2idx)
        if texts:
            for t in texts:
                for token in self._tokenize(t):
                    if token not in self.word2idx and len(self.word2idx) < max_vocab_size:
                        self.word2idx[token] = len(self.word2idx)
        self.idx2word: Dict[int, str] = {i: w for w, i in self.word2idx.items()}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        cleaned = (
            text.lower()
            .replace(".", " . ")
            .replace(",", " , ")
            .replace(";", " ; ")
            .replace("-", " ")
        )
        return [w for w in cleaned.split() if w]

    @property
    def vocab_size(self) -> int:
        return max(len(self.word2idx), 64)

    def encode(self, text: str, max_len: int = 24) -> torch.Tensor:
        tokens = [self.word2idx["<bos>"]]
        for w in self._tokenize(text)[: max_len - 2]:
            tokens.append(self.word2idx.get(w, self.word2idx["<unk>"]))
        tokens.append(self.word2idx["<eos>"])
        while len(tokens) < max_len:
            tokens.append(self.word2idx["<pad>"])
        return torch.tensor(tokens[:max_len], dtype=torch.long)

    def encode_keywords(self, keywords: Sequence[str], num_keywords: int = 16) -> torch.Tensor:
        ids: List[int] = []
        for kw in keywords[:num_keywords]:
            ids.append(self.word2idx.get(kw.lower(), self.word2idx["<unk>"]))
        while len(ids) < num_keywords:
            ids.append(self.word2idx.get(self.DEFAULT_DAMAGE_KEYWORDS[len(ids) % len(self.DEFAULT_DAMAGE_KEYWORDS)], 0))
        return torch.tensor(ids[:num_keywords], dtype=torch.long)

    def decode(self, token_ids: Sequence[int]) -> str:
        words: List[str] = []
        for idx in token_ids:
            idx_int = int(idx)
            if idx_int == self.word2idx["<eos>"]:
                break
            if idx_int in (self.word2idx["<pad>"], self.word2idx["<bos>"]):
                continue
            words.append(self.idx2word.get(idx_int, "<unk>"))
        return " ".join(words)


class BiTemporalCaptionDataset(Dataset):
    """
    Custom dataset for bi-temporal building damage captioning (`KGBDCNet`).
    Natively supports:
    1. Standard `A/`, `B/`, and optional `captions.json` folders.
    2. Nested Hue GeoTIFF + EPSG:3857 GeoJSON tiles (`dataset/image-hue_RR_CC/`), generating
       polygon-count-aware captions, keywords, and spatial change masks directly from each Hue tile.
    """

    def __init__(
        self,
        data_dir: str,
        img_size: Tuple[int, int] = (256, 256),
        max_seq_len: int = 24,
        num_keywords: int = 16,
        tokenizer: Optional[SimpleWordTokenizer] = None,
        max_samples: Optional[int] = None,
        split: str = "all",
        split_strategy: str = "spatial_block",
        single_image_mode: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.max_seq_len = max_seq_len
        self.num_keywords = num_keywords
        self.split = _normalize_split_name(split)
        self.split_strategy = split_strategy
        self.single_image_mode = single_image_mode

        dir_a = _resolve_subfolder(data_dir, ("A", "t1", "T1", "pre", "before"))
        dir_b = _resolve_subfolder(data_dir, ("B", "t2", "T2", "post", "after"))
        self.dir_a = dir_a
        self.dir_b = dir_b
        self.dir_label = _resolve_subfolder(data_dir, ("label", "labels", "mask", "masks", "gt", "OUT"))
        self.hue_tiles: List[Dict[str, Optional[str]]] = []
        self.paths_a: List[str] = []

        split_manifest = (
            create_or_load_hue_splits(data_dir, strategy=split_strategy)
            if self.split != "all"
            else None
        )
        allowed_stems = (
            set(split_manifest["splits"][self.split]) if split_manifest is not None else None
        )
        train_only_stems = (
            set(split_manifest["splits"]["train"]) if split_manifest is not None else None
        )

        if dir_a is not None and dir_b is not None:
            self.paths_a = _list_image_files(dir_a)
            if allowed_stems is not None:
                self.paths_a = [
                    p
                    for p in self.paths_a
                    if os.path.splitext(os.path.basename(p))[0] in allowed_stems
                ]
            if max_samples is not None and max_samples > 0:
                self.paths_a = self.paths_a[:max_samples]
        else:
            self.hue_tiles = discover_hue_geotiff_tiles(data_dir)
            if allowed_stems is not None:
                self.hue_tiles = [t for t in self.hue_tiles if t["stem"] in allowed_stems]
            if max_samples is not None and max_samples > 0:
                self.hue_tiles = self.hue_tiles[:max_samples]
            if not self.hue_tiles:
                raise FileNotFoundError(
                    f"BiTemporalCaptionDataset expects 'A/' & 'B/' folders or Hue GeoTIFF tiles in {data_dir}"
                )

        caption_file = os.path.join(data_dir, "captions.json")
        self.annotations: Dict[str, Dict[str, Any]] = {}
        all_texts: List[str] = [
            "dense residential buildings in hue urban sector experienced major roof damage while surrounding buildings remained intact",
            "moderate residential buildings in hue urban sector experienced minor roof damage while surrounding buildings remained intact",
            "sparse residential buildings in hue riverfront sector experienced minor roof damage while road and trees remained intact",
            "no building damage observed in this hue riverfront sector with intact road and trees",
        ]
        if os.path.isfile(caption_file):
            with open(caption_file, "r", encoding="utf-8") as f:
                records = json.load(f)
            if isinstance(records, list):
                for rec in records:
                    fname = rec.get("id") or rec.get("file_name") or rec.get("image")
                    if fname:
                        stem = os.path.splitext(os.path.basename(fname))[0]
                        self.annotations[stem] = rec
                        if "caption" in rec and (train_only_stems is None or stem in train_only_stems):
                            all_texts.append(rec["caption"])
            elif isinstance(records, dict):
                for k, v in records.items():
                    stem = os.path.splitext(os.path.basename(k))[0]
                    if isinstance(v, str):
                        self.annotations[stem] = {"caption": v}
                        if train_only_stems is None or stem in train_only_stems:
                            all_texts.append(v)
                    elif isinstance(v, dict):
                        self.annotations[stem] = v
                        if "caption" in v and (train_only_stems is None or stem in train_only_stems):
                            all_texts.append(v["caption"])

        self.tokenizer = tokenizer or SimpleWordTokenizer(texts=all_texts)
        self._cache: Dict[
            int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, torch.Tensor, torch.Tensor, str, bool]
        ] = {}

    def __len__(self) -> int:
        return len(self.hue_tiles) if self.hue_tiles else len(self.paths_a)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx in self._cache:
            b_a, b_b, b_mask, caption_str, cap_tok, kw_ids, sample_id, has_label = self._cache[idx]
            img_a = b_a.clone()
            img_b = b_b.clone()
            mask = b_mask.clone()
            caption_tokens = cap_tok.clone()
            keyword_ids = kw_ids.clone()
        else:
            if self.hue_tiles:
                tile = self.hue_tiles[idx]
                tif_path = tile["tif_path"]
                gj_path = tile["geojson_path"]
                stem = tile["stem"] or f"tile_{idx:03d}"
                assert tif_path is not None
                img_rgb = _load_rgb_tensor(tif_path, self.img_size)
                mask_np, polys = rasterize_hue_geojson(tif_path, gj_path, out_size=self.img_size)
                img_a, img_b, mask, _, meta = synthesize_bitemporal_from_hue_tile(
                    img_rgb,
                    polys,
                    self.img_size,
                    stem,
                    multi_class_damage=False,
                    single_image_mode=self.single_image_mode,
                    full_building_mask_np=mask_np,
                )
                sample_id = str(tile["id"])
                has_label = True
                if stem in self.annotations:
                    ann = self.annotations[stem]
                elif meta["total_buildings"] == 0:
                    ann = {
                        "caption": "no building damage observed in this hue riverfront sector with intact road and trees",
                        "keywords": ["no", "damage", "intact", "road", "trees", "riverfront"],
                    }
                elif meta["total_buildings"] < 60:
                    ann = {
                        "caption": "sparse residential buildings in hue riverfront sector experienced minor roof damage while road and trees remained intact",
                        "keywords": ["sparse", "residential", "buildings", "minor", "roof", "damage", "intact"],
                    }
                elif meta["total_buildings"] < 220:
                    ann = {
                        "caption": "moderate residential buildings in hue urban sector experienced minor roof damage while surrounding buildings remained intact",
                        "keywords": ["moderate", "residential", "buildings", "minor", "roof", "damage", "intact"],
                    }
                else:
                    ann = {
                        "caption": "dense residential buildings in hue urban sector experienced major roof damage while surrounding buildings remained intact",
                        "keywords": ["dense", "residential", "buildings", "major", "roof", "damage", "intact"],
                    }
                if self.single_image_mode:
                    # In single-image mode, do not leak ground-truth density tier via input keywords
                    ann = dict(ann)
                    ann["keywords"] = SimpleWordTokenizer.DEFAULT_DAMAGE_KEYWORDS[: self.num_keywords]
            else:
                path_a = self.paths_a[idx]
                stem = os.path.splitext(os.path.basename(path_a))[0]
                path_b = BiTemporalChangeDataset._find_by_stem(self.dir_b, stem)
                if path_b is None:
                    raise FileNotFoundError(f"Matching post-event image for '{stem}' not found in {self.dir_b}")

                img_a = _load_rgb_tensor(path_a, self.img_size)
                img_b = _load_rgb_tensor(path_b, self.img_size)
                label_path = BiTemporalChangeDataset._find_by_stem(self.dir_label, stem)
                if label_path is not None:
                    mask = _load_mask_tensor(label_path, self.img_size, multi_class=False)
                else:
                    mask = torch.zeros((1, self.img_size[0], self.img_size[1]), dtype=torch.float32)
                sample_id = os.path.basename(path_a)
                has_label = label_path is not None
                ann = self.annotations.get(
                    stem,
                    {
                        "caption": "several buildings in the residential area suffered major roof damage and debris",
                        "keywords": ["buildings", "residential", "major", "roof", "damage", "debris"],
                    },
                )

            caption_str = ann.get("caption", "")
            keywords_list = ann.get("keywords", SimpleWordTokenizer.DEFAULT_DAMAGE_KEYWORDS[: self.num_keywords])

            caption_tokens = self.tokenizer.encode(caption_str, max_len=self.max_seq_len)
            keyword_ids = self.tokenizer.encode_keywords(keywords_list, num_keywords=self.num_keywords)
            self._cache[idx] = (
                img_a.clone(),
                img_b.clone(),
                mask.clone(),
                caption_str,
                caption_tokens.clone(),
                keyword_ids.clone(),
                sample_id,
                has_label,
            )

        edge_gt = EdgeGuidedChangeLoss.extract_edge_gt(mask.unsqueeze(0)).squeeze(0)
        return {
            "id": sample_id,
            "img_t1": img_a,
            "img_t2": img_b,
            "mask": mask,
            "boundary_gt": edge_gt,
            "has_label": has_label,
            "caption_text": caption_str,
            "caption_tokens": caption_tokens,
            "keyword_ids": keyword_ids,
        }


def generate_synthetic_custom_data(root_dir: str, num_samples: int = 6, img_size: int = 64) -> str:
    """
    Creates a complete sample custom dataset directory at `root_dir` containing:
    - `images/` and `masks/` (for single-image building extraction: `CBRNet`, `ImprovedMaskRCNNBuilding`)
    - `A/`, `B/`, `label/`, `vlm_pseudo/`, `loc_label/` (for bi-temporal CD/BDA: `BiSAMCD`, `SAM2CD`,
      `GMFIMamba`, `SemiBCD`, `CGCCENet`, `EdgeRefNet`)
    - `captions.json` (for bi-temporal building damage captioning: `KGBDCNet`)
    """
    dirs = ["images", "masks", "A", "B", "label", "vlm_pseudo", "loc_label"]
    for d in dirs:
        os.makedirs(os.path.join(root_dir, d), exist_ok=True)

    captions_payload = []
    rng = np.random.default_rng(42)

    for i in range(num_samples):
        fname = f"sample_{i:03d}.png"
        # Create pre-event image with synthetic rectangular buildings
        img_a = rng.integers(30, 90, size=(img_size, img_size, 3), dtype=np.uint8)
        img_b = img_a.copy()
        mask_single = np.zeros((img_size, img_size), dtype=np.uint8)
        mask_change = np.zeros((img_size, img_size), dtype=np.uint8)
        mask_loc = np.zeros((img_size, img_size), dtype=np.uint8)

        # Building 1 (unchanged in B, present in A and single-image)
        r1, c1 = img_size // 6, img_size // 6
        h1, w1 = img_size // 4, img_size // 4
        img_a[r1 : r1 + h1, c1 : c1 + w1] = [200, 190, 180]
        img_b[r1 : r1 + h1, c1 : c1 + w1] = [200, 190, 180]
        mask_single[r1 : r1 + h1, c1 : c1 + w1] = 255
        mask_loc[r1 : r1 + h1, c1 : c1 + w1] = 255

        # Building 2 (changed / newly constructed or damaged in B)
        r2, c2 = img_size // 2, img_size // 2
        h2, w2 = img_size // 3, img_size // 3
        img_b[r2 : r2 + h2, c2 : c2 + w2] = [230, 120, 100]
        mask_single[r2 : r2 + h2, c2 : c2 + w2] = 255
        mask_change[r2 : r2 + h2, c2 : c2 + w2] = 255

        Image.fromarray(img_b).save(os.path.join(root_dir, "images", fname))
        Image.fromarray(mask_single).save(os.path.join(root_dir, "masks", fname))
        Image.fromarray(img_a).save(os.path.join(root_dir, "A", fname))
        Image.fromarray(img_b).save(os.path.join(root_dir, "B", fname))
        Image.fromarray(mask_change).save(os.path.join(root_dir, "label", fname))
        Image.fromarray(mask_change).save(os.path.join(root_dir, "vlm_pseudo", fname))
        Image.fromarray(mask_loc).save(os.path.join(root_dir, "loc_label", fname))

        captions_payload.append(
            {
                "id": fname,
                "caption": "buildings in the southeast sector experienced major roof damage while northwest buildings remained intact",
                "keywords": ["buildings", "major", "roof", "damage", "intact"],
            }
        )

    with open(os.path.join(root_dir, "captions.json"), "w", encoding="utf-8") as f:
        json.dump(captions_payload, f, indent=2)

    return root_dir


def download_and_prepare_whu_pretrain_subset(
    cache_dir: str = "./dataset_whu_pretrain",
    target_whu_aerial: int = 160,
    target_east_asia_sat: int = 160,
    min_fg_ratio: float = 0.04,
) -> Dict[str, Any]:
    """
    Downloads and curates a balanced, building-dense Stage-1 pretraining dataset combining:
    1. WHU Aerial Building Dataset (`giswqs/WHU-Building-Dataset`, 0.3m 512x512 RGB + mask pairs,
       filtered for building foreground ratio >= `min_fg_ratio`).
    2. Wuhan University East Asian Satellite Building Dataset (`chloechia/loveda`, 0.3m-0.5m satellite
       imagery from Wuhan, Nanjing, and Changzhou where pixel class `2` = Building, cropped into 512x512
       quadrants with building foreground ratio >= `min_fg_ratio`).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import io
    import urllib.request

    aerial_img_dir = os.path.join(cache_dir, "whu_aerial", "images")
    aerial_msk_dir = os.path.join(cache_dir, "whu_aerial", "masks")
    sat_img_dir = os.path.join(cache_dir, "whu_east_asia_sat", "images")
    sat_msk_dir = os.path.join(cache_dir, "whu_east_asia_sat", "masks")
    for d in (aerial_img_dir, aerial_msk_dir, sat_img_dir, sat_msk_dir):
        os.makedirs(d, exist_ok=True)

    existing_aerial = _list_image_files(aerial_img_dir)
    existing_sat = _list_image_files(sat_img_dir)

    # 1. Curate WHU Aerial dense tiles if needed
    if len(existing_aerial) < target_whu_aerial:
        needed = target_whu_aerial - len(existing_aerial)
        # Deterministic stride across 1..5732 to sample diverse urban sectors
        candidate_ids = [((i * 19) % 5732) + 1 for i in range(1200)]
        seen = set()
        ordered_ids = [x for x in candidate_ids if not (x in seen or seen.add(x))]

        def _try_fetch_whu_aerial(idx_num: int) -> Optional[str]:
            fname = f"train_{idx_num:04d}.png"
            out_i = os.path.join(aerial_img_dir, fname)
            out_m = os.path.join(aerial_msk_dir, fname)
            if os.path.isfile(out_i) and os.path.isfile(out_m):
                return fname
            url_m = f"https://huggingface.co/datasets/giswqs/WHU-Building-Dataset/resolve/main/train/Mask/{fname}"
            try:
                raw_m = urllib.request.urlopen(url_m, timeout=10).read()
                m_np = np.asarray(Image.open(io.BytesIO(raw_m)).convert("L"), dtype=np.uint8)
                if float((m_np > 127).mean()) < min_fg_ratio:
                    return None
                url_i = f"https://huggingface.co/datasets/giswqs/WHU-Building-Dataset/resolve/main/train/Image/{fname}"
                raw_i = urllib.request.urlopen(url_i, timeout=12).read()
                im_rgb = Image.open(io.BytesIO(raw_i)).convert("RGB")
                im_rgb.save(out_i)
                Image.fromarray(((m_np > 127).astype(np.uint8) * 255)).save(out_m)
                return fname
            except Exception:
                return None

        saved_count = len(existing_aerial)
        batch_step = 64
        for start in range(0, len(ordered_ids), batch_step):
            if saved_count >= target_whu_aerial:
                break
            chunk = ordered_ids[start : start + batch_step]
            with ThreadPoolExecutor(max_workers=16) as ex:
                futs = [ex.submit(_try_fetch_whu_aerial, cid) for cid in chunk]
                for fut in as_completed(futs):
                    res = fut.result()
                    if res is not None:
                        saved_count += 1
                        if saved_count >= target_whu_aerial:
                            break

    # 2. Curate Wuhan East Asian Satellite (`LoveDA`) dense building quadrants if needed
    existing_sat = _list_image_files(sat_img_dir)
    if len(existing_sat) < target_east_asia_sat:
        candidate_sat_ids = [((i * 7) % 1366) for i in range(600)]
        seen_s = set()
        ordered_sat_ids = [x for x in candidate_sat_ids if not (x in seen_s or seen_s.add(x))]

        def _try_fetch_loveda_sat(idx_num: int) -> int:
            url_m = (
                f"https://huggingface.co/datasets/chloechia/loveda/resolve/main/"
                f"urban%3Arural%20train%20masks/{idx_num}.png"
            )
            try:
                raw_m = urllib.request.urlopen(url_m, timeout=10).read()
                m_np = np.asarray(Image.open(io.BytesIO(raw_m)), dtype=np.uint8)
                bld_mask = (m_np == 2).astype(np.uint8) * 255
                if float((bld_mask > 0).mean()) < (min_fg_ratio * 0.5):
                    return 0
                # Check which 512x512 quadrants have building ratio >= min_fg_ratio
                h, w = bld_mask.shape[:2]
                mh, mw = h // 2, w // 2
                quads = [
                    ("q0", (0, 0, mw, mh)),
                    ("q1", (mw, 0, w, mh)),
                    ("q2", (0, mh, mw, h)),
                    ("q3", (mw, mh, w, h)),
                ]
                valid_quads = []
                for qname, (x1, y1, x2, y2) in quads:
                    q_m = bld_mask[y1:y2, x1:x2]
                    if float((q_m > 0).mean()) >= min_fg_ratio:
                        valid_quads.append((qname, (x1, y1, x2, y2), q_m))
                if not valid_quads:
                    return 0

                url_i = (
                    f"https://huggingface.co/datasets/chloechia/loveda/resolve/main/"
                    f"urban%3Arural%20train%20images/{idx_num}.png"
                )
                raw_i = urllib.request.urlopen(url_i, timeout=15).read()
                im_rgb = Image.open(io.BytesIO(raw_i)).convert("RGB")
                saved_q = 0
                for qname, box, q_m in valid_quads:
                    fname = f"loveda_{idx_num:04d}_{qname}.png"
                    out_i = os.path.join(sat_img_dir, fname)
                    out_m = os.path.join(sat_msk_dir, fname)
                    im_rgb.crop(box).save(out_i)
                    Image.fromarray(q_m).save(out_m)
                    saved_q += 1
                return saved_q
            except Exception:
                return 0

        saved_sat_count = len(existing_sat)
        batch_step = 32
        for start in range(0, len(ordered_sat_ids), batch_step):
            if saved_sat_count >= target_east_asia_sat:
                break
            chunk = ordered_sat_ids[start : start + batch_step]
            with ThreadPoolExecutor(max_workers=12) as ex:
                futs = [ex.submit(_try_fetch_loveda_sat, sid) for sid in chunk]
                for fut in as_completed(futs):
                    saved_sat_count += fut.result()

    final_aerial = _list_image_files(aerial_img_dir)
    final_sat = _list_image_files(sat_img_dir)
    return {
        "cache_dir": os.path.abspath(cache_dir),
        "whu_aerial_tiles": len(final_aerial),
        "whu_east_asia_sat_tiles": len(final_sat),
    }


class MultiSourceBuildingPretrainDataset(Dataset):
    """
    Option 3 Stage-1 Multi-Source Pretraining Dataset combining:
    1. Curated WHU Aerial dense building tiles (`whu_aerial`, 512x512, with 0.3m->0.5m satellite PSF simulation).
    2. Curated Wuhan East Asian Satellite building tiles (`whu_east_asia_sat`, 512x512).
    3. Multi-scale native-resolution 512x512 quadrants extracted strictly from the 109 Hue `train` split tiles
       (zero overlap with `val` or `unseen_test`).
    """

    def __init__(
        self,
        hue_data_dir: str,
        whu_cache_dir: str = "./dataset_whu_pretrain",
        img_size: Tuple[int, int] = (256, 256),
        augment: bool = True,
        split_strategy: str = "spatial_block",
        target_whu_aerial: int = 160,
        target_east_asia_sat: int = 160,
    ):
        super().__init__()
        self.img_size = img_size
        self.augment = augment

        download_and_prepare_whu_pretrain_subset(
            cache_dir=whu_cache_dir,
            target_whu_aerial=target_whu_aerial,
            target_east_asia_sat=target_east_asia_sat,
        )

        self.samples: List[Tuple[torch.Tensor, torch.Tensor, str, str]] = []

        # 1. Load WHU Aerial tiles
        aerial_img_dir = os.path.join(whu_cache_dir, "whu_aerial", "images")
        aerial_msk_dir = os.path.join(whu_cache_dir, "whu_aerial", "masks")
        for p_img in _list_image_files(aerial_img_dir)[:target_whu_aerial]:
            fname = os.path.basename(p_img)
            p_msk = os.path.join(aerial_msk_dir, fname)
            if os.path.isfile(p_msk):
                img_t = _load_rgb_tensor(p_img, img_size)
                msk_t = _load_mask_tensor(p_msk, img_size, multi_class=False)
                self.samples.append((img_t, msk_t, f"whu_aerial_{fname}", "whu_aerial"))

        # 2. Load Wuhan East Asian Satellite tiles
        sat_img_dir = os.path.join(whu_cache_dir, "whu_east_asia_sat", "images")
        sat_msk_dir = os.path.join(whu_cache_dir, "whu_east_asia_sat", "masks")
        for p_img in _list_image_files(sat_img_dir)[:target_east_asia_sat]:
            fname = os.path.basename(p_img)
            p_msk = os.path.join(sat_msk_dir, fname)
            if os.path.isfile(p_msk):
                img_t = _load_rgb_tensor(p_img, img_size)
                msk_t = _load_mask_tensor(p_msk, img_size, multi_class=False)
                self.samples.append((img_t, msk_t, f"whu_sat_{fname}", "whu_sat"))

        # 3. Load strictly `train`-split Hue native quadrants (4 quadrants per 1024x1024 train tile)
        split_manifest = create_or_load_hue_splits(hue_data_dir, strategy=split_strategy)
        train_stems = set(split_manifest["splits"]["train"])
        hue_tiles = [t for t in discover_hue_geotiff_tiles(hue_data_dir) if t["stem"] in train_stems]

        th, tw = img_size
        full_size = (th * 2, tw * 2)  # 512x512 -> four 256x256 native-scale quadrants
        for tile in hue_tiles:
            tif_path = tile["tif_path"]
            gj_path = tile["geojson_path"]
            assert tif_path is not None
            img_full = _load_rgb_tensor(tif_path, full_size)
            msk_full_np, _ = rasterize_hue_geojson(tif_path, gj_path, out_size=full_size)
            msk_full = torch.from_numpy((msk_full_np > 0).astype(np.float32)).unsqueeze(0)

            for q_idx, (r0, c0) in enumerate(((0, 0), (0, tw), (th, 0), (th, tw))):
                q_img = img_full[:, r0 : r0 + th, c0 : c0 + tw].clone()
                q_msk = msk_full[:, r0 : r0 + th, c0 : c0 + tw].clone()
                if float(q_msk.mean().item()) >= 0.01:
                    self.samples.append((q_img, q_msk, f"hue_train_{tile['stem']}_q{q_idx}", "hue_patch"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_img, base_msk, sample_id, source_tag = self.samples[idx]
        image = base_img.clone()
        mask = base_msk.clone()

        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[-1])
                mask = torch.flip(mask, dims=[-1])
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[-2])
                mask = torch.flip(mask, dims=[-2])
            if self.img_size[0] == self.img_size[1]:
                k = int(torch.randint(0, 4, (1,)).item())
                if k > 0:
                    image = torch.rot90(image, k, dims=[-2, -1])
                    mask = torch.rot90(mask, k, dims=[-2, -1])

            # Satellite sensor simulation on 50% of WHU Aerial tiles (0.3m -> 0.5m PSF blur + contrast/haze shift)
            if source_tag == "whu_aerial" and torch.rand(1).item() > 0.5:
                h, w = image.shape[-2:]
                down = F.interpolate(
                    image.unsqueeze(0), size=(int(h * 0.65), int(w * 0.65)), mode="bilinear", align_corners=False
                )
                image = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
                gamma = 0.85 + 0.30 * float(torch.rand(1).item())
                image = image.clamp(1e-5, 1.0).pow(gamma)

        bnd_gt, cor_gt = CBRNetCompositeLoss.extract_boundary_and_corner_gt(mask.unsqueeze(0))
        return {
            "id": sample_id,
            "image": image,
            "mask": mask,
            "boundary_gt": bnd_gt.squeeze(0),
            "corner_gt": cor_gt.squeeze(0),
            "has_label": True,
        }

