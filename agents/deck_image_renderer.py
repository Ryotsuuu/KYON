"""
agents/deck_image_renderer.py
=============================
Visual Deck Image Renderer for Pokémon TCG Decks.
Based on the official competition deck renderer notebook.

Features:
- Extracts card artwork directly from official PDF files (if available via PyMuPDF)
- High-resolution fallback card canvas rendering with full type colors, HP, attacks, and stages
- Translucent count overlays (e.g. `×4 ID:716`) with black-background grid compositing
- Progressive JPEG compression optimizing output size under 1MB
- Saves directly to the project root directory
"""
import os
import math
import numpy as np
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image, ImageDraw, ImageFont

from agents.csv_data import get_csv_index

TYPE_COLORS = {
    'Grass': ((46, 139, 87), (30, 90, 55)),
    'Fire': ((220, 70, 40), (150, 45, 25)),
    'Water': ((30, 144, 255), (20, 95, 170)),
    'Lightning': ((230, 190, 20), (160, 130, 15)),
    'Psychic': ((153, 50, 204), (105, 35, 140)),
    'Fighting': ((184, 115, 51), (130, 80, 35)),
    'Darkness': ((47, 79, 79), (30, 50, 50)),
    'Metal': ((140, 140, 160), (95, 95, 110)),
    'Colorless': ((180, 180, 185), (120, 120, 125)),
    'Dragon': ((180, 150, 40), (125, 100, 25)),
}


def ordered_deck_counts(deck_ids: List[int]) -> Tuple[List[int], Counter]:
    """Count Card IDs while preserving first appearance order."""
    counts = Counter(deck_ids)
    ordered_ids = []
    seen = set()
    for card_id in deck_ids:
        if card_id not in seen:
            ordered_ids.append(card_id)
            seen.add(card_id)
    return ordered_ids, counts


def get_system_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Get a crisp TrueType font across Windows/Linux with fallback."""
    font_candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "seguiui.ttf",
        "tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for fp in font_candidates:
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def resize_keep_aspect(img: Image.Image, target_width: int) -> Image.Image:
    """Resize an image preserving aspect ratio."""
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    target_width = int(target_width)
    scale = target_width / max(1, img.width)
    target_height = max(1, int(round(img.height * scale)))
    return img.resize((target_width, target_height), resampling)


def text_bbox(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font: ImageFont.ImageFont) -> Tuple[int, int, int, int]:
    """Compatibility wrapper for text bounding box calculation."""
    if hasattr(draw, "textbbox"):
        return draw.textbbox(xy, text, font=font)
    try:
        w, h = draw.textsize(text, font=font)
        x, y = xy
        return (x, y, x + w, y + h)
    except Exception:
        return (xy[0], xy[1], xy[0] + len(text) * 8, xy[1] + 16)


def crop_card_from_page_image(page_img: Image.Image, expected_card_aspect: float = 0.714) -> Image.Image:
    """Crop the card artwork out of an official PDF page using darkness density thresholds."""
    img = page_img.convert("RGB")
    arr = np.asarray(img)
    h, w = arr.shape[:2]

    x0 = int(w * 0.05)
    x1 = int(w * 0.95)
    y0 = int(h * 0.02)
    y1 = int(h * 0.72)

    roi = arr[y0:y1, x0:x1, :]
    darkness = np.max(255 - roi.astype(np.int16), axis=2)
    mask = darkness > 18

    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)

    if row_counts.max() <= 0 or col_counts.max() <= 0:
        left = int(w * 0.36)
        right = int(w * 0.64)
        top = int(h * 0.07)
        bottom = int(h * 0.62)
    else:
        row_threshold = max(10, int(row_counts.max() * 0.055))
        col_threshold = max(10, int(col_counts.max() * 0.055))

        ys = np.where(row_counts > row_threshold)[0]
        xs = np.where(col_counts > col_threshold)[0]

        if len(xs) < 10 or len(ys) < 10:
            left = int(w * 0.36)
            right = int(w * 0.64)
            top = int(h * 0.07)
            bottom = int(h * 0.62)
        else:
            left = x0 + int(xs.min())
            right = x0 + int(xs.max()) + 1
            top = y0 + int(ys.min())
            bottom = y0 + int(ys.max()) + 1

            pad = max(8, int(max(w, h) * 0.004))
            left = max(0, left - pad)
            right = min(w, right + pad)
            top = max(0, top - pad)
            bottom = min(h, bottom + pad)

    box_w = right - left
    box_h = bottom - top
    if box_w <= 0 or box_h <= 0:
        return img

    current_aspect = box_w / box_h
    center_x = (left + right) / 2

    if current_aspect > expected_card_aspect * 1.18:
        target_w = int(round(box_h * expected_card_aspect))
        left = int(round(center_x - target_w / 2))
        right = left + target_w
    elif current_aspect < expected_card_aspect * 0.82:
        target_w = int(round(box_h * expected_card_aspect))
        left = int(round(center_x - target_w / 2))
        right = left + target_w

    if left < 0:
        right -= left
        left = 0
    if right > w:
        left = max(0, left - (right - w))
        right = w

    return img.crop((left, top, right, bottom))


def generate_highres_card_art(card_data, card_id: int, width: int = 320, height: int = 448) -> Image.Image:
    """Generate a crisp, styled Pokémon card canvas with type gradient borders."""
    card_img = Image.new("RGBA", (width, height), (24, 28, 36, 255))
    draw = ImageDraw.Draw(card_img)

    c_type = getattr(card_data, 'type', 'Colorless') if card_data else 'Colorless'
    colors = TYPE_COLORS.get(c_type, TYPE_COLORS['Colorless'])
    theme_col, dark_col = colors[0], colors[1]

    # Outer card border
    draw.rounded_rectangle([0, 0, width, height], radius=14, fill=(20, 22, 28), outline=theme_col, width=4)

    # Top Header Bar
    draw.rounded_rectangle([6, 6, width - 6, 54], radius=8, fill=dark_col)
    
    font_name = get_system_font(18, bold=True)
    font_hp = get_system_font(16, bold=True)
    font_sub = get_system_font(12, bold=False)
    font_atk = get_system_font(14, bold=True)

    name = card_data.name if card_data else f"Card #{card_id}"
    draw.text((16, 14), name[:18], fill=(255, 255, 255), font=font_name)

    hp_str = f"{card_data.hp} HP" if card_data and card_data.hp else ""
    if hp_str:
        draw.text((width - 80, 16), hp_str, fill=(255, 220, 100), font=font_hp)

    # Center Art Box
    draw.rounded_rectangle([14, 62, width - 14, 230], radius=8, fill=(10, 14, 20), outline=(50, 56, 68), width=2)

    # Category / Stage ribbon
    stage_str = "BASIC"
    if card_data:
        if card_data.is_trainer:
            stage_str = card_data.category.upper()
        elif card_data.is_basic_energy:
            stage_str = "BASIC ENERGY"
        elif card_data.is_special_energy:
            stage_str = "SPECIAL ENERGY"
        elif getattr(card_data, 'pokemon_stage', 0) == 1:
            stage_str = "STAGE 1"
        elif getattr(card_data, 'pokemon_stage', 0) == 2:
            stage_str = "STAGE 2"

    draw.rounded_rectangle([width // 2 - 60, 70, width // 2 + 60, 92], radius=6, fill=theme_col)
    draw.text((width // 2 - 40, 74), stage_str, fill=(255, 255, 255), font=font_sub)

    # Middle artwork watermark icon
    draw.text((width // 2 - 35, 140), f"#{card_id}", fill=(60, 68, 80), font=get_system_font(28, bold=True))

    # Attacks & text info
    y_atk = 244
    if card_data and getattr(card_data, 'attacks', None):
        for atk in card_data.attacks[:2]:
            draw.text((20, y_atk), f"⚔ {atk.name[:16]}", fill=(240, 240, 240), font=font_atk)
            if atk.damage:
                draw.text((width - 60, y_atk), f"{atk.damage}", fill=(255, 100, 100), font=font_atk)
            y_atk += 32
    else:
        desc = card_data.category if card_data else "Game Card"
        draw.text((20, 260), desc, fill=(180, 190, 205), font=font_sub)

    return card_img


def make_labeled_card_tile(
    card_img: Image.Image,
    card_id: int,
    count: int,
    card_width: int = 320,
    label_alpha: int = 200
) -> Image.Image:
    """Create one card tile with a translucent centered label at the bottom."""
    card = resize_keep_aspect(card_img, card_width).convert("RGBA")
    label_h = max(48, int(card.height * 0.155))

    overlay = Image.new("RGBA", card.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [0, card.height - label_h, card.width, card.height],
        fill=(0, 0, 0, int(label_alpha)),
    )

    tile = Image.alpha_composite(card, overlay)
    draw = ImageDraw.Draw(tile)

    text = f"×{count}  ID:{card_id}"
    font_size = max(24, int(label_h * 0.62))
    font = get_system_font(font_size, bold=True)

    max_text_width = int(card.width * 0.94)
    while font_size > 10:
        bbox = text_bbox(draw, (0, 0), text, font)
        if (bbox[2] - bbox[0]) <= max_text_width:
            break
        font_size -= 1
        font = get_system_font(font_size, bold=True)

    bbox = text_bbox(draw, (0, 0), text, font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (card.width - text_w) // 2
    y = card.height - label_h + (label_h - text_h) // 2 - 1

    try:
        draw.text(
            (x, y),
            text,
            fill=(255, 255, 255, 255),
            font=font,
            stroke_width=max(1, int(font_size * 0.04)),
            stroke_fill=(0, 0, 0, 240),
        )
    except TypeError:
        draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)

    return tile


def render_deck_grid(
    ordered_ids: List[int],
    counts: Counter,
    card_images: Dict[int, Image.Image],
    columns: int = 8,
    card_width: int = 320,
    gap: int = 8,
    padding: int = 0,
    label_alpha: int = 200,
) -> Image.Image:
    """Render all unique deck cards into one black-background image."""
    tiles = []
    for card_id in ordered_ids:
        if card_id not in card_images:
            continue
        tile = make_labeled_card_tile(
            card_images[card_id],
            card_id=card_id,
            count=counts[card_id],
            card_width=card_width,
            label_alpha=label_alpha,
        )
        tiles.append(tile)

    if not tiles:
        raise ValueError("No cards available to render.")

    columns = max(1, int(columns))
    rows = math.ceil(len(tiles) / columns)

    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)

    width = padding * 2 + columns * tile_w + (columns - 1) * gap
    height = padding * 2 + rows * tile_h + (rows - 1) * gap

    canvas = Image.new("RGB", (width, height), (10, 12, 16))

    for idx, tile in enumerate(tiles):
        row = idx // columns
        col = idx % columns
        cell_x = padding + col * (tile_w + gap)
        cell_y = padding + row * (tile_h + gap)

        x = cell_x + (tile_w - tile.width) // 2
        y = cell_y + (tile_h - tile.height) // 2

        canvas.paste(tile.convert("RGB"), (x, y))

    return canvas


def save_jpeg_under_size(
    image: Image.Image,
    output_path: Path,
    max_bytes: int = 1_000_000,
    start_quality: int = 90,
    min_quality: int = 45,
    downscale_step: float = 0.92,
    min_width: int = 1200,
) -> Tuple[Image.Image, int, int]:
    """Save image as JPEG optimized under target file size."""
    output_path = Path(output_path)
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    working = image.convert("RGB")

    while True:
        for quality in range(start_quality, min_quality - 1, -5):
            working.save(
                output_path,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
                subsampling=2,
            )
            size_bytes = output_path.stat().st_size
            if size_bytes <= max_bytes:
                return working, quality, size_bytes

        if working.width <= min_width:
            working.save(
                output_path,
                format="JPEG",
                quality=min_quality,
                optimize=True,
                progressive=True,
                subsampling=2,
            )
            return working, min_quality, output_path.stat().st_size

        new_width = max(min_width, int(working.width * downscale_step))
        scale = new_width / working.width
        new_height = max(1, int(round(working.height * scale)))
        working = working.resize((new_width, new_height), resampling)


def render_deck_to_image(
    deck: List[int],
    agent_name: str = "PTCG_Deck",
    output_path: Optional[Path] = None,
    csv_index=None
) -> Path:
    """Render a 60-card deck list into a visual grid image matching the official visualizer."""
    idx = csv_index or get_csv_index(Path("data"))
    ordered_ids, counts = ordered_deck_counts(deck)

    # Output path setup: default to root folder (main folder)
    if output_path is None:
        output_path = Path(f"{agent_name}_deck.jpg")
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if PDF files exist for official card artwork extraction
    pdf_candidates = [
        Path("data") / "Card_ID List_EN.pdf",
        Path("Card_ID List_EN.pdf"),
        Path("data") / "Card_ID List_JP.pdf",
    ]
    card_pdf = next((p for p in pdf_candidates if p.exists()), None)

    card_images = {}
    if card_pdf is not None:
        try:
            import fitz
            with fitz.open(card_pdf) as doc:
                for card_id in ordered_ids:
                    # PDF page 40 corresponds to ID 1 (0-indexed page 39)
                    page_index = max(0, min(len(doc) - 1, 39 + card_id - 1))
                    page = doc.load_page(page_index)
                    pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), alpha=False)
                    page_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    card_images[card_id] = crop_card_from_page_image(page_img)
        except Exception:
            card_images = {}

    # High-resolution generated card artwork fallback
    for card_id in ordered_ids:
        if card_id not in card_images:
            card_data = idx.get_card(card_id) if idx else None
            card_images[card_id] = generate_highres_card_art(card_data, card_id)

    # Render 8-column grid
    deck_canvas = render_deck_grid(
        ordered_ids=ordered_ids,
        counts=counts,
        card_images=card_images,
        columns=8,
        card_width=320,
        gap=8,
        padding=10,
        label_alpha=200,
    )

    # Save progressive compressed JPEG
    save_jpeg_under_size(deck_canvas, output_path=output_path, max_bytes=1_000_000)

    # Also save deck_image.jpg directly in root directory
    root_deck_jpg = Path("deck_image.jpg")
    save_jpeg_under_size(deck_canvas, output_path=root_deck_jpg, max_bytes=1_000_000)

    return output_path
