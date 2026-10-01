#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_image_ocr.py - Macht Text in eingebetteten PDF-Bildern durchsuchbar (englische OCR).

Vorgehen pro Seite:
  1. Alle Bildplatzierungen ermitteln (PyMuPDF get_images / get_image_rects).
  2. Genau diesen Seitenbereich hochaufloesend rendern (so stimmen die Koordinaten
     auch bei skalierten/gedrehten Bildern und ueberlagernden Elementen).
  3. RapidOCR (onnxruntime, CPU, offline, kein PyTorch) erkennen lassen.
  4. Erkannten Text UNSICHTBAR (render_mode=3) an exakt die Fundstelle schreiben.
     Bereiche, in denen bereits nativer PDF-Text liegt, werden uebersprungen
     (keine Dubletten).
  5. Verifikation: Original-Woerter unveraendert vorhanden + Seiten sehen
     pixelgleich aus (unsichtbarer Text darf nichts am Erscheinungsbild aendern).

Komplett offline: Die OCR-Modelle sind im pip-Paket rapidocr_onnxruntime enthalten,
es wird nichts nachgeladen.

Abhaengigkeiten (pip):  pymupdf  rapidocr_onnxruntime  (bringt onnxruntime, numpy, opencv mit)

Version : 1.1.0
Stand   : 2026-10-01 19:50
Release : OCR-Engine von EasyOCR auf RapidOCR/onnxruntime umgestellt, da die
          Windows-Anwendungssteuerung die PyTorch-DLLs blockiert (WinError 4551).
          Modellordner/--models/--allow-download/--try-rotations entfallen
          (180-Grad-Erkennung macht RapidOCR selbst). import pymupdf statt fitz.

Historie:
  1.0.0  2026-10-01  Erstversion (EasyOCR), Batch, Logging, Verifikation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (neuer Modulname, fitz ist veraltet)
import numpy as np

__version__ = "1.1.0"

log = logging.getLogger("pdf_image_ocr")


# --------------------------------------------------------------------------- #
# Konfiguration & Statistik
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    dpi: int = 300                 # Renderaufloesung fuer OCR
    min_conf: float = 0.50         # Mindest-Konfidenz eines OCR-Treffers
    min_image_pt: float = 20.0     # Bilder kleiner als das (in pt) ignorieren (Icons, Logos)
    overlap_skip: float = 0.5      # Anteil Ueberdeckung mit nativem Text -> OCR-Wort verwerfen
    verify: bool = True
    verify_dpi: int = 50


@dataclass
class FileStats:
    input: str
    output: str = ""
    pages: int = 0
    image_regions: int = 0
    ocr_items_added: int = 0
    ocr_items_low_conf: int = 0
    ocr_items_overlap_native: int = 0
    verify_words_ok: bool | None = None
    verify_visual_ok: bool | None = None
    seconds: float = 0.0
    error: str = ""
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# OCR-Engine (gekapselt, damit sie austauschbar/testbar bleibt)
# --------------------------------------------------------------------------- #
class EnglishOcr:
    """Adapter fuer RapidOCR. Unterstuetzt das Paket rapidocr_onnxruntime (1.x)
    und - falls stattdessen installiert - das neuere Paket rapidocr (2.x+)."""

    def __init__(self):
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
            self._api = "v1"
        except ImportError:
            from rapidocr import RapidOCR  # type: ignore
            self._api = "v2"
        self._engine = RapidOCR()
        log.info("OCR-Engine: RapidOCR (%s), onnxruntime CPU", self._api)

    def read(self, img_rgb: np.ndarray):
        """Liefert Liste von (box[4 Punkte in px], text, conf)."""
        img = np.ascontiguousarray(img_rgb[:, :, ::-1])  # RGB -> BGR (OpenCV-Konvention)
        if self._api == "v1":
            result, _ = self._engine(img)
            return [(r[0], r[1], float(r[2])) for r in (result or [])]
        out = self._engine(img)
        boxes = getattr(out, "boxes", None)
        if boxes is None:
            return []
        return [(b, t, float(c)) for b, t, c in zip(boxes, out.txts, out.scores)]


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def pixmap_to_array(pix: "fitz.Pixmap") -> np.ndarray:
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        arr = arr[:, :, :3]
    elif pix.n == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.ascontiguousarray(arr)


def overlap_ratio(a: fitz.Rect, b: fitz.Rect) -> float:
    """Anteil der Flaeche von a, der von b ueberdeckt wird."""
    inter = fitz.Rect(a) & b
    if inter.is_empty or a.is_empty:
        return 0.0
    return inter.get_area() / a.get_area()


def image_regions(page: fitz.Page, min_pt: float) -> list[fitz.Rect]:
    """Alle (deduplizierten) Bildplatzierungen auf der Seite, auf die Seite beschnitten."""
    seen, regions = set(), []
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception as exc:  # defekte/ungewoehnliche Bildreferenzen
            log.debug("get_image_rects fehlgeschlagen fuer xref %s: %s", xref, exc)
            continue
        for r in rects:
            r = fitz.Rect(r) & page.rect
            if r.is_empty or r.width < min_pt or r.height < min_pt:
                continue
            key = tuple(round(v, 1) for v in r)
            if key not in seen:
                seen.add(key)
                regions.append(r)
    return regions


def insert_invisible(page: fitz.Page, rect: fitz.Rect, text: str) -> None:
    """Text unsichtbar so einfuegen, dass er die Fundstelle moeglichst genau abdeckt
    (wichtig fuer korrekte Treffer-Markierung im PDF-Viewer)."""
    unit_len = fitz.get_text_length(text, fontname="helv", fontsize=1)
    if unit_len <= 0:
        return
    fs = min(rect.height * 0.95, rect.width / unit_len)
    if fs < 1:
        fs = 1
    baseline = fitz.Point(rect.x0, rect.y1 - rect.height * 0.18)
    page.insert_text(baseline, text, fontsize=fs, fontname="helv", render_mode=3)


# --------------------------------------------------------------------------- #
# Kernverarbeitung
# --------------------------------------------------------------------------- #
def process_page(page: fitz.Page, ocr: EnglishOcr, cfg: Config, st: FileStats) -> None:
    rotation = page.rotation
    if rotation:
        page.set_rotation(0)  # in ungedrehten Koordinaten arbeiten
    try:
        regions = image_regions(page, cfg.min_image_pt)
        if not regions:
            return
        native = [fitz.Rect(w[:4]) for w in page.get_text("words")]  # vor dem Einfuegen!

        for region in regions:
            st.image_regions += 1
            pix = page.get_pixmap(clip=region, dpi=cfg.dpi, alpha=False)
            if pix.w < 8 or pix.h < 8:
                continue
            sx, sy = region.width / pix.w, region.height / pix.h

            for box, text, conf in ocr.read(pixmap_to_array(pix)):
                text = " ".join(str(text).split())
                if not text:
                    continue
                if conf < cfg.min_conf:
                    st.ocr_items_low_conf += 1
                    continue
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                r = fitz.Rect(region.x0 + min(xs) * sx, region.y0 + min(ys) * sy,
                              region.x0 + max(xs) * sx, region.y0 + max(ys) * sy)
                if r.is_empty:
                    continue
                if any(overlap_ratio(r, n) >= cfg.overlap_skip for n in native):
                    st.ocr_items_overlap_native += 1
                    continue
                insert_invisible(page, r, text)
                st.ocr_items_added += 1
    finally:
        if rotation:
            page.set_rotation(rotation)


def word_set(doc: fitz.Document) -> set[tuple]:
    out = set()
    for pno, page in enumerate(doc):
        for w in page.get_text("words"):
            out.add((pno, round(w[0], 1), round(w[1], 1), w[4]))
    return out


def verify(src: Path, dst: Path, cfg: Config, st: FileStats) -> None:
    with fitz.open(src) as a, fitz.open(dst) as b:
        if a.page_count != b.page_count:
            st.verify_words_ok = st.verify_visual_ok = False
            st.warnings.append("Seitenanzahl unterschiedlich")
            return

        missing = word_set(a) - word_set(b)
        st.verify_words_ok = not missing
        if missing:
            st.warnings.append(f"{len(missing)} Original-Woerter nicht mehr auffindbar, "
                               f"z.B. {sorted(missing)[:3]}")

        diff_pages = []
        for pno in range(a.page_count):
            pa = a[pno].get_pixmap(dpi=cfg.verify_dpi, alpha=False)
            pb = b[pno].get_pixmap(dpi=cfg.verify_dpi, alpha=False)
            if (pa.w, pa.h) != (pb.w, pb.h) or pa.samples != pb.samples:
                diff_pages.append(pno + 1)
        st.verify_visual_ok = not diff_pages
        if diff_pages:
            st.warnings.append(f"Optische Abweichung auf Seite(n) {diff_pages[:20]}")


def process_file(src: Path, dst: Path, ocr: EnglishOcr, cfg: Config) -> FileStats:
    st = FileStats(input=str(src), output=str(dst))
    t0 = time.perf_counter()
    try:
        with fitz.open(src) as doc:
            if doc.needs_pass:
                raise RuntimeError("PDF ist passwortgeschuetzt")
            st.pages = doc.page_count
            for page in doc:
                process_page(page, ocr, cfg, st)
                log.debug("  Seite %d/%d fertig", page.number + 1, st.pages)
            dst.parent.mkdir(parents=True, exist_ok=True)
            doc.save(dst, garbage=3, deflate=True)
        if cfg.verify:
            verify(src, dst, cfg, st)
    except Exception as exc:
        st.error = f"{type(exc).__name__}: {exc}"
        log.exception("Fehler bei %s", src)
    st.seconds = round(time.perf_counter() - t0, 1)
    return st


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def collect_inputs(paths: list[Path], recursive: bool, suffix: str) -> list[Path]:
    files = []
    for p in paths:
        if p.is_dir():
            it = p.rglob("*.pdf") if recursive else p.glob("*.pdf")
            files += [f for f in it if not f.stem.endswith(suffix)]
        elif p.is_file() and p.suffix.lower() == ".pdf":
            files.append(p)
        else:
            log.warning("Uebersprungen (keine PDF-Datei/Ordner): %s", p)
    return sorted(set(files))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Text in eingebetteten PDF-Bildern per OCR (Englisch, RapidOCR) durchsuchbar machen.")
    ap.add_argument("inputs", nargs="+", type=Path, help="PDF-Dateien und/oder Ordner")
    ap.add_argument("-o", "--out-dir", type=Path,
                    help="Zielordner (Standard: neben der Quelldatei)")
    ap.add_argument("--suffix", default="_ocr", help="Dateinamen-Suffix (Standard: _ocr)")
    ap.add_argument("-r", "--recursive", action="store_true", help="Ordner rekursiv durchsuchen")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--min-conf", type=float, default=0.50)
    ap.add_argument("--min-image-size", type=float, default=20.0, help="in pt")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="Vorhandene Ausgaben ueberschreiben")
    ap.add_argument("--report", type=Path, help="JSON-Bericht schreiben")
    ap.add_argument("--log", type=Path, help="Logdatei")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log:
        handlers.append(logging.FileHandler(args.log, encoding="utf-8"))
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", handlers=handlers)
    log.info("pdf_image_ocr %s | PyMuPDF %s", __version__, fitz.VersionBind)

    cfg = Config(dpi=args.dpi, min_conf=args.min_conf, min_image_pt=args.min_image_size,
                 verify=not args.no_verify)

    files = collect_inputs(args.inputs, args.recursive, args.suffix)
    if not files:
        log.error("Keine PDF-Dateien gefunden.")
        return 2

    try:
        ocr = EnglishOcr()
    except Exception as exc:
        log.error("OCR-Engine konnte nicht gestartet werden: %s", exc)
        return 2

    results, failed = [], 0
    for i, src in enumerate(files, 1):
        out_dir = args.out_dir or src.parent
        dst = out_dir / f"{src.stem}{args.suffix}.pdf"
        if dst.resolve() == src.resolve():
            log.error("Ziel = Quelle, uebersprungen: %s", src)
            failed += 1
            continue
        if dst.exists() and not args.overwrite:
            log.info("[%d/%d] existiert bereits, uebersprungen: %s", i, len(files), dst)
            continue

        log.info("[%d/%d] %s", i, len(files), src)
        st = process_file(src, dst, ocr, cfg)
        results.append(st)

        ok = not st.error and st.verify_words_ok is not False and st.verify_visual_ok is not False
        failed += 0 if ok else 1
        log.info("  %s | %d Seiten, %d Bildbereiche, %d Textstellen ergaenzt "
                 "(%d niedrige Konfidenz, %d bereits nativer Text) | %.1fs",
                 "OK" if ok else "PROBLEM", st.pages, st.image_regions, st.ocr_items_added,
                 st.ocr_items_low_conf, st.ocr_items_overlap_native, st.seconds)
        for w in st.warnings:
            log.warning("  %s", w)
        if st.error:
            log.error("  %s", st.error)

    if args.report:
        args.report.write_text(json.dumps(
            {"version": __version__, "config": asdict(cfg),
             "files": [asdict(r) for r in results]}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        log.info("Bericht: %s", args.report)

    log.info("Fertig: %d Datei(en), %d mit Problemen.", len(results), failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
