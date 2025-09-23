#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Study Area Map Builder — menu-driven
- Fixed basemap list (with "None")
- All prompts prefixed with "-->"
- Add categorized/uncategorized layers; optional point labels
- Colors for polygon/line layers (single or categorical)
- NEW: Change line thickness (polygon outlines / line layers)
- DPI & image size editor
- 5% extent padding, legend/north/scalebar positioning, regenerate loop

Dependencies:
  pip install geopandas contextily matplotlib shapely pyproj
"""

import os
import sys
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import contextily as cx
from matplotlib.colors import ListedColormap, to_rgba


# ---------------- Fixed basemap list ----------------
with open('basemaps.json', 'r') as file:
    data = json.load(file)
BASEMAP_KEYS = data['basemap_keys']
POS_MAP = data['pos_map']

# Map human-friendly loc strings to the integer codes expected by Anchored* artists
_LOC_STR2INT = {
    "upper right": 1,
    "upper left": 2,
    "lower left": 3,
    "lower right": 4,
    "right": 5,
    "center left": 6,
    "center right": 7,
    "lower center": 8,
    "upper center": 9,
    "center": 10,
}
def _loc_to_int(loc_str: str) -> int:
    return _LOC_STR2INT.get(loc_str, 1)  # default to "upper right"


# ---------------- I/O helpers (with "-->") ----------------
def ask_yes_no(prompt: str) -> bool:
    while True:
        ans = input(f"--> {prompt} (y/n): ").strip().lower()
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")

def ask_nonempty(prompt: str) -> str:
    while True:
        val = input(f"--> {prompt}: ").strip()
        if val:
            return val
        print("Input cannot be empty.")

def choose_from_list(prompt: str, options: List[str]) -> str:
    if not options:
        raise ValueError("No options to choose from.")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        choice = ask_nonempty(prompt + " (type number or value)")
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1]
            else:
                print(f"Please select a number between 1 and {len(options)}.")
        else:
            if choice in options:
                return choice
            print("Please type one of the values above, or a valid number.")

# ---------------- File helpers ----------------
def list_kml_shp_files(folder: str) -> List[str]:
    try:
        files = os.listdir(folder)
    except FileNotFoundError:
        return []
    return sorted([f for f in files if f.lower().endswith((".shp", ".kml"))])

def resolve_by_index_or_name(
        user_entry: str, available_files: List[str]) -> Optional[str]:
    s = user_entry.strip()
    if s.isdigit():
        idx = int(s)
        if 1 <= idx <= len(available_files):
            return available_files[idx - 1]
        print(f"Please select a number between 1 and {len(available_files)}.")
        return None
    return s

def validate_and_locate(
        user_entry: str, data_folder: str, 
        available_files: List[str]) -> Optional[str]:
    if os.path.isabs(user_entry):
        print("⚠️ Provide only a filename (no absolute paths).")
        return None
    resolved = resolve_by_index_or_name(user_entry, available_files)
    if not resolved:
        return None
    base, ext = os.path.splitext(resolved)
    ext = ext.lower()
    candidates = [f"{base}.shp", f"{base}.kml"] if ext == "" else [resolved] if ext in {".shp", ".kml"} else []
    if not candidates:
        print("❌ Extension must be .shp or .kml (or omit it).")
        return None
    for cand in candidates:
        p = os.path.abspath(os.path.join(data_folder, cand))
        if os.path.isfile(p):
            return p
    print("❌ File not found in the data folder (tried .shp and .kml).")
    return None

# ---------------- Color helpers ----------------
def normalize_color(user_input: str) -> str:
    s = user_input.strip()
    if not s:
        raise ValueError("Empty color.")
    if s.startswith("#"):
        _ = to_rgba(s)  # validate hex
        return s
    s_norm = s.replace(" ", "").lower()
    _ = to_rgba(s_norm)  # validate name
    return s_norm

# ---------------- Data structures ----------------
@dataclass
class SimpleLayer:
    path: str
    kind: str                 # 'polygon' | 'line' | 'point'
    categorized: bool
    legend_name: Optional[str] = None
    category_column: Optional[str] = None
    label_column: Optional[str] = None     # for point labels
    color_single: Optional[str] = None                # for uncategorized polygon/line
    colors_categorical: Optional[List[str]] = None    # for categorized polygon/line
    line_width: Optional[float] = None                # NEW: outline width (polygon) or line width

@dataclass
class StudyAreaConfig:
    epsg: int = 4326
    title: str = "Study Area Map"
    layers: List[SimpleLayer] = field(default_factory=list)

# ---------------- Geo helpers ----------------
def load_vector(path: str) -> gpd.GeoDataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".kml":
        try:
            gdf = gpd.read_file(path, driver="KML")
        except Exception:
            gdf = gpd.read_file(path)
    else:
        gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Layer appears empty: {os.path.basename(path)}")
    if gdf.crs is None:
        print(f"⚠️ No CRS for {os.path.basename(path)}. Assuming EPSG:4326.")
        gdf.set_crs(epsg=4326, inplace=True)
    return gdf

def infer_kind(gdf: gpd.GeoDataFrame) -> str:
    try:
        geom_types = gdf.geometry.geom_type.dropna().unique().tolist()
    except Exception:
        geom_types = []
    if any(t in ("Polygon", "MultiPolygon") for t in geom_types):
        return "polygon"
    if any(t in ("LineString", "MultiLineString") for t in geom_types):
        return "line"
    return "point"

# ---------------- Mapper ----------------
class StudyAreaMapper:
    def __init__(self, config: StudyAreaConfig):
        self.config = config
        self.epsg = config.epsg or 4326
        self.loaded: Dict[str, gpd.GeoDataFrame] = {}
        for lyr in self.config.layers:
            gdf = load_vector(lyr.path)
            if gdf.crs is None or (gdf.crs.to_epsg() != self.epsg if gdf.crs.to_epsg() else True):
                gdf = gdf.to_crs(epsg=self.epsg)
            self.loaded[lyr.path] = gdf

    @staticmethod
    def _extent_union(gdfs: List[gpd.GeoDataFrame]):
        import numpy as np
        bounds = [g.total_bounds for g in gdfs if not g.empty]
        if not bounds:
            return None
        arr = np.array(bounds)
        return (arr[:,0].min(), arr[:,1].min(), arr[:,2].max(), arr[:,3].max())

    def _apply_extent_padding(self, ax, extent, pad_ratio=0.05):
        if not extent:
            return
        minx, miny, maxx, maxy = extent
        w, h = (maxx - minx), (maxy - miny)
        if w <= 0 or h <= 0:
            return
        dx, dy = w * pad_ratio, h * pad_ratio
        ax.set_xlim(minx - dx, maxx + dx)
        ax.set_ylim(miny - dy, maxy + dy)

    def _draw_point_labels(self, ax, gdf_3857, col: str):
        if col not in gdf_3857.columns:
            print(f"⚠️ Label column '{col}' not found; skipping.")
            return
        max_labels = 1000
        if len(gdf_3857) > max_labels:
            print(f"ℹ️ {len(gdf_3857)} points; labeling first {max_labels}.")
        for _, row in gdf_3857.head(max_labels).iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            try:
                x, y = geom.x, geom.y
            except Exception:
                c = geom.centroid
                x, y = c.x, c.y
            ax.text(x, y, str(row[col]), fontsize=7, ha="left", va="bottom")

    def make_map(
        self,
        out_png: str,
        basemap_key: str = "CartoDB.Positron",  # "None" to remove basemap
        title: str = "Study Area Map",
        dpi: int = 150,
        figsize: Tuple[float, float] = (10.0, 10.0),
        legend_loc: str = "lower left",
        cat_legend_loc: str = "upper right",
        north_loc: str = "upper left",
        scalebar_loc: str = "lower right",
    ):
        plot_crs = 3857
        gdfs_plot: Dict[str, gpd.GeoDataFrame] = {p: g.to_crs(epsg=plot_crs) for p, g in self.loaded.items()}
        extent = self._extent_union(list(gdfs_plot.values()))

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        order = ["polygon", "line", "point"]
        uncats = [lyr for lyr in self.config.layers if not lyr.categorized]
        cats   = [lyr for lyr in self.config.layers if lyr.categorized]

        # defaults
        default_poly_lw = 0.8
        default_line_lw = 1.5

        # --- Un-categorized ---
        for kind in order:
            for lyr in [l for l in uncats if l.kind == kind]:
                gdf = gdfs_plot[lyr.path]
                if kind == "polygon":
                    face = lyr.color_single if lyr.color_single else None
                    lw = lyr.line_width if lyr.line_width is not None else default_poly_lw
                    style = dict(alpha=0.5, edgecolor="black", linewidth=lw)
                    if face:
                        style["color"] = face  # polygon fill
                elif kind == "line":
                    col = lyr.color_single if lyr.color_single else None
                    lw = lyr.line_width if lyr.line_width is not None else default_line_lw
                    style = dict(linewidth=lw)
                    if col:
                        style["color"] = col
                else:
                    style = dict(markersize=12, alpha=0.9)

                if lyr.legend_name and lyr.legend_name.lower() != "none":
                    gdf.plot(ax=ax, label=lyr.legend_name, **style)
                else:
                    gdf.plot(ax=ax, **style)

                if kind == "point" and lyr.label_column:
                    self._draw_point_labels(ax, gdf, lyr.label_column)

        # --- Categorized ---
        for kind in order:
            for lyr in [l for l in cats if l.kind == kind]:
                gdf = gdfs_plot[lyr.path]
                col = lyr.category_column
                if (col is None) or (col not in gdf.columns):
                    print(f"⚠️ Column '{col}' missing in {os.path.basename(lyr.path)}; plotting uncategorized.")
                    # fallback to uncat style
                    if kind == "polygon":
                        lw = lyr.line_width if lyr.line_width is not None else default_poly_lw
                        gdf.plot(ax=ax, linewidth=lw)
                    elif kind == "line":
                        lw = lyr.line_width if lyr.line_width is not None else default_line_lw
                        gdf.plot(ax=ax, linewidth=lw)
                    else:
                        gdf.plot(ax=ax)
                else:
                    legend_kw = {"loc": cat_legend_loc}
                    if lyr.legend_name and lyr.legend_name.lower() != "none":
                        legend_kw["title"] = lyr.legend_name
                    kwargs = dict(legend=True, legend_kwds=legend_kw)
                    # apply line width consistently for polygons/lines
                    if kind in ("polygon", "line"):
                        lw = lyr.line_width if lyr.line_width is not None else (default_poly_lw if kind=="polygon" else default_line_lw)
                        kwargs["linewidth"] = lw
                    if lyr.colors_categorical:
                        kwargs["cmap"] = ListedColormap(lyr.colors_categorical)
                    gdf.plot(ax=ax, column=col, **kwargs)

                if kind == "point" and lyr.label_column:
                    self._draw_point_labels(ax, gdf, lyr.label_column)

        # basemap + extent padding
        try:
            if extent is not None:
                self._apply_extent_padding(ax, extent, pad_ratio=0.05)
            if basemap_key != "None":
                node = cx.providers
                for part in basemap_key.split("."):
                    node = node[part]
                cx.add_basemap(ax, source=node, crs="EPSG:3857", attribution_size=6)
        except Exception as e:
            print(f"⚠️ Basemap '{basemap_key}' failed: {e}")

        ax.set_title(title, fontsize=14, weight="bold", pad=12)
        ax.grid(True, linewidth=0.3, alpha=0.5)

        # legend (dedup for any uncats)
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uniq = [(h, l) for h, l in zip(handles, labels) if l and not (l in seen or seen.add(l))]
        if uniq:
            # Create the legend directly at the requested location
            ax.legend(
                [h for h, _ in uniq],
                [l for _, l in uniq],
                loc=legend_loc,
                frameon=True
            )

        # north + scalebar (convert string locs to int codes)
        north = AnchoredText("N ↑", loc=_loc_to_int(north_loc), pad=0.3,
                             prop=dict(size=10, weight='bold'), frameon=True)
        ax.add_artist(north)
        try:
            x0, x1 = ax.get_xlim()
            approx_len = (x1 - x0) * 0.2
            candidates = [50,100,200,250,500,1000,2000,5000,10000,20000,25000,50000,100000]
            length = min(candidates, key=lambda c: abs(c - approx_len))
            bar = AnchoredSizeBar(ax.transData, length,
                                  f"{int(length)} m" if length < 1000 else f"{int(length/1000)} km",
                                  loc=_loc_to_int(scalebar_loc), pad=0.4,
                                  borderpad=0.5, sep=4, frameon=True)
            ax.add_artist(bar)
        except Exception:
            pass

        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Map exported: {out_png}")

# ---------------- Modify helpers ----------------
def choose_position(what: str) -> str:
    print(f"\nChoose {what} position:")
    return choose_from_list("Pick a position", POS_MAP)

def pick_layer_for_color_edit(layers: List[SimpleLayer]) -> Optional[SimpleLayer]:
    candidates = [lyr for lyr in layers if lyr.kind in ("polygon", "line")]
    if not candidates:
        print("No polygon or line layers available to recolor.")
        return None
    opts = [f"{os.path.basename(l.path)}  [{l.kind}]  ({'categorized' if l.categorized else 'uncategorized'})" for l in candidates]
    choice = choose_from_list("Choose a layer to recolor", opts)
    return candidates[opts.index(choice)]

def pick_layer_for_thickness_edit(layers: List[SimpleLayer]) -> Optional[SimpleLayer]:
    candidates = [lyr for lyr in layers if lyr.kind in ("polygon", "line")]
    if not candidates:
        print("No polygon or line layers available to change thickness.")
        return None
    opts = [f"{os.path.basename(l.path)}  [{l.kind}]  ({'categorized' if l.categorized else 'uncategorized'})  "
            f"(current width: {l.line_width if l.line_width is not None else ('0.8' if l.kind=='polygon' else '1.5')})"
            for l in candidates]
    choice = choose_from_list("Choose a layer to change thickness", opts)
    return candidates[opts.index(choice)]

def recolor_layer(lyr: SimpleLayer, gdf: gpd.GeoDataFrame):
    if not lyr.categorized:
        while True:
            s = ask_nonempty("Enter a color for this layer (hex like #32a852 or a name like 'light blue')")
            try:
                lyr.color_single = normalize_color(s)
                print(f"Set color to: {lyr.color_single}")
                break
            except Exception as e:
                print(f"Invalid color: {e}. Please try again.")
    else:
        col = lyr.category_column
        if (col is None) or (col not in gdf.columns):
            print(f"⚠️ Category column '{col}' missing; cannot set categorical colors.")
            return
        cats = sorted(map(str, gdf[col].dropna().unique().tolist()))
        n = len(cats)
        print(f"\nThis layer has {n} categories (sorted):")
        for c in cats[:50]:
            print(f"  - {c}")
        if n > 50:
            print(f"  ... and {n-50} more.")
        colors: List[str] = []
        for i in range(n):
            while True:
                s = ask_nonempty(f"Enter color {i+1}/{n} (hex or name)")
                try:
                    colors.append(normalize_color(s))
                    break
                except Exception as e:
                    print(f"Invalid color: {e}. Please try again.")
        lyr.colors_categorical = colors
        print("Categorical colors set.")

def change_thickness(lyr: SimpleLayer):
    while True:
        val = ask_nonempty("Enter new line thickness (e.g., 0.8 for polygons, 1.5 for lines)")
        try:
            w = float(val)
            if w <= 0:
                print("Please enter a positive number.")
                continue
            lyr.line_width = w
            print(f"Set line thickness to: {w}")
            break
        except ValueError:
            print("Please enter a numeric value.")

# ---------------- Main program ----------------
def main():
    print("=== Study Area Map Builder (menu-driven) ===\n")

    # Data folder + list
    while True:
        data_folder = ask_nonempty("Enter the path to your data folder")
        data_folder = os.path.abspath(os.path.expanduser(data_folder))
        if os.path.isdir(data_folder):
            break
        print("❌ Folder does not exist. Try again.")
    available = list_kml_shp_files(data_folder)
    if available:
        print("\n📂 Available spatial files:")
        for i, name in enumerate(available, 1):
            print(f"  {i}. {name}")
        print("You can refer to files by NUMBER or NAME in the next steps.\n")
    else:
        print("No .shp or .kml files found in this folder.\n")

    # Add layers
    layers: List[SimpleLayer] = []
    while True:
        print("\nAdd data:")
        print("  1. Add categorized spatial data")
        print("  2. Add uncategorized spatial data")
        print("  3. Done adding")
        choice = ask_nonempty("Choose an option (1/2/3)")
        if choice == "1":
            path = None
            while path is None:
                entry = ask_nonempty("Enter filename or number for the categorized layer")
                path = validate_and_locate(entry, data_folder, available)
            gdf_tmp = load_vector(path)
            kind = infer_kind(gdf_tmp)
            cols = [c for c in gdf_tmp.columns if c != gdf_tmp.geometry.name]
            if not cols:
                print("⚠️ No attribute columns found; adding as uncategorized.")
                legend = ask_nonempty("Legend title for this layer (type 'None' to hide)")
                legend = None if legend.lower() == "none" else legend
                lbl_col = None
                if kind == "point" and ask_yes_no("Label points from an attribute"):
                    print("No columns available for labels; skipping.")
                layers.append(SimpleLayer(path=path, kind=kind, categorized=False, legend_name=legend))
            else:
                print("\nAvailable fields:")
                field = choose_from_list("Choose field to categorize by", sorted(cols))
                legend = ask_nonempty("Legend title for this categorized layer (type 'None' to hide)")
                legend = None if legend.lower() == "none" else legend
                lbl_col = None
                if kind == "point" and ask_yes_no("Label points from an attribute"):
                    lbl_col = choose_from_list("Choose label column", sorted(cols))
                layers.append(SimpleLayer(path=path, kind=kind, categorized=True,
                                          legend_name=legend, category_column=field, label_column=lbl_col))
                print("\nLayer Added. Choose next option\n")
        elif choice == "2":
            path = None
            while path is None:
                entry = ask_nonempty("Enter filename or number for the uncategorized layer")
                path = validate_and_locate(entry, data_folder, available)
            gdf_tmp = load_vector(path)
            kind = infer_kind(gdf_tmp)
            legend = ask_nonempty("Legend title for this layer (type 'None' to hide)")
            legend = None if legend.lower() == "none" else legend
            lbl_col = None
            if kind == "point" and ask_yes_no("Label points from an attribute"):
                cols = [c for c in gdf_tmp.columns if c != gdf_tmp.geometry.name]
                if cols:
                    lbl_col = choose_from_list("Choose label column", sorted(cols))
                else:
                    print("No columns available for labels; skipping.")
            layers.append(SimpleLayer(path=path, kind=kind, categorized=False,
                                      legend_name=legend, label_column=lbl_col))
        elif choice == "3":
            if not layers:
                print("You haven't added any layers yet.")
                continue
            break
        else:
            print("Please choose 1, 2, or 3.")

    # Title & EPSG
    title = ask_nonempty("\nMap title")
    epsg = 4326
    if ask_yes_no("Do you want to change the current EPSG 4326"):
        while True:
            epsg_str = ask_nonempty("Enter EPSG code (e.g., 3857)")
            try:
                epsg = int(epsg_str); break
            except ValueError:
                print("Please enter a valid integer EPSG code.")

    cfg = StudyAreaConfig(epsg=epsg, title=title, layers=layers)
    mapper = StudyAreaMapper(cfg)

    # Defaults
    current_basemap = "CartoDB.Positron"
    legend_loc = "lower left"
    cat_legend_loc = "upper right"
    north_loc = "upper left"
    scalebar_loc = "lower right"
    dpi = 150
    figsize = (10.0, 10.0)

    # Initial render
    mapper.make_map(
        out_png="study area map.png",
        basemap_key=current_basemap,
        title=title,
        dpi=dpi,
        figsize=figsize,
        legend_loc=legend_loc,
        cat_legend_loc=cat_legend_loc,
        north_loc=north_loc,
        scalebar_loc=scalebar_loc,
    )

    # Modification loop
    while True:
        print("\nModify map:")
        print("  1. Basemap")
        print("  2. Legend position")
        print("  3. Scalebar position")
        print("  4. North arrow position")
        print("  5. Colors for polygons/lines")
        print("  6. DPI & image size")
        print("  7. Line thickness (polygon outlines / lines)")
        print("  8. Done (regenerate & export)")
        mod = ask_nonempty("Choose an option (1/2/3/4/5/6/7/8)")
        if mod == "1":
            print("\nAvailable basemaps (fixed list):")
            for i, name in enumerate(BASEMAP_KEYS, 1):
                print(f"  {i}. {name}")
            choice = ask_nonempty("Pick a basemap (number or name)")
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(BASEMAP_KEYS):
                    current_basemap = BASEMAP_KEYS[idx - 1]
                else:
                    print(f"Please select a number between 1 and {len(BASEMAP_KEYS)}.")
            else:
                if choice in BASEMAP_KEYS:
                    current_basemap = choice
                else:
                    print("Please choose from the list shown above.")
        elif mod == "2":
            legend_loc = choose_position("legend")
            cat_legend_loc = legend_loc
        elif mod == "3":
            scalebar_loc = choose_position("scale bar")
        elif mod == "4":
            north_loc = choose_position("north arrow")
        elif mod == "5":
            lyr = pick_layer_for_color_edit(layers)
            if lyr:
                gdf = mapper.loaded[lyr.path]
                recolor_layer(lyr, gdf)
        elif mod == "6":
            print(f"\nCurrent DPI: {dpi}")
            while True:
                new_dpi = ask_nonempty("Enter new DPI (positive integer)")
                if new_dpi.isdigit() and int(new_dpi) > 0:
                    dpi = int(new_dpi)
                    break
                print("Please enter a positive integer.")
            print(f"Current image size (inches): {figsize}")
            while True:
                w = ask_nonempty("Enter new width in inches (e.g., 10)")
                h = ask_nonempty("Enter new height in inches (e.g., 10)")
                try:
                    w_f = float(w); h_f = float(h)
                    if w_f > 0 and h_f > 0:
                        figsize = (w_f, h_f)
                        break
                    else:
                        print("Width and height must be positive numbers.")
                except ValueError:
                    print("Please enter numeric values.")
        elif mod == "7":
            lyr = pick_layer_for_thickness_edit(layers)
            if lyr:
                change_thickness(lyr)
        elif mod == "8":
            mapper.make_map(
                out_png="generated_map.png",
                basemap_key=current_basemap,
                title=title,
                dpi=dpi,
                figsize=figsize,
                legend_loc=legend_loc,
                cat_legend_loc=cat_legend_loc,
                north_loc=north_loc,
                scalebar_loc=scalebar_loc,
            )
            print("✅ Map updated. Please check 'generated_map.png'.")
            if ask_yes_no("\nDo you want to make more modifications"):
                continue
            else:
                break
        else:
            print("Please choose 1, 2, 3, 4, 5, 6, 7, or 8.")

    print("\n🙏 Thanks for using Study Area Map Builder. Good luck!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Bye!")
        sys.exit(1)
