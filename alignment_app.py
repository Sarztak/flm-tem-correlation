"""
FLM-TEM Alignment Tool
Place next to detect_lines.py and run: python alignment_app.py
"""

import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
from scipy.ndimage import rotate as scipy_rotate
from detect_lines import detect_grid_lines, group_lines


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_image(path):
    return np.array(Image.open(path))

def to_gray_uint8(img):
    if img.ndim == 3:
        g = img.mean(axis=2)
    else:
        g = img.astype(float)
    g = (g - g.min()) / (g.max() - g.min() + 1e-12)
    return (g * 255).astype(np.uint8)

def fit_image_to_canvas(img_gray, canvas_w, canvas_h):
    """Resize image to fit canvas preserving aspect ratio, white background."""
    ih, iw = img_gray.shape
    scale = min(canvas_w / iw, canvas_h / ih)
    new_w = max(1, int(iw * scale))
    new_h = max(1, int(ih * scale))
    pil = Image.fromarray(img_gray).resize((new_w, new_h), Image.LANCZOS)
    out = Image.new("L", (canvas_w, canvas_h), 255)
    ox = (canvas_w - new_w) // 2
    oy = (canvas_h - new_h) // 2
    out.paste(pil, (ox, oy))
    return out

def get_angles(img):
    lines, _ = detect_grid_lines(
        img,
        sigma=4,
        threshold=0.1,
        min_angle=50,
        min_distance=25,
        num_peaks=200,
    )
    if not lines:
        return []
    grp_lines = group_lines(lines)
    grp_info = [(len(v), np.mean(v)) for v in grp_lines.values()]
    grp_info = sorted(grp_info, key=lambda x: x[0], reverse=True)
    return grp_info

def apply_transforms(flm_gray, rotation_deg, flip_h_count, flip_v_count):
    img = flm_gray.astype(float)
    img = scipy_rotate(img, rotation_deg, reshape=False)
    if flip_h_count % 2 == 1:
        img = np.fliplr(img)
    if flip_v_count % 2 == 1:
        img = np.flipud(img)
    return np.clip(img, 0, 255).astype(np.uint8)

def make_overlay(flm_gray, tem_gray, rotation_deg, flip_h_count, flip_v_count,
                 scale, tx, ty, canvas_w, canvas_h):
    """FLM=green, TEM=magenta, white background."""
    flm_t = apply_transforms(flm_gray, rotation_deg, flip_h_count, flip_v_count)

    # TEM: fit to canvas preserving aspect ratio
    ih, iw = tem_gray.shape
    tem_scale = min(canvas_w / iw, canvas_h / ih)
    tw = max(1, int(iw * tem_scale))
    th = max(1, int(ih * tem_scale))
    tem_pil = Image.fromarray(tem_gray).resize((tw, th), Image.LANCZOS)

    # FLM: fit to canvas then apply user scale
    fh, fw = flm_t.shape
    base_scale = min(canvas_w / fw, canvas_h / fh)
    final_scale = base_scale * scale
    fw2 = max(1, int(fw * final_scale))
    fh2 = max(1, int(fh * final_scale))
    flm_pil = Image.fromarray(flm_t).resize((fw2, fh2), Image.LANCZOS)

    # white canvas
    out = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # paste TEM as magenta (dark pixels become magenta)
    tem_arr = np.array(tem_pil)
    tox = (canvas_w - tw) // 2
    toy = (canvas_h - th) // 2
    inv = 255 - tem_arr
    out[toy:toy+th, tox:tox+tw, 0] = 255 - inv // 2   # R: stay bright, magenta in dark areas
    out[toy:toy+th, tox:tox+tw, 1] = tem_arr            # G: bright where TEM is bright
    out[toy:toy+th, tox:tox+tw, 2] = 255 - inv // 2   # B

    # paste FLM as green channel with tx/ty
    fox = (canvas_w - fw2) // 2 + int(tx)
    foy = (canvas_h - fh2) // 2 + int(ty)
    flm_arr = np.array(flm_pil)
    sx = max(0, -fox); sy = max(0, -foy)
    dx = max(0, fox);  dy = max(0, foy)
    pw = min(fw2 - sx, canvas_w - dx)
    ph = min(fh2 - sy, canvas_h - dy)
    if pw > 0 and ph > 0:
        region = flm_arr[sy:sy+ph, sx:sx+pw]
        inv_flm = 255 - region
        # subtract from R and B to produce green tint
        out[dy:dy+ph, dx:dx+pw, 0] = np.clip(out[dy:dy+ph, dx:dx+pw, 0].astype(int) - inv_flm // 2, 0, 255)
        out[dy:dy+ph, dx:dx+pw, 2] = np.clip(out[dy:dy+ph, dx:dx+pw, 2].astype(int) - inv_flm // 2, 0, 255)

    return Image.fromarray(out)


# ── App ───────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FLM–TEM Alignment")
        self.configure(bg="#f0f0f0")

        self.flm_img  = None
        self.tem_img  = None
        self.flm_gray = None
        self.tem_gray = None

        self.rotation     = 0.0
        self.flip_h_count = 0
        self.flip_v_count = 0
        self.scale        = 1.0
        self.tx           = 0.0
        self.ty           = 0.0

        self._tk_imgs = {}
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # sidebar on right
        sidebar = tk.Frame(self, bg="#e4e4e4", width=270, bd=1, relief=tk.RIDGE)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=0, pady=0)
        sidebar.pack_propagate(False)
        self._build_controls(sidebar)

        # image area fills the rest
        img_area = tk.Frame(self, bg="#f0f0f0")
        img_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        top_row = tk.Frame(img_area, bg="#f0f0f0")
        top_row.pack(fill=tk.BOTH, expand=True)
        self.canvas_flm = self._img_panel(top_row, "FLM", "#1a6e1a")
        self.canvas_tem = self._img_panel(top_row, "TEM", "#6e1a6e")

        bot_row = tk.Frame(img_area, bg="#f0f0f0")
        bot_row.pack(fill=tk.BOTH, expand=True)
        self.canvas_ov = self._img_panel(bot_row, "Overlay — FLM green, TEM magenta", "#7a5500", wide=True)

    def _img_panel(self, parent, label, color, wide=False):
        f = tk.Frame(parent, bg="#f0f0f0")
        if wide:
            f.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        else:
            f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        tk.Label(f, text=label, bg="#f0f0f0", fg=color,
                 font=("Helvetica", 9, "bold")).pack()
        c = tk.Canvas(f, bg="#cccccc", highlightthickness=1,
                      highlightbackground="#aaa")
        c.pack(fill=tk.BOTH, expand=True)
        return c

    def _sep(self, parent):
        tk.Frame(parent, bg="#bbb", height=1).pack(fill=tk.X, padx=6, pady=4)

    def _label(self, parent, text):
        tk.Label(parent, text=text, bg="#e4e4e4", fg="#333",
                 font=("Helvetica", 8, "bold")).pack(anchor="w", padx=8, pady=(6,0))

    def _build_controls(self, p):
        # scroll canvas for controls
        canvas = tk.Canvas(p, bg="#e4e4e4", highlightthickness=0)
        sb = tk.Scrollbar(p, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = tk.Frame(canvas, bg="#e4e4e4")
        win = canvas.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))

        # ── Images
        self._label(inner, "IMAGES")
        tk.Button(inner, text="Load FLM", command=self._load_flm,
                  bg="#cce8cc", fg="#1a5c1a", font=("Helvetica", 9),
                  relief=tk.GROOVE).pack(fill=tk.X, padx=8, pady=2)
        tk.Button(inner, text="Load TEM", command=self._load_tem,
                  bg="#e8cce8", fg="#5c1a5c", font=("Helvetica", 9),
                  relief=tk.GROOVE).pack(fill=tk.X, padx=8, pady=2)
        self.status_lbl = tk.Label(inner, text="Load both images",
                                    bg="#e4e4e4", fg="#666",
                                    font=("Helvetica", 8), wraplength=240,
                                    justify=tk.LEFT)
        self.status_lbl.pack(anchor="w", padx=8)

        # ── Auto rotate
        self._sep(inner)
        self._label(inner, "AUTO ROTATION — HOUGH")
        tk.Button(inner, text="Detect & Auto-Rotate",
                  command=self._auto_rotate,
                  bg="#fff3cc", fg="#5c4000", font=("Helvetica", 9),
                  relief=tk.GROOVE).pack(fill=tk.X, padx=8, pady=2)
        self.hough_lbl = tk.Label(inner, text="", bg="#e4e4e4", fg="#333",
                                   font=("Courier", 8), justify=tk.LEFT,
                                   wraplength=240)
        self.hough_lbl.pack(anchor="w", padx=8)

        # ── Rotation
        self._sep(inner)
        self._label(inner, "ROTATION — FLM")
        self.rot_var = tk.DoubleVar(value=0.0)
        self.rot_total_lbl = tk.Label(inner, text="total: 0.00°",
                                       bg="#e4e4e4", fg="#333",
                                       font=("Courier", 9))
        self._make_slider(inner, "coarse", self.rot_var, -180, 180, 1, self._on_rot_change)
        self.fine_var = tk.DoubleVar(value=0.0)
        self._make_slider(inner, "fine ±2", self.fine_var, -2, 2, 0.01, self._on_rot_change)
        self.rot_total_lbl.pack(anchor="w", padx=8)

        # ── Flip
        self._sep(inner)
        self._label(inner, "FLIP — FLM  (each press = one flip)")
        f = tk.Frame(inner, bg="#e4e4e4")
        f.pack(fill=tk.X, padx=8, pady=3)
        tk.Button(f, text="↔  Flip Horizontal", command=self._flip_h,
                  bg="#dde0ff", fg="#00008b", font=("Helvetica", 9),
                  relief=tk.GROOVE).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(f, text="↕  Flip Vertical", command=self._flip_v,
                  bg="#dde0ff", fg="#00008b", font=("Helvetica", 9),
                  relief=tk.GROOVE).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.flip_lbl = tk.Label(inner, text="H: 0 flips   V: 0 flips",
                                  bg="#e4e4e4", fg="#555", font=("Courier", 8))
        self.flip_lbl.pack(anchor="w", padx=8)

        # ── Scale
        self._sep(inner)
        self._label(inner, "SCALE — FLM")
        self.sc_var = tk.DoubleVar(value=1.0)
        self._make_slider(inner, "scale", self.sc_var, 0.1, 5.0, 0.01,
                          lambda: setattr(self, 'scale', self.sc_var.get()) or self._redraw())

        # ── Translation
        self._sep(inner)
        self._label(inner, "TRANSLATION")
        self.tx_var = tk.DoubleVar(value=0.0)
        self.ty_var = tk.DoubleVar(value=0.0)
        self._make_slider(inner, "tx", self.tx_var, -600, 600, 1,
                          lambda: setattr(self, 'tx', self.tx_var.get()) or self._redraw())
        self._make_slider(inner, "ty", self.ty_var, -600, 600, 1,
                          lambda: setattr(self, 'ty', self.ty_var.get()) or self._redraw())

        # ── Matrix
        self._sep(inner)
        self._label(inner, "AFFINE MATRIX")
        self.matrix_lbl = tk.Label(inner, text="", bg="#fff", fg="#222",
                                    font=("Courier", 8), justify=tk.LEFT,
                                    padx=6, pady=4, relief=tk.SUNKEN)
        self.matrix_lbl.pack(fill=tk.X, padx=8, pady=2)
        tk.Button(inner, text="Copy as numpy", command=self._copy_matrix,
                  bg="#e4e4e4", fg="#333", font=("Helvetica", 9),
                  relief=tk.GROOVE).pack(fill=tk.X, padx=8, pady=2)

        # ── Reset
        self._sep(inner)
        tk.Button(inner, text="Reset All", command=self._reset,
                  bg="#ffd0d0", fg="#8b0000", font=("Helvetica", 9),
                  relief=tk.GROOVE).pack(fill=tk.X, padx=8, pady=6)

    def _make_slider(self, parent, label, var, from_, to, res, cmd):
        f = tk.Frame(parent, bg="#e4e4e4")
        f.pack(fill=tk.X, padx=8, pady=1)
        tk.Label(f, text=label, bg="#e4e4e4", fg="#444",
                 font=("Helvetica", 9), width=8, anchor="w").pack(side=tk.LEFT)
        val_lbl = tk.Label(f, bg="#e4e4e4", fg="#000",
                           font=("Courier", 8), width=7)
        val_lbl.pack(side=tk.RIGHT)
        def on_slide(v):
            val_lbl.config(text=f"{float(v):.2f}")
            cmd()
        s = tk.Scale(f, from_=from_, to=to, resolution=res,
                     orient=tk.HORIZONTAL, variable=var, showvalue=False,
                     bg="#e4e4e4", troughcolor="#bbb", highlightthickness=0,
                     command=on_slide)
        s.pack(side=tk.LEFT, fill=tk.X, expand=True)
        val_lbl.config(text=f"{var.get():.2f}")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_rot_change(self):
        self.rotation = self.rot_var.get() + self.fine_var.get()
        self.rot_total_lbl.config(text=f"total: {self.rotation:.2f}°")
        self._redraw()

    def _flip_h(self):
        self.flip_h_count += 1
        self.flip_lbl.config(text=f"H: {self.flip_h_count} flips   V: {self.flip_v_count} flips")
        self._redraw()

    def _flip_v(self):
        self.flip_v_count += 1
        self.flip_lbl.config(text=f"H: {self.flip_h_count} flips   V: {self.flip_v_count} flips")
        self._redraw()

    def _load_flm(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image", "*.tif *.tiff *.png *.jpg")])
        if not path:
            return
        self.flm_img  = load_image(path)
        self.flm_gray = to_gray_uint8(self.flm_img)
        self.status_lbl.config(text=f"FLM: {self.flm_img.shape}")
        self._redraw()

    def _load_tem(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image", "*.tif *.tiff *.png *.jpg")])
        if not path:
            return
        self.tem_img  = load_image(path)
        self.tem_gray = to_gray_uint8(self.tem_img)
        self.status_lbl.config(text=f"TEM: {self.tem_img.shape}")
        self._redraw()

    def _auto_rotate(self):
        if self.flm_img is None or self.tem_img is None:
            self.status_lbl.config(text="Load both images first")
            return
        self.status_lbl.config(text="Detecting…")
        self.update()

        flm_grp = get_angles(self.flm_img)
        tem_grp = get_angles(self.tem_img)

        if len(flm_grp) < 1 or len(tem_grp) < 1:
            self.hough_lbl.config(text="Not enough lines detected")
            self.status_lbl.config(text="Detection failed")
            return

        flm_a1  = flm_grp[0][1]
        tem_a1  = tem_grp[0][1]
        diff    = tem_a1 - flm_a1
        applied = -diff   # rotate FLM by opposite sign

        # add on top of whatever rotation already exists
        self.rotation += applied
        self.rot_var.set(round(self.rotation))
        self.fine_var.set(round(self.rotation - round(self.rotation), 2))
        self.rot_total_lbl.config(text=f"total: {self.rotation:.2f}°")

        self.hough_lbl.config(
            text=(f"FLM a1:     {flm_a1:.2f}°\n"
                  f"TEM a1:     {tem_a1:.2f}°\n"
                  f"Δ(TEM-FLM): {diff:.2f}°\n"
                  f"Applied:    {applied:.2f}°\n"
                  f"Total rot:  {self.rotation:.2f}°")
        )
        self.status_lbl.config(text="Done")
        self._redraw()

    def _reset(self):
        self.rotation     = 0.0
        self.flip_h_count = 0
        self.flip_v_count = 0
        self.scale        = 1.0
        self.tx           = 0.0
        self.ty           = 0.0
        self.rot_var.set(0)
        self.fine_var.set(0)
        self.sc_var.set(1.0)
        self.tx_var.set(0)
        self.ty_var.set(0)
        self.rot_total_lbl.config(text="total: 0.00°")
        self.flip_lbl.config(text="H: 0 flips   V: 0 flips")
        self.hough_lbl.config(text="")
        self._redraw()

    def _copy_matrix(self):
        import math
        r   = math.radians(self.rotation)
        s   = self.scale
        cos, sin = s * math.cos(r), s * math.sin(r)
        sx  = -1 if self.flip_h_count % 2 == 1 else 1
        sy  = -1 if self.flip_v_count % 2 == 1 else 1
        txt = (f"import numpy as np\n"
               f"M = np.array([[{cos*sx:.4f}, {-sin*sx:.4f}, {self.tx:.1f}],\n"
               f"              [{sin*sy:.4f},  {cos*sy:.4f}, {self.ty:.1f}]])")
        self.clipboard_clear()
        self.clipboard_append(txt)
        self.status_lbl.config(text="Matrix copied")

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _canvas_wh(self, canvas):
        self.update_idletasks()
        return max(canvas.winfo_width(), 100), max(canvas.winfo_height(), 100)

    def _redraw(self):
        if self.flm_gray is not None:
            w, h = self._canvas_wh(self.canvas_flm)
            pil = fit_image_to_canvas(self.flm_gray, w, h)
            tk_img = ImageTk.PhotoImage(pil)
            self.canvas_flm.delete("all")
            self.canvas_flm.create_image(0, 0, anchor=tk.NW, image=tk_img)
            self._tk_imgs["flm"] = tk_img

        if self.tem_gray is not None:
            w, h = self._canvas_wh(self.canvas_tem)
            pil = fit_image_to_canvas(self.tem_gray, w, h)
            tk_img = ImageTk.PhotoImage(pil)
            self.canvas_tem.delete("all")
            self.canvas_tem.create_image(0, 0, anchor=tk.NW, image=tk_img)
            self._tk_imgs["tem"] = tk_img

        if self.flm_gray is not None and self.tem_gray is not None:
            w, h = self._canvas_wh(self.canvas_ov)
            ov = make_overlay(
                self.flm_gray, self.tem_gray,
                self.rotation,
                self.flip_h_count, self.flip_v_count,
                self.scale, self.tx, self.ty,
                w, h
            )
            tk_img = ImageTk.PhotoImage(ov)
            self.canvas_ov.delete("all")
            self.canvas_ov.create_image(0, 0, anchor=tk.NW, image=tk_img)
            self._tk_imgs["ov"] = tk_img

        self._update_matrix()

    def _update_matrix(self):
        import math
        r   = math.radians(self.rotation)
        s   = self.scale
        cos, sin = s * math.cos(r), s * math.sin(r)
        sx  = -1 if self.flip_h_count % 2 == 1 else 1
        sy  = -1 if self.flip_v_count % 2 == 1 else 1
        self.matrix_lbl.config(
            text=(f"[{cos*sx:7.3f}  {-sin*sx:7.3f}  {self.tx:5.0f}]\n"
                  f"[{sin*sy:7.3f}   {cos*sy:7.3f}  {self.ty:5.0f}]\n"
                  f"[  0.000    0.000      1]")
        )


if __name__ == "__main__":
    app = App()
    app.geometry("1300x780")
    app.mainloop()
