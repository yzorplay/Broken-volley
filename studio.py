"""Студия моделей мира: разметка видео -> запекание -> игра на сгенерированной модели.

Запуск:  python studio.py [video.mp4 | project.json | model.wm]
Рядом должны лежать bake.py и wm.py. Нужны: numpy, opencv-python, pillow, torch.

Разметка: запустите видео и ДЕРЖИТЕ клавишу действия, пока кнопка «нажата» в кадре.
Проект сохраняется автоматически рядом с видео: <видео>.project.json
"""
import json
import os
import queue
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
except ImportError as _e:
    sys.exit(f"Не хватает библиотеки: {_e}. Установите: pip install numpy opencv-python pillow torch")

if hasattr(sys, '_MEIPASS'):
    # Если запущено из .exe (PyInstaller)
    HERE = sys._MEIPASS
else:
    # Если запускается как обычный .py файл
    HERE = os.path.dirname(os.path.abspath(__file__))

# Ваша проверка (теперь она будет искать файлы внутри временной папки .exe)
for _f in ("bake.py", "wm.py"):
    if not os.path.exists(os.path.join(HERE, _f)):
        sys.exit(f"Не найден файл {os.path.join(HERE, _f)} — положите его рядом со studio.py")

import bake as bakemod  # noqa: E402


# ───────────────────────── утилиты ─────────────────────────

def read_json(path):
    """Читает JSON, пробуя UTF-8 (с BOM или без) и Windows-1251."""
    raw = open(path, "rb").read()
    last = None
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return json.loads(raw.decode(enc))
        except ValueError as e:
            last = e
    raise ValueError(f"{path}: не удалось прочитать JSON ({last})")


def num(var, typ, name):
    try:
        return typ(str(var.get()).strip().replace(",", "."))
    except ValueError:
        raise ValueError(f"Поле «{name}»: ожидается число, введено «{var.get()}»")


def fmt_time(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def merge(ivs):
    ivs = sorted([float(s), float(e)] for s, e in ivs if e - s >= 0.02)
    out = []
    for s, e in ivs:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def new_project():
    return {"name": "model", "fps": 30, "latency": 0, "actions": [], "intervals": [],
            "video_name": "", "video_path": "", "bake": {}}


class KeyTracker:
    """Какие клавиши зажаты прямо сейчас (с защитой от автоповтора на Linux)."""

    def __init__(self, root, on_down, on_up):
        self.root, self.on_down, self.on_up = root, on_down, on_up
        self.down, self.pending = set(), {}
        root.bind_all("<KeyPress>", self._p, add="+")
        root.bind_all("<KeyRelease>", self._r, add="+")

    def _p(self, e):
        k = e.keysym.lower()
        if k in self.pending:
            self.root.after_cancel(self.pending.pop(k))
            return
        if k in self.down:
            return
        self.down.add(k)
        self.on_down(k)

    def _r(self, e):
        k = e.keysym.lower()
        if k in self.down and k not in self.pending:
            self.pending[k] = self.root.after(12, lambda: self._fin(k))

    def _fin(self, k):
        self.pending.pop(k, None)
        if k in self.down:
            self.down.discard(k)
            self.on_up(k)

    def release_all(self):
        for k in list(self.down):
            if k in self.pending:
                self.root.after_cancel(self.pending.pop(k))
            self.down.discard(k)
            self.on_up(k)


class VideoView:
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError("OpenCV не смог открыть файл (mp4/h264 обычно ок)")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.total <= 0:
            raise RuntimeError("Не удалось определить длину видео")
        self.dur = self.total / self.fps
        self.cur, self.img = -1, None
        if not self.seek(0):
            raise RuntimeError("Не удалось прочитать первый кадр")

    def seek(self, f):
        f = max(0, min(int(f), self.total - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = self.cap.read()
        if ok:
            self.img, self.cur = img, f
        return ok

    def advance_to(self, target):
        while self.cur + 1 < target:
            if not self.cap.grab():
                return False
            self.cur += 1
        ok, img = self.cap.read()
        if not ok:
            return False
        self.img, self.cur = img, target
        return True

    def time(self):
        return max(0, self.cur) / self.fps

    def release(self):
        self.cap.release()


# ───────────────────────── приложение ─────────────────────────

class App:
    def __init__(self, root):
        self.root = root
        root.title("Студия Broken Valley")
        root.geometry("1120x800")
        self.proj, self.proj_path, self.video = new_project(), None, None
        self.is_playing, self.live, self._job = False, {}, None
        self.pf0 = self.pt0 = 0
        self._upd = False
        self.capture = None
        self.q, self.worker, self.stop_flag = queue.Queue(), None, False
        self._prev_path, self._prev_mtime, self._prev_photo = None, 0, None
        self.player, self.game_run, self._game_job = None, False, None
        self.game_keys, self._last_t, self._fps_ema, self._gn = [], 0, 0.0, 0
        self._photo = self._gphoto = None

        self.status = tk.StringVar(value="Откройте видео на вкладке «Разметка»")
        ttk.Label(root, textvariable=self.status, anchor="w", relief="sunken").pack(side="bottom", fill="x")
        self.keys = KeyTracker(root, self.on_key_down, self.on_key_up)
        self.dev_var = tk.StringVar(value="auto")

        self.nb = ttk.Notebook(root)
        t1, t2, t3 = ttk.Frame(self.nb), ttk.Frame(self.nb), ttk.Frame(self.nb)
        self.nb.add(t1, text="1. Разметка")
        self.nb.add(t2, text="2. Запекание")
        self.nb.add(t3, text="3. Игра")
        self.nb.pack(fill="both", expand=True)
        self._build_annotate(t1)
        self._build_bake(t2)
        self._build_play(t3)
        self.nb.bind("<<NotebookTabChanged>>", self.on_tab)
        self._apply_project()
        self._poll()

    # ─── общее ───

    def err(self, msg):
        print("ОШИБКА:", msg, file=sys.stderr)
        self.status.set("⚠ " + msg.splitlines()[0])
        self.log("⚠ " + msg)
        messagebox.showerror("Ошибка", msg)

    def warn(self, msg):
        print("ПРЕДУПРЕЖДЕНИЕ:", msg, file=sys.stderr)
        self.status.set("⚠ " + msg)

    def save_project(self):
        if not self.proj_path:
            return
        try:
            tmp = self.proj_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.proj, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.proj_path)
        except OSError as e:
            self.warn(f"Не удалось сохранить проект {self.proj_path}: {e}")

    def on_tab(self, _=None):
        self.stop_play()
        self.keys.release_all()
        self.game_run = self.nb.index(self.nb.select()) == 2 and self.player is not None
        if self.game_run:
            self._game_kick()
        self._sync_game_btn()
        self.root.focus_set()

    def on_key_down(self, k):
        if self.capture is not None:
            i, self.capture = self.capture, None
            if k != "escape" and i < len(self.proj["actions"]):
                self.proj["actions"][i]["key"] = k
                self.save_project()
                self.refresh_tree()
                self.status.set(f"Клавиша «{k}» назначена")
            return
        if self.nb.index(self.nb.select()) == 0 and self.is_playing:
            if isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry, ttk.Spinbox, ttk.Combobox)):
                return
            for i, a in enumerate(self.proj["actions"]):
                if a["key"] == k and i not in self.live:
                    self.live[i] = self.video.time()

    def on_key_up(self, k):
        if self.nb.index(self.nb.select()) == 0 and self.video:
            for i, a in enumerate(self.proj["actions"]):
                if a["key"] == k and i in self.live:
                    self._finish_live(i, self.video.time())

    # ─── вкладка 1: разметка ───

    def _build_annotate(self, tab):
        top = ttk.Frame(tab)
        top.pack(fill="both", expand=True)
        left = ttk.Frame(top)
        left.pack(side="left", fill="both", expand=True)
        self.cv = tk.Canvas(left, bg="#111", width=640, height=360, highlightthickness=0)
        self.cv.pack(fill="both", expand=True)
        self.cv.bind("<Button-1>", lambda e: self.toggle_play())
        self.cv.bind("<Configure>", lambda e: self.show_frame())
        ctl = ttk.Frame(left)
        ctl.pack(fill="x")
        self.btn_play = ttk.Button(ctl, text="▶", width=4, command=self.toggle_play, takefocus=False)
        self.btn_play.pack(side="left", padx=4, pady=4)
        self.scale = ttk.Scale(ctl, from_=0, to=1, command=self.on_scale)
        self.scale.pack(side="left", fill="x", expand=True, padx=4)
        self.lbl_time = ttk.Label(ctl, text="0:00.0 / 0:00.0", width=16)
        self.lbl_time.pack(side="left", padx=4)

        right = ttk.Frame(top, width=300)
        right.pack(side="right", fill="y", padx=6, pady=4)
        ttk.Button(right, text="Открыть видео…", command=self.ask_video, takefocus=False).pack(fill="x")
        self.lbl_video = ttk.Label(right, text="видео не выбрано", wraplength=280)
        self.lbl_video.pack(fill="x", pady=2)
        row = ttk.Frame(right)
        row.pack(fill="x", pady=2)
        ttk.Button(row, text="Открыть проект…", command=self.ask_project, takefocus=False).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Сохранить как…", command=self.save_project_as, takefocus=False).pack(side="left", expand=True, fill="x")

        form = ttk.Frame(right)
        form.pack(fill="x", pady=4)
        self.fps_var, self.lat_var, self.speed_var = tk.StringVar(value="30"), tk.StringVar(value="0"), tk.StringVar(value="1")
        for r, (lbl, var, w) in enumerate([("Кадров/с модели", self.fps_var, 6), ("Задержка меток, кадров", self.lat_var, 6)]):
            ttk.Label(form, text=lbl).grid(row=r, column=0, sticky="w")
            ttk.Spinbox(form, textvariable=var, from_=1 if r == 0 else -30, to=120 if r == 0 else 30, width=w).grid(row=r, column=1, padx=4, pady=1)
        ttk.Label(form, text="Скорость просмотра").grid(row=2, column=0, sticky="w")
        ttk.Combobox(form, textvariable=self.speed_var, values=["0.25", "0.5", "1"], width=5).grid(row=2, column=1, padx=4, pady=1)
        self.fps_var.trace_add("write", lambda *a: self._sync_vars())
        self.lat_var.trace_add("write", lambda *a: self._sync_vars())

        ttk.Label(right, text="Действия (кнопки)").pack(anchor="w", pady=(6, 0))
        self.tree = ttk.Treeview(right, columns=("key", "n"), height=6, selectmode="browse")
        self.tree.heading("#0", text="Название")
        self.tree.heading("key", text="Клавиша")
        self.tree.heading("n", text="Нажатий / сек")
        self.tree.column("#0", width=100)
        self.tree.column("key", width=70)
        self.tree.column("n", width=90)
        self.tree.pack(fill="x")
        for text, cmd in [("Добавить", self.add_action), ("Назначить клавишу", self.set_key),
                          ("Переименовать", self.rename_action), ("Стереть разметку", self.clear_action),
                          ("Удалить действие", self.remove_action)]:
            ttk.Button(right, text=text, command=cmd, takefocus=False).pack(fill="x", pady=1)
        ttk.Label(right, wraplength=280, foreground="#666", text=(
            "Как размечать: запустите видео (▶ или клик по кадру) и держите клавишу действия, "
            "пока кнопка нажата. ПКМ по полоске на таймлайне — удалить интервал. "
            "Для точности включите скорость 0.5 или 0.25.")).pack(fill="x", pady=6)

        self.tl = tk.Canvas(tab, bg="#1b1b1b", height=60, highlightthickness=0)
        self.tl.pack(fill="x", side="bottom")
        self._tlg = (110, 1, 22, 1)
        self.tl.bind("<Configure>", lambda e: self.draw_timeline())
        self.tl.bind("<Button-1>", self.tl_click)
        self.tl.bind("<B1-Motion>", self.tl_click)
        self.tl.bind("<Button-3>", self.tl_delete)
        self.tl.bind("<Button-2>", self.tl_delete)

    def _sync_vars(self):
        try:
            self.proj["fps"] = float(self.fps_var.get().replace(",", "."))
            self.proj["latency"] = float(self.lat_var.get().replace(",", "."))
        except ValueError:
            return
        self.save_project()

    def _speed(self):
        try:
            return max(0.05, float(self.speed_var.get().replace(",", ".")))
        except ValueError:
            return 1.0

    def ask_video(self):
        p = filedialog.askopenfilename(title="Видео", filetypes=[("Видео", "*.mp4 *.avi *.mov *.mkv *.webm"), ("Все файлы", "*.*")])
        if p:
            self.open_video(p)

    def open_video(self, path):
        path = os.path.abspath(path)
        try:
            v = VideoView(path)
        except Exception as e:
            self.err(f"Видео {path}: {e}")
            return
        self.stop_play()
        if self.video:
            self.video.release()
        self.video = v
        pp = path + ".project.json"
        if os.path.exists(pp):
            self.load_project(pp, open_video=False)
        else:
            self.proj["intervals"] = [[] for _ in self.proj["actions"]]
            self.proj_path = pp
        self.proj["video_path"], self.proj["video_name"] = path, os.path.basename(path)
        self.proj["name"] = os.path.splitext(os.path.basename(path))[0]
        self.lbl_video.config(text=f"{os.path.basename(path)}\n{fmt_time(v.dur)}, {v.fps:.1f} к/с")
        self.scale.config(to=v.dur)
        self.out_var.set(os.path.splitext(path)[0] + ".wm")
        self.save_project()
        self.refresh_tree()
        self.update_pos()
        self.show_frame()
        self.status.set(f"Видео открыто: {path}. Проект: {self.proj_path}")

    def ask_project(self):
        p = filedialog.askopenfilename(title="Проект", filetypes=[("Проект", "*.json"), ("Все файлы", "*.*")])
        if p:
            self.load_project(p)

    def save_project_as(self):
        p = filedialog.asksaveasfilename(title="Сохранить проект", defaultextension=".json", filetypes=[("Проект", "*.json")])
        if p:
            self.proj_path = p
            self.save_project()
            self.status.set(f"Проект сохранён: {p}")

    def load_project(self, path, open_video=True):
        try:
            data = read_json(path)
            if not isinstance(data, dict) or "actions" not in data:
                raise ValueError("в файле нет поля actions — это не проект студии")
        except (OSError, ValueError) as e:
            self.err(f"Проект {path}: {e}")
            return
        self.proj = {**new_project(), **data}
        self.proj_path = path
        n = len(self.proj["actions"])
        ivs = list(self.proj.get("intervals", []))[:n]
        self.proj["intervals"] = ivs + [[] for _ in range(n - len(ivs))]
        for a in self.proj["actions"]:
            a.setdefault("key", "")
        self._apply_project()
        vp = self.proj.get("video_path", "")
        if open_video and vp:
            if os.path.exists(vp):
                self.open_video(vp)
            else:
                self.warn(f"Видео из проекта не найдено: {vp} — откройте его вручную")
        self.refresh_tree()

    def _apply_project(self):
        fps, lat = self.proj["fps"], self.proj["latency"]   # trace в _sync_vars перезаписывает proj
        self.fps_var.set(f"{fps:g}")
        self.lat_var.set(f"{lat:g}")
        cfg = {**bakemod.DEFAULTS, **self.proj.get("bake", {})}
        bv = self.bv
        bv["w"].set(cfg["size"][0]); bv["h"].set(cfg["size"][1]); bv["color"].set(bool(cfg["color"]))
        for k in ("ctx", "act_ctx", "width", "steps", "batch", "rollout", "lr"):
            bv[k].set(cfg[k])
        for k, v in zip(("cl", "ct", "cr", "cb"), cfg["crop"]):
            bv[k].set(v)
        self.refresh_tree()
        self.draw_timeline()

    # действия

    def sel_action(self):
        s = self.tree.selection()
        if not s:
            self.warn("Сначала выберите действие в списке")
            return None
        return int(s[0])

    def add_action(self):
        name = simpledialog.askstring("Новое действие", "Название (например, «Прыжок»):", parent=self.root)
        if not name or not name.strip():
            return
        self.proj["actions"].append({"name": name.strip(), "key": ""})
        self.proj["intervals"].append([])
        self.refresh_tree()
        self.begin_capture(len(self.proj["actions"]) - 1)

    def begin_capture(self, i):
        self.capture = i
        self.root.focus_set()
        self.status.set("Нажмите клавишу для действия… (Esc — отмена)")

    def set_key(self):
        i = self.sel_action()
        if i is not None:
            self.begin_capture(i)

    def rename_action(self):
        i = self.sel_action()
        if i is None:
            return
        name = simpledialog.askstring("Переименовать", "Название:", initialvalue=self.proj["actions"][i]["name"], parent=self.root)
        if name and name.strip():
            self.proj["actions"][i]["name"] = name.strip()
            self.save_project()
            self.refresh_tree()

    def clear_action(self):
        i = self.sel_action()
        if i is not None and messagebox.askyesno("Стереть", f"Стереть всю разметку «{self.proj['actions'][i]['name']}»?"):
            self.proj["intervals"][i] = []
            self.save_project()
            self.refresh_tree()

    def remove_action(self):
        i = self.sel_action()
        if i is not None and messagebox.askyesno("Удалить", f"Удалить действие «{self.proj['actions'][i]['name']}» вместе с разметкой?"):
            self.proj["actions"].pop(i)
            self.proj["intervals"].pop(i)
            self.live.clear()
            self.save_project()
            self.refresh_tree()

    def refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i, a in enumerate(self.proj["actions"]):
            ivs = self.proj["intervals"][i]
            tot = sum(e - s for s, e in ivs)
            self.tree.insert("", "end", iid=str(i), text=a["name"], values=(a["key"] or "—", f"{len(ivs)} / {tot:.1f}"))
        self.tl.config(height=max(44, 16 + 22 * len(self.proj["actions"])))
        self.draw_timeline()

    def _finish_live(self, i, t_end):
        s = self.live.pop(i, None)
        if s is None or i >= len(self.proj["intervals"]):
            return
        if t_end - s >= 0.02:
            self.proj["intervals"][i] = merge(self.proj["intervals"][i] + [[s, t_end]])
            self.save_project()
            self.refresh_tree()

    # воспроизведение разметки

    def toggle_play(self):
        if self.is_playing:
            self.stop_play()
            return
        v = self.video
        if not v:
            self.warn("Сначала откройте видео")
            return
        if v.cur >= v.total - 2:
            v.seek(0)
        self.is_playing = True
        self.btn_play.config(text="⏸")
        self.root.focus_set()
        self._anchor()
        for i, a in enumerate(self.proj["actions"]):
            if a["key"] and a["key"] in self.keys.down:
                self.live[i] = v.time()
        self._job = self.root.after(1, self._play_tick)

    def stop_play(self):
        if self._job:
            self.root.after_cancel(self._job)
            self._job = None
        was = self.is_playing
        self.is_playing = False
        self.btn_play.config(text="▶")
        if was and self.video:
            for i in list(self.live):
                self._finish_live(i, self.video.time())
        self.live.clear()

    def _anchor(self):
        self.pf0, self.pt0 = self.video.cur, time.perf_counter()

    def _play_tick(self):
        self._job = None
        if not self.is_playing:
            return
        v = self.video
        target = self.pf0 + int((time.perf_counter() - self.pt0) * v.fps * self._speed())
        if target > v.cur and (target >= v.total or not v.advance_to(target)):
            self.stop_play()
            self.update_pos()
            return
        self.update_pos()
        self.show_frame()
        self.draw_timeline()
        self._job = self.root.after(5, self._play_tick)

    def on_scale(self, val):
        if self._upd or not self.video:
            return
        self.seek_time(float(val))

    def seek_time(self, t):
        v = self.video
        for i in list(self.live):
            self._finish_live(i, v.time())
        v.seek(t * v.fps)
        if self.is_playing:
            self._anchor()
            for i, a in enumerate(self.proj["actions"]):
                if a["key"] and a["key"] in self.keys.down:
                    self.live[i] = v.time()
        self.update_pos()
        self.show_frame()
        self.draw_timeline()

    def update_pos(self):
        if not self.video:
            return
        self._upd = True
        self.scale.set(self.video.time())
        self._upd = False
        self.lbl_time.config(text=f"{fmt_time(self.video.time())} / {fmt_time(self.video.dur)}")

    def get_crop(self):
        try:
            c = [float(self.bv[k].get().replace(",", ".")) if isinstance(self.bv[k].get(), str) else float(self.bv[k].get())
                 for k in ("cl", "ct", "cr", "cb")]
            return c if all(0 <= x < 1 for x in c) and c[0] + c[2] < 1 and c[1] + c[3] < 1 else [0, 0, 0, 0]
        except (ValueError, tk.TclError):
            return [0, 0, 0, 0]

    def show_frame(self):
        c = self.cv
        c.delete("all")
        cw, ch = max(c.winfo_width(), 2), max(c.winfo_height(), 2)
        if not self.video or self.video.img is None:
            c.create_text(cw // 2, ch // 2, fill="#777", text="Откройте видео")
            return
        img = self.video.img
        h, w = img.shape[:2]
        s = min(cw / w, ch / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        pil = Image.fromarray(cv2.cvtColor(cv2.resize(img, (nw, nh)), cv2.COLOR_BGR2RGB))
        self._photo = ImageTk.PhotoImage(pil)
        ox, oy = (cw - nw) // 2, (ch - nh) // 2
        c.create_image(ox, oy, anchor="nw", image=self._photo)
        l, t, r, b = self.get_crop()
        if l or t or r or b:
            c.create_rectangle(ox + nw * l, oy + nh * t, ox + nw * (1 - r), oy + nh * (1 - b), outline="#ff5050", dash=(4, 3))
        held = [self.proj["actions"][i]["name"] for i in self.live if i < len(self.proj["actions"])]
        if held:
            c.create_text(ox + 8, oy + 8, anchor="nw", fill="#4caf50", font=("TkDefaultFont", 14, "bold"), text="● " + ", ".join(held))

    def draw_timeline(self):
        c = self.tl
        c.delete("all")
        acts = self.proj["actions"]
        dur = self.video.dur if self.video else 0
        if not acts or not dur:
            c.create_text(8, 22, anchor="w", fill="#777", text="Добавьте действия и откройте видео — здесь появится разметка")
            return
        x0, rowh = 110, 22
        span = max(1, c.winfo_width() - x0 - 8)
        self._tlg = (x0, span, rowh, dur)
        now = self.video.time()
        for i, a in enumerate(acts):
            y = 8 + i * rowh
            c.create_text(4, y + 9, anchor="w", fill="#ddd", text=f"{a['name']} [{a['key'] or '?'}]")
            c.create_rectangle(x0, y, x0 + span, y + rowh - 4, fill="#2a2a2a", outline="")
            for s, e in self.proj["intervals"][i]:
                c.create_rectangle(x0 + s / dur * span, y, x0 + e / dur * span, y + rowh - 4, fill="#4caf50", outline="")
            if i in self.live:
                c.create_rectangle(x0 + self.live[i] / dur * span, y, x0 + now / dur * span, y + rowh - 4, fill="#ffb300", outline="")
        px = x0 + now / dur * span
        c.create_line(px, 0, px, 8 + len(acts) * rowh, fill="#ff5050")

    def tl_click(self, e):
        if not self.video:
            return
        x0, span, _, dur = self._tlg
        self.seek_time(min(max((e.x - x0) / span * dur, 0), dur))

    def tl_delete(self, e):
        if not self.video:
            return
        x0, span, rowh, dur = self._tlg
        i, t = (e.y - 8) // rowh, (e.x - x0) / span * dur
        if 0 <= i < len(self.proj["actions"]):
            ivs = self.proj["intervals"][i]
            keep = [iv for iv in ivs if not (iv[0] <= t <= iv[1])]
            if len(keep) != len(ivs):
                self.proj["intervals"][i] = keep
                self.save_project()
                self.refresh_tree()

    # ─── вкладка 2: запекание ───

    def _build_bake(self, tab):
        self.bv = {k: tk.StringVar() for k in ("w", "h", "ctx", "act_ctx", "width", "steps", "batch", "rollout", "lr", "cl", "ct", "cr", "cb")}
        self.bv["color"] = tk.BooleanVar()
        self.out_var = tk.StringVar()
        top = ttk.Frame(tab)
        top.pack(fill="x", padx=8, pady=6)
        form = ttk.LabelFrame(top, text="Параметры")
        form.pack(side="left", fill="y")
        fields = [("w", "Ширина кадра (÷8)"), ("h", "Высота кадра (÷8)"), ("ctx", "Кадров контекста"), ("act_ctx", "Кадров кнопок"),
                  ("width", "Ширина сети"), ("steps", "Шагов обучения"), ("batch", "Батч"), ("rollout", "Rollout"), ("lr", "lr")]
        for r, (k, lbl) in enumerate(fields):
            ttk.Label(form, text=lbl).grid(row=r, column=0, sticky="w", padx=4)
            ttk.Entry(form, textvariable=self.bv[k], width=9).grid(row=r, column=1, padx=4, pady=1)
        ttk.Checkbutton(form, text="Цветное", variable=self.bv["color"]).grid(row=0, column=2, rowspan=2, padx=8)
        ttk.Label(form, text="Обрезка (доли): лево / верх / право / низ").grid(row=len(fields), column=0, columnspan=3, sticky="w", padx=4, pady=(6, 0))
        cr = ttk.Frame(form)
        cr.grid(row=len(fields) + 1, column=0, columnspan=3, sticky="w", padx=4, pady=2)
        for k in ("cl", "ct", "cr", "cb"):
            e = ttk.Entry(cr, textvariable=self.bv[k], width=6)
            e.pack(side="left", padx=2)
            self.bv[k].trace_add("write", lambda *a: self.show_frame())
        ttk.Label(form, text="Устройство").grid(row=len(fields) + 2, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(form, textvariable=self.dev_var, values=["auto", "gpu", "cpu"], width=7).grid(row=len(fields) + 2, column=1, padx=4)

        right = ttk.Frame(top)
        right.pack(side="left", fill="both", expand=True, padx=10)
        orow = ttk.Frame(right)
        orow.pack(fill="x")
        ttk.Label(orow, text="Файл модели:").pack(side="left")
        ttk.Entry(orow, textvariable=self.out_var).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(orow, text="…", width=3, command=self.ask_out, takefocus=False).pack(side="left")
        brow = ttk.Frame(right)
        brow.pack(fill="x", pady=8)
        self.btn_bake = ttk.Button(brow, text="Запечь", command=self.start_bake, takefocus=False)
        self.btn_bake.pack(side="left")
        self.btn_stop = ttk.Button(brow, text="Стоп (сохранить что есть)", command=self.stop_bake, state="disabled", takefocus=False)
        self.btn_stop.pack(side="left", padx=6)
        self.pb = ttk.Progressbar(right, maximum=100)
        self.pb.pack(fill="x")
        self.lbl_prog = ttk.Label(right, text="—")
        self.lbl_prog.pack(anchor="w", pady=2)
        self.logbox = tk.Text(right, height=14, state="disabled", wrap="word")
        self.logbox.pack(fill="both", expand=True)
        self.lbl_prev = ttk.Label(tab, text="Превью появится после первых 1000 шагов: сверху настоящее видео, снизу — что рисует модель сама")
        self.lbl_prev.pack(fill="x", padx=8, pady=4)

    def ask_out(self):
        p = filedialog.asksaveasfilename(title="Модель", defaultextension=".wm", filetypes=[("Модель мира", "*.wm")])
        if p:
            self.out_var.set(p)

    def log(self, s):
        self.logbox.config(state="normal")
        self.logbox.insert("end", s + "\n")
        self.logbox.see("end")
        self.logbox.config(state="disabled")

    def _bake_cfg(self):
        v = self.bv
        W, H = num(v["w"], int, "Ширина кадра"), num(v["h"], int, "Высота кадра")
        if W % 8 or H % 8 or W < 16 or H < 16:
            raise ValueError("Ширина и высота кадра должны делиться на 8 и быть не меньше 16 (например, 96×56)")
        crop = [num(v[k], float, f"Обрезка {k}") for k in ("cl", "ct", "cr", "cb")]
        if any(x < 0 or x >= 1 for x in crop) or crop[0] + crop[2] >= 0.95 or crop[1] + crop[3] >= 0.95:
            raise ValueError("Обрезка: доли от 0 до 1, сумма по каждой оси меньше 0.95")
        return dict(size=[W, H], color=bool(v["color"].get()), ctx=num(v["ctx"], int, "Кадров контекста"),
                    act_ctx=num(v["act_ctx"], int, "Кадров кнопок"), width=num(v["width"], int, "Ширина сети"),
                    steps=num(v["steps"], int, "Шагов обучения"), batch=num(v["batch"], int, "Батч"),
                    rollout=num(v["rollout"], int, "Rollout"), lr=num(v["lr"], float, "lr"), crop=crop)

    def start_bake(self):
        if self.worker and self.worker.is_alive():
            return
        if not self.video:
            self.err("Сначала откройте видео на вкладке «Разметка»")
            return
        acts = self.proj["actions"]
        if not acts:
            self.err("Нет ни одного действия. Добавьте действие и разметьте нажатия на вкладке «Разметка»")
            return
        for a in acts:
            if not a["key"]:
                self.err(f"У действия «{a['name']}» не назначена клавиша (вкладка «Разметка» → «Назначить клавишу»)")
                return
        try:
            cfg = self._bake_cfg()
        except ValueError as e:
            self.err(str(e))
            return
        self.proj["bake"] = cfg
        self.save_project()
        out = self.out_var.get().strip() or os.path.splitext(self.proj["video_path"])[0] + ".wm"
        self.out_var.set(out)
        project = json.loads(json.dumps(self.proj))
        video, dev = self.proj["video_path"], self.dev_var.get()
        self._prev_path, self._prev_mtime = os.path.splitext(out)[0] + ".preview.png", 0
        self.stop_flag = False
        self.btn_bake.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.pb.config(value=0)
        self.log(f"— запекание: {video} → {out}")

        def run():
            try:
                bakemod.bake(project, video, out, log=lambda s: self.q.put(("log", s)),
                             progress=lambda s, n, l, eta: self.q.put(("prog", s, n, l, eta)),
                             should_stop=lambda: self.stop_flag, device=dev, preview_path=self._prev_path)
                self.q.put(("done", out))
            except Exception:
                self.q.put(("err", traceback.format_exc()))

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def stop_bake(self):
        self.stop_flag = True
        self.btn_stop.config(state="disabled")

    def _poll(self):
        last = None
        try:
            while True:
                m = self.q.get_nowait()
                if m[0] == "log":
                    self.log(m[1])
                elif m[0] == "prog":
                    last = m
                elif m[0] == "done":
                    self._bake_finished(m[1])
                elif m[0] == "err":
                    self.btn_bake.config(state="normal")
                    self.btn_stop.config(state="disabled")
                    self.err("Ошибка при запекании:\n" + m[1])
        except queue.Empty:
            pass
        if last:
            _, s, n, loss, eta = last
            self.pb.config(value=100 * s / n)
            self.lbl_prog.config(text=f"шаг {s}/{n}   ошибка {loss:.4f}   осталось ~{int(eta // 60)} мин {int(eta % 60)} с")
        self._load_preview()
        self.root.after(150, self._poll)

    def _load_preview(self):
        p = self._prev_path
        if not p or not os.path.exists(p):
            return
        try:
            mt = os.path.getmtime(p)
            if mt == self._prev_mtime:
                return
            with Image.open(p) as im:
                im = im.convert("RGB")
            im.thumbnail((1080, 420))
            self._prev_photo = ImageTk.PhotoImage(im)
            self.lbl_prev.config(image=self._prev_photo, text="")
            self._prev_mtime = mt
        except Exception:
            pass  # файл мог быть записан не до конца — попробуем в следующий раз

    def _bake_finished(self, out):
        self.btn_bake.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status.set(f"Готово: {out}")
        self._load_preview()
        self.load_game(out)
        self.nb.select(2)

    # ─── вкладка 3: игра ───

    def _build_play(self, tab):
        bar = ttk.Frame(tab)
        bar.pack(fill="x", padx=8, pady=6)
        self.play_path = tk.StringVar()
        ttk.Button(bar, text="Открыть модель (.wm)…", command=self.ask_model, takefocus=False).pack(side="left")
        ttk.Entry(bar, textvariable=self.play_path, state="readonly").pack(side="left", fill="x", expand=True, padx=6)
        self.btn_game = ttk.Button(bar, text="⏸ Пауза", command=self.toggle_game, takefocus=False)
        self.btn_game.pack(side="left", padx=2)
        ttk.Button(bar, text="Новый старт", command=self.reset_game, takefocus=False).pack(side="left", padx=2)
        self.gfps = tk.StringVar(value="30")
        ttk.Label(bar, text="к/с:").pack(side="left", padx=(8, 0))
        ttk.Spinbox(bar, textvariable=self.gfps, from_=1, to=120, width=5).pack(side="left")
        self.gcv = tk.Canvas(tab, bg="#000", width=640, height=360, highlightthickness=0)
        self.gcv.pack(pady=6)
        self.ind_frame = ttk.Frame(tab)
        self.ind_frame.pack(pady=4)
        self.ind = []
        self.ginfo = ttk.Label(tab, text="Откройте файл .wm или запеките модель на вкладке 2")
        self.ginfo.pack()
        ttk.Label(tab, foreground="#666", wraplength=900, text=(
            "Это не видео, а генерация: модель рисует каждый следующий кадр по вашим нажатиям. "
            "Если картинка поплыла — «Новый старт» берёт другой кадр из исходного видео.")).pack(pady=4)

    def ask_model(self):
        p = filedialog.askopenfilename(title="Модель", filetypes=[("Модель мира", "*.wm"), ("Все файлы", "*.*")])
        if p:
            self.load_game(p)

    def load_game(self, path):
        self.status.set(f"Загружаю модель {path}…")
        self.root.update_idletasks()
        try:
            from wm import Player
            p = Player(path, self.dev_var.get())
        except Exception as e:
            self.err(f"Модель {path}: {e}")
            return
        self.player = p
        self.play_path.set(path)
        meta = p.meta
        self.game_keys = [a["key"].lower() for a in meta["actions"]]
        self.gfps.set(f"{meta['fps']:g}")
        for w in self.ind:
            w.destroy()
        self.ind = []
        for a in meta["actions"]:
            lab = tk.Label(self.ind_frame, text=f"{a['name']} [{a['key']}]", bg="#333", fg="white", padx=10, pady=4)
            lab.pack(side="left", padx=3)
            self.ind.append(lab)
        W, H = meta["size"]
        self.gz = max(1, min(900 // W, 520 // H))
        self.gcv.config(width=W * self.gz, height=H * self.gz)
        self.ginfo.config(text=f"{meta.get('name', '')}: {meta['frames']} кадров, обучено {meta['steps']} шагов, ошибка {meta['loss']} (копирование кадра: {meta['baseline']})")
        self.draw_game(p.last_image())
        self.status.set(f"Модель загружена: {path}")
        self.game_run = self.nb.index(self.nb.select()) == 2
        self._sync_game_btn()
        if self.game_run:
            self._game_kick()

    def _sync_game_btn(self):
        self.btn_game.config(text="⏸ Пауза" if self.game_run else "▶ Играть")

    def toggle_game(self):
        if not self.player:
            return
        self.game_run = not self.game_run
        self._sync_game_btn()
        self.root.focus_set()
        if self.game_run:
            self._game_kick()

    def reset_game(self):
        if self.player:
            self.player.reset()
            self.draw_game(self.player.last_image())
            self.root.focus_set()

    def _game_kick(self):
        if self._game_job is None:
            self._last_t = time.perf_counter()
            self._game_job = self.root.after(1, self._game_step)

    def _game_step(self):
        self._game_job = None
        if not (self.game_run and self.player):
            return
        t0 = time.perf_counter()
        vec = [1 if k in self.keys.down else 0 for k in self.game_keys]
        try:
            img = self.player.step(vec)
        except Exception:
            self.game_run = False
            self._sync_game_btn()
            self.err("Ошибка при генерации кадра:\n" + traceback.format_exc())
            return
        self.draw_game(img)
        for lab, on in zip(self.ind, vec):
            lab.config(bg="#4caf50" if on else "#333")
        now = time.perf_counter()
        self._fps_ema = 0.9 * self._fps_ema + 0.1 / max(now - self._last_t, 1e-3)
        self._last_t = now
        self._gn += 1
        if self._gn % 15 == 0:
            self.status.set(f"Генерация: {self._fps_ema:.1f} к/с")
        try:
            target = max(1.0, float(self.gfps.get().replace(",", ".")))
        except ValueError:
            target = float(self.player.meta["fps"])
        delay = max(1, int(1000 / target - (now - t0) * 1000))
        self._game_job = self.root.after(delay, self._game_step)

    def draw_game(self, img):
        z = getattr(self, "gz", 3)
        h, w = img.shape[:2]
        pil = Image.fromarray(img).resize((w * z, h * z), Image.NEAREST)
        self._gphoto = ImageTk.PhotoImage(pil)
        self.gcv.delete("all")
        self.gcv.create_image(0, 0, anchor="nw", image=self._gphoto)

    def close(self):
        self.stop_flag = True
        self.stop_play()
        self.save_project()
        self.root.destroy()


def main():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    if len(sys.argv) > 1:
        a = sys.argv[1]
        ext = os.path.splitext(a)[1].lower()
        if ext == ".wm":
            app.load_game(a)
            app.nb.select(2)
        elif ext == ".json":
            app.load_project(a)
        else:
            app.open_video(a)
    root.mainloop()


if __name__ == "__main__":
    main()