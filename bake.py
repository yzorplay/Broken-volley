"""Запекание: видео + разметка кнопок -> обученная модель .wm

Можно запускать из студии (кнопка «Запечь») или из консоли:
    python bake.py project.json video.mp4 -o model.wm [--device cuda]
"""
import argparse
import datetime
import json
import math
import os
import time

import numpy as np

DEFAULTS = dict(size=[96, 56], color=False, ctx=4, act_ctx=4, width=32,
                steps=8000, batch=48, rollout=3, lr=2e-3, crop=[0, 0, 0, 0])


# ───────────────────────── данные ─────────────────────────

def frames_from_video(video, fps, size, color, crop, log=print):
    """Читает видео, берёт кадры с частотой fps, режет по crop (доли слева/сверху/справа/снизу),
    уменьшает до size=(W,H). Возвращает uint8 (T, C, H, W). Кадр k соответствует времени k/fps."""
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео (нужен кодек, который понимает OpenCV; mp4/h264 обычно ок)")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W, H = size
    l, t, r, b = crop
    out, j, next_t, dt = [], 0, 0.0, 1.0 / fps
    while True:
        ok, img = cap.read()
        if not ok:
            break
        ts = j / src_fps
        j += 1
        if ts + 1e-9 < next_t:
            continue
        h0, w0 = img.shape[:2]
        img = img[int(h0 * t):max(int(h0 * (1 - b)), int(h0 * t) + 1),
                  int(w0 * l):max(int(w0 * (1 - r)), int(w0 * l) + 1)]
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        if color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[None]
        while ts + 1e-9 >= next_t:       # если fps видео ниже целевого — повторяем кадр
            out.append(img)
            next_t += dt
        if total and j % 500 == 0:
            log(f"  прочитано {j}/{total} кадров видео")
    cap.release()
    if not out:
        raise RuntimeError("В видео не нашлось ни одного кадра")
    return np.ascontiguousarray(np.stack(out))


def actions_from_project(project, T):
    """Интервалы удержания (секунды) -> массив (T, n_actions) из 0/1.
    latency > 0 означает «метки опаздывают на столько кадров»: берём метку из будущего."""
    fps = float(project["fps"])
    n = len(project["actions"])
    lat = float(project.get("latency", 0))
    t = (np.arange(T) + 0.5 + lat) / fps
    A = np.zeros((T, n), np.uint8)
    for k, ivs in enumerate(project["intervals"][:n]):
        for s, e in ivs:
            A[(t >= s) & (t < e), k] = 1
    return A


# ───────────────────────── превью ─────────────────────────

def _hwc(x):
    x = np.clip(x, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    return np.repeat(x, 3, axis=2) if x.shape[2] == 1 else x


def save_preview(path, real, pred01, act):
    """Верхняя строка — настоящее видео, нижняя — что нарисовала модель сама. Полоска сверху: кнопка зажата."""
    from PIL import Image
    N = real.shape[0]
    cols = list(range(0, N, 5))
    pred = (np.clip(pred01, 0, 1) * 255).round()
    W = real.shape[3]
    gap = np.full((1, 2, 3), 0, np.uint8)

    def row(frames):
        tiles = []
        for k in cols:
            bar = np.full((3, W, 3), 255 if act[k] else 45, np.uint8)
            tiles.append(np.concatenate([bar, _hwc(frames[k])], 0))
            tiles.append(np.repeat(gap, tiles[-1].shape[0], axis=0))
        return np.concatenate(tiles, 1)

    top, bot = row(real), row(pred)
    sep = np.full((6, top.shape[1], 3), 0, np.uint8)
    grid = np.concatenate([top, sep, bot], 0)
    scale = max(1, 160 // W)
    im = Image.fromarray(grid)
    if scale > 1:
        im = im.resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST)
    im.save(path)


# ───────────────────────── обучение ─────────────────────────

def bake(project, video_path, out_path, log=print, progress=None, should_stop=None,
         device="auto", preview_path=None):
    import torch
    from wm import WorldModel, save_wm, pick_device, device_label

    should_stop = should_stop or (lambda: False)
    progress = progress or (lambda *a, **k: None)
    cfg = {**DEFAULTS, **project.get("bake", {})}
    W, H = [int(v) for v in cfg["size"]]
    color = bool(cfg["color"])
    C = 3 if color else 1
    if W % 8 or H % 8:
        raise ValueError("Ширина и высота кадра должны делиться на 8")
    fps = float(project["fps"])
    n = len(project["actions"])
    ctx, Ka, R = int(cfg["ctx"]), int(cfg["act_ctx"]), int(cfg["rollout"])
    L = max(ctx, Ka)
    steps, B = int(cfg["steps"]), int(cfg["batch"])
    dev = pick_device(device)
    log(f"Устройство: {device_label(dev)}")
    if dev == "cpu":
        log(f"  ⚠ Видеокарта не найдена (torch {torch.__version__}), считаю на процессоре. "
            "NVIDIA: поставьте torch с CUDA. Radeon на Windows: pip install torch-directml (в отдельном окружении).")

    log("Читаю видео…")
    frames = frames_from_video(video_path, fps, (W, H), color, cfg["crop"], log)
    T = len(frames)
    log(f"Кадров: {T} ({T / fps:.1f} с при {fps:g} к/с), размер {W}x{H}, {'цвет' if color else 'ч/б'}")
    if T < 300:
        raise ValueError("Слишком короткое видео: нужно хотя бы ~300 кадров (10 секунд при 30 к/с)")

    A = actions_from_project(project, T)
    if not A.any():
        raise ValueError("В разметке нет ни одного нажатия")
    for k, a in enumerate(project["actions"]):
        log(f"  кнопка «{a['name']}»: зажата в {A[:, k].mean():.0%} кадров")

    horizon = 60
    T_val = max(horizon + L + 5, int(T * 0.08)) if T > 600 else 0
    T_train = T - T_val
    n_start = T_train - (L + R) + 1
    log(f"Обучение на {T_train} кадрах, проверка на {T_val}")

    mcfg = dict(channels=C, n_actions=n, ctx=ctx, act_ctx=Ka, width=int(cfg["width"]), h=H, w=W)
    model = WorldModel(**mcfg).to(dev)
    log(f"Параметров в модели: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} млн")
    use_amp = False
    if dev == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            use_amp = torch.cuda.is_bf16_supported()
        except Exception:
            use_amp = False
        log(f"  смешанная точность bf16: {'вкл' if use_amp else 'выкл'}")

    frames_t = torch.from_numpy(frames[:T_train]).to(dev)
    acts_t = torch.from_numpy(A[:T_train]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4, fused=(dev == "cuda"))
    ar = torch.arange(L + R, device=dev)

    def lr_at(s):
        warm = min(1.0, s / 200)
        cos = 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0)))
        return cfg["lr"] * warm * (0.03 + 0.97 * cos)

    def preview(tag=""):
        if not preview_path:
            return
        model.eval()
        lo = T_train if T_val else L
        hi = T - 2
        i = lo
        for k in range(lo, hi - horizon):
            if not A[k].any() and A[k + 4:k + 10].any():
                i = k
                break
        i = max(L, min(i, T - horizon - 2))
        with torch.no_grad():
            f = torch.from_numpy(frames[i - ctx + 1:i + 1]).float().div(255)[None].to(dev)
            a = torch.from_numpy(A[i - Ka:i]).float()[None].to(dev)
            outs = []
            for t in range(horizon):
                an = torch.from_numpy(A[i + t]).float().view(1, 1, -1).to(dev)
                a = torch.cat([a[:, 1:], an], 1)
                p = model(f, a).clamp(0, 1)
                f = torch.cat([f[:, 1:], p[:, None]], 1)
                outs.append(p[0].cpu().numpy())
        save_preview(preview_path, frames[i + 1:i + 1 + horizon], np.stack(outs), A[i:i + horizon, 0])
        model.train()

    log("Начинаю обучение…")
    model.train()
    t_start, ema, ema_base, done = time.time(), None, None, 0
    for step in range(1, steps + 1):
        if should_stop():
            log("Остановлено вручную, сохраняю то, что есть")
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        s0 = torch.randint(0, n_start, (B,), device=dev)
        idx = s0[:, None] + ar[None]
        Fw = frames_t[idx].float().div_(255)         # (B, L+R, C, H, W)
        Aw = acts_t[idx].float()                      # (B, L+R, n)
        cur = Fw[:, L - ctx:L]
        sigma = torch.rand(B, 1, 1, 1, 1, device=dev) * 0.03
        cur = cur + torch.randn_like(cur) * sigma     # шум на входе: меньше накопление ошибок
        R_now = 1 if step < 0.3 * steps else R        # сначала простой режим, потом «на своих кадрах»
        loss, base = 0.0, 0.0
        for r in range(R_now):
            a = Aw[:, L - Ka + r:L + r]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = model(cur, a)
            pred = pred.float()
            tgt = Fw[:, L + r]
            prev_true = Fw[:, L - 1 + r]
            w = 1 + 8 * ((tgt - prev_true).abs() > 0.04).float()   # где реально что-то изменилось — штраф больше
            loss = loss + (w * (pred - tgt).abs()).mean()
            if r == 0:
                base = (w * (prev_true - tgt).abs()).mean().detach()
            cur = torch.cat([cur[:, 1:], pred.detach().clamp(0, 1)[:, None]], 1)
        loss = loss / R_now
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        done = step
        lv = loss.detach()
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv
        ema_base = base if ema_base is None else 0.98 * ema_base + 0.02 * base
        if step % 50 == 0 or step == steps:
            ema_f, base_f = float(ema), float(ema_base)   # единственная синхронизация
            el = time.time() - t_start
            eta = el / step * (steps - step)
            progress(step, steps, ema_f, eta)
            if step % 500 == 0:
                log(f"шаг {step}/{steps}  ошибка {ema_f:.4f} (если просто копировать кадр: {base_f:.4f})  "
                    f"{step / el:.1f} шаг/с")
        if step % 1000 == 0:
            preview()

    preview()
    model.eval()

    # стартовые кадры для игры: самое начало + равномерно по видео
    pos = sorted(set([L] + [int(v) for v in np.linspace(L, T - 2, 15)]))
    seeds = np.stack([frames[i - ctx + 1:i + 1] for i in pos])
    seed_acts = np.stack([A[i - Ka:i] for i in pos])
    meta = {
        "name": project.get("name", "model"), "fps": fps, "size": [W, H], "color": color,
        "actions": [{"name": a["name"], "key": a["key"]} for a in project["actions"]],
        "frames": T, "steps": done, "loss": round(float(ema or 0), 5),
        "baseline": round(float(ema_base) if ema_base is not None else 0.0, 5),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "video": project.get("video_name", ""),
        "loss": round(float(ema) if ema is not None else 0.0, 5),
    }
    save_wm(out_path, mcfg, model, meta, seeds, seed_acts)
    log(f"Готово: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} МБ)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default="model.wm")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--preview", default="preview.png")
    args = ap.parse_args()
    with open(args.project, encoding="utf-8") as f:
        proj = json.load(f)
    bake(proj, args.video, args.out, device=args.device, preview_path=args.preview,
         progress=lambda s, n, l, eta: print(f"\r{s}/{n} loss {l:.4f} eta {eta:.0f}s", end="", flush=True))
    print()