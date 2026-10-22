"""Модель мира (маленький U-Net) и формат файла .wm

.wm — обычный zip-архив (его можно открыть любым архиватором):
  meta.json      — настройки модели, размер кадра, fps, список кнопок (имя + клавиша), сведения об обучении
  weights.pt     — веса (torch state_dict)
  seeds.npy      — стартовые кадры из исходного видео, форма (S, ctx, C, H, W), uint8
  seed_acts.npy  — история кнопок перед этими стартами, форма (S, act_ctx, n_actions), uint8

Что делает модель:
  по последним ctx кадрам и последним act_ctx состояниям кнопок (зажата / нет)
  предсказывает следующий кадр.
"""
import io
import json
import random
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FORMAT_VERSION = 1


def _directml():
    """Видеокарта через DirectML (Radeon/Intel на Windows), если стоит пакет torch-directml."""
    try:
        import torch_directml
        return torch_directml.device()
    except Exception:
        return None


def pick_device(name="auto"):
    """Выбор устройства без ошибок: 'auto' / 'gpu' / 'cpu'.
    Порядок: CUDA (NVIDIA, а также AMD через ROCm) -> DirectML (Radeon на Windows) -> CPU.
    Если выбрано то, чего нет (например 'cuda' без CUDA-сборки torch), тихо берётся лучшее из доступного."""
    name = (name or "auto").lower()
    if name == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    d = _directml()
    if d is not None:
        return d
    return "cpu"


def device_label(dev):
    if dev == "cuda":
        try:
            return f"cuda ({torch.cuda.get_device_name(0)})"
        except Exception:
            return "cuda"
    if dev == "cpu":
        return "cpu"
    return f"directml ({dev})"


class Block(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.a = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.b = nn.Conv2d(c_out, c_out, 3, padding=1)

    def forward(self, x):
        return F.silu(self.b(F.silu(self.a(x))))


class WorldModel(nn.Module):
    def __init__(self, channels, n_actions, ctx=4, act_ctx=4, width=32, h=56, w=96):
        super().__init__()
        assert h % 8 == 0 and w % 8 == 0, "высота и ширина кадра должны делиться на 8"
        self.channels, self.n_actions, self.ctx, self.act_ctx = channels, n_actions, ctx, act_ctx
        self.h, self.w = h, w
        c = width
        a_dim = act_ctx * n_actions
        cin = ctx * channels + a_dim + 2   # кадры + плоскости кнопок + координаты x,y

        ys = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(1, 1, h, w)
        xs = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(1, 1, h, w)
        self.register_buffer("coords", torch.cat([xs, ys], 1).contiguous(), persistent=False)

        self.e1 = Block(cin, c)
        self.down1 = nn.Conv2d(c, 2 * c, 4, 2, 1)
        self.e2 = Block(2 * c, 2 * c)
        self.down2 = nn.Conv2d(2 * c, 3 * c, 4, 2, 1)
        self.e3 = Block(3 * c, 3 * c)
        self.down3 = nn.Conv2d(3 * c, 4 * c, 4, 2, 1)

        self.film = nn.Sequential(nn.Linear(a_dim, 128), nn.SiLU(), nn.Linear(128, 8 * c))
        nn.init.zeros_(self.film[2].weight)
        nn.init.zeros_(self.film[2].bias)
        self.mid1 = nn.Conv2d(4 * c, 4 * c, 3, padding=2, dilation=2)
        self.mid2 = nn.Conv2d(4 * c, 4 * c, 3, padding=2, dilation=2)

        self.up3 = nn.ConvTranspose2d(4 * c, 3 * c, 4, 2, 1)
        self.d3 = Block(6 * c, 3 * c)
        self.up2 = nn.ConvTranspose2d(3 * c, 2 * c, 4, 2, 1)
        self.d2 = Block(4 * c, 2 * c)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 4, 2, 1)
        self.d1 = Block(2 * c, c)
        self.out = nn.Conv2d(c, channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)   # старт: «копируем прошлый кадр»
        nn.init.zeros_(self.out.bias)

    def forward(self, frames, acts):
        """frames: (B, ctx, C, H, W) в [0,1];  acts: (B, act_ctx, n_actions) 0/1.  -> (B, C, H, W)"""
        B = frames.shape[0]
        x = frames.flatten(1, 2)
        a = acts.flatten(1)
        ap = a[:, :, None, None].expand(-1, -1, self.h, self.w)
        x = torch.cat([x, ap, self.coords.expand(B, -1, -1, -1)], 1)

        e1 = self.e1(x)
        e2 = self.e2(F.silu(self.down1(e1)))
        e3 = self.e3(F.silu(self.down2(e2)))
        m = F.silu(self.down3(e3))
        g, b = self.film(a).chunk(2, 1)
        m = m * (1 + g[:, :, None, None]) + b[:, :, None, None]
        m = m + F.silu(self.mid2(F.silu(self.mid1(m))))

        d3 = self.d3(torch.cat([F.silu(self.up3(m)), e3], 1))
        d2 = self.d2(torch.cat([F.silu(self.up2(d3)), e2], 1))
        d1 = self.d1(torch.cat([F.silu(self.up1(d2)), e1], 1))
        return frames[:, -1] + self.out(d1)


# ───────────────────────── формат .wm ─────────────────────────

def _npy_bytes(arr):
    b = io.BytesIO()
    np.save(b, arr)
    return b.getvalue()


def save_wm(path, model_cfg, model, meta, seeds, seed_acts):
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buf)
    full_meta = {"format": "wm", "version": FORMAT_VERSION, "model": model_cfg, **meta}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("meta.json", json.dumps(full_meta, ensure_ascii=False, indent=2))
        z.writestr("weights.pt", buf.getvalue())
        z.writestr("seeds.npy", _npy_bytes(seeds))
        z.writestr("seed_acts.npy", _npy_bytes(seed_acts))


def load_wm(path, device="cpu"):
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("meta.json"))
        state = torch.load(io.BytesIO(z.read("weights.pt")), map_location="cpu")
        seeds = np.load(io.BytesIO(z.read("seeds.npy")))
        seed_acts = np.load(io.BytesIO(z.read("seed_acts.npy")))
    if meta.get("format") != "wm":
        raise ValueError("Это не файл .wm")
    model = WorldModel(**meta["model"]).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, meta, seeds, seed_acts


# ───────────────────────── игра ─────────────────────────

class Player:
    """Запускает обученную модель как «игру»: каждый step() — один кадр."""

    def __init__(self, path, device="auto"):
        self.device = pick_device(device)
        self.model, self.meta, self.seeds, self.seed_acts = load_wm(path, self.device)
        self.n = self.meta["model"]["n_actions"]
        self.color = self.meta["model"]["channels"] == 3
        self.reset()

    def reset(self):
        k = random.randrange(len(self.seeds))
        self.frames = torch.from_numpy(self.seeds[k]).float().div(255).unsqueeze(0).to(self.device)
        self.acts = torch.from_numpy(self.seed_acts[k]).float().unsqueeze(0).to(self.device)

    def _image(self, t):
        img = (t.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
        return img[:, :, 0] if img.shape[2] == 1 else img

    def last_image(self):
        return self._image(self.frames[0, -1])

    @torch.no_grad()
    def step(self, action_vec):
        a = torch.tensor(action_vec, dtype=torch.float32, device=self.device).view(1, 1, -1)
        self.acts = torch.cat([self.acts[:, 1:], a], 1)
        pred = self.model(self.frames, self.acts).clamp(0, 1)
        self.frames = torch.cat([self.frames[:, 1:], pred[:, None]], 1)
        return self._image(pred[0])