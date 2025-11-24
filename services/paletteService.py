# services/palette.py
from core.autoclass import UNKNOWN, MULTIPLE

class PaletteService:
    def __init__(self, avoid_red=True, min_hue_gap_deg=18):
        self.avoid_red = avoid_red
        self.min_gap = min_hue_gap_deg/360.0
        self.phi = 0.61803398875

    def _hsv2rgb(self, h, s, v):
        i = int(h*6); f = h*6 - i
        p = int(255*v*(1-s)); q = int(255*v*(1-f*s)); t = int(255*v*(1-(1-f)*s)); vv = int(255*v)
        i %= 6
        return [(vv,t,p),(q,vv,p),(p,vv,t),(p,q,vv),(t,p,vv),(vv,p,q)][i]

    def _is_red_like(self, h):  # [345°,15°]
        return (h <= (15/360.0)) or (h >= (345/360.0))

    def make(self, class_ids: list[int]):
        # UNKNOWN/MULTIPLE 예약
        pal = {UNKNOWN:(0,0,0), MULTIPLE:(255,0,0)}
        used = []
        def too_close(h):
            from math import fmod
            for uh in used:
                d = abs((h-uh+0.5)%1.0-0.5)
                if d < self.min_gap: return True
            return False

        s, v = 0.65, 0.95
        for cid in sorted(cid for cid in class_ids if cid not in (UNKNOWN, MULTIPLE)):
            h = (cid * self.phi) % 1.0
            tries = 0
            while tries < 24 and ((self.avoid_red and self._is_red_like(h)) or too_close(h)):
                h = (h + self.phi) % 1.0; tries += 1
            used.append(h)
            pal[cid] = self._hsv2rgb(h, s, v)
        return pal
