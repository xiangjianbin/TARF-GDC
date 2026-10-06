"""Original tensor-rotation and relative-RMS definitions."""
import numpy as np
import json
COMPONENTS=("xx","xy","xz","yy","yz","zz")

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def relative_rms(a, b, axis):
    diff = np.sqrt(np.mean((a-b)**2, axis=axis))
    scale = np.sqrt(np.mean((a*a+b*b)/2, axis=axis))
    return np.divide(diff, scale, out=np.zeros_like(diff), where=scale>0), diff, scale

def rotate(a, angle):
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    xx, xy, xz, yy, yz, zz = a
    return np.stack((c*c*xx+s*s*yy-2*c*s*xy, c*s*(xx-yy)+(c*c-s*s)*xy,
                     c*xz-s*yz, s*s*xx+c*c*yy+2*c*s*xy, s*xz+c*yz, zz))
