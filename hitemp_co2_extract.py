"""One streaming pass over 02_HITEMP2024.par.bz2 (sorted by wavenumber, 161-byte records):
writes filtered CO2 lines for given wavenumber ranges and tabulates 100 cm-1 bin sums.

Filter as the recipe: isotopologues A and B dropped; kept if S >= 1e-30 cm/molecule at 296 K
or 750 K. S(T) uses each isotopologue's own Q(T) from SOCRATES parsum.dat.
Bins: line count, kept count, summed S at 735 K and 230 K, all lines and kept lines.

Usage: python3 -I hitemp_co2_extract.py <HITEMP .par.bz2> <parsum.dat> <out dir>
                  <lo-hi> [<lo-hi> ...] --bins <lo-hi>
"""
import bz2, os, sys
import numpy as np

C2, SMIN = 1.4387769, 1e-30
args = sys.argv[1:]
src, parsum_f, outdir = args[:3]
ib = args.index("--bins")
ranges = [tuple(float(x) for x in r.split("-")) for r in args[3:ib]]
blo, bhi = (float(x) for x in args[ib + 1].split("-"))
nu_stop = max([r[1] for r in ranges] + [bhi])

# Isotopologue code (HITRAN column 3) -> parsum column; A, B are dropped
codes = b"1234567890"
cols = ["CO2_7", "CO2_8", "CO2_9", "CO2_10", "CO2_11", "CO2_12", "CO2_13", "CO2_14",
        "CO2_121", "CO2_15"]
hdr = open(parsum_f).readline().split()
tab = np.loadtxt(parsum_f, skiprows=1, usecols=[0] + [hdr.index(c) for c in cols])
tab = tab[tab[:, 0] <= 1000]
def qratio(t):
    """Q(296)/Q(t) per isotopologue, indexed by position in `codes`."""
    return np.array([np.interp(296.0, tab[:, 0], tab[:, j]) / np.interp(t, tab[:, 0], tab[:, j])
                     for j in range(1, tab.shape[1])])
QR = {t: qratio(t) for t in (230.0, 735.0, 750.0)}

def s_at(s, e, v, iso, t):
    return (s * QR[t][iso] * np.exp(-C2 * e * (1 / t - 1 / 296.0))
            * (1 - np.exp(-C2 * v / t)) / (1 - np.exp(-C2 * v / 296.0)))

os.makedirs(outdir, exist_ok=True)
outs = [open(f"{outdir}/hitemp2024_co2_{lo:.0f}-{hi:.0f}_filtered.par", "wb") for lo, hi in ranges]
nb = int(round((bhi - blo) / 100))
bins = np.zeros((nb, 6))   # n, kept, S735 all, S735 kept, S230 all, S230 kept
n_ab = 0
REC = 161
CHUNK = REC * 200000
lut = np.full(256, -1, dtype=np.int64)
for k, c in enumerate(codes):
    lut[c] = k

with bz2.open(src, "rb") as f:
    rest = b""
    while True:
        buf = f.read(CHUNK)
        if not buf and not rest:
            break
        buf = rest + buf
        n = len(buf) // REC
        rest = buf[n * REC:]
        if n == 0:
            break
        a = np.frombuffer(buf[:n * REC], dtype="S161")
        assert all(x[-1:] == b"\n" for x in a[[0, -1]]), "record length is not 161 bytes"
        raw = np.frombuffer(buf[:n * REC], dtype=np.uint8).reshape(n, REC)
        v = np.array([float(x[3:15]) for x in a])
        if v[0] > nu_stop:
            break
        want = np.zeros(n, bool)
        for lo, hi in ranges:
            want |= (v >= lo) & (v < hi)
        want |= (v >= blo) & (v < bhi)
        if not want.any():
            continue
        idx = np.where(want)[0]
        sub = a[idx]; vs = v[idx]
        iso = lut[raw[idx, 2]]
        n_ab += int(np.sum(iso < 0))
        ok = iso >= 0
        sub, vs, iso = sub[ok], vs[ok], iso[ok]
        s = np.array([float(x[15:25]) for x in sub])
        e = np.maximum(np.array([float(x[45:55]) for x in sub]), 0.0)
        keep = (s >= SMIN) | (s_at(s, e, vs, iso, 750.0) >= SMIN)
        for (lo, hi), fo in zip(ranges, outs):
            m = keep & (vs >= lo) & (vs < hi)
            if m.any():
                fo.write(b"".join(sub[m]))
        m = (vs >= blo) & (vs < bhi)
        if m.any():
            j = ((vs[m] - blo) // 100).astype(int)
            s735, s230, kk = s_at(s[m], e[m], vs[m], iso[m], 735.0), s_at(s[m], e[m], vs[m], iso[m], 230.0), keep[m]
            for col, w in enumerate([np.ones_like(s735), kk * 1.0, s735, s735 * kk, s230, s230 * kk]):
                bins[:, col] += np.bincount(j, weights=w, minlength=nb)
        print(f"  reached {v[-1]:9.2f} cm-1", flush=True)
for fo in outs:
    fo.close()
print(f"isotopologue A/B lines dropped (in wanted ranges): {n_ab}")
np.savetxt(f"{outdir}/hitemp2024_co2_bins_{blo:.0f}-{bhi:.0f}.txt",
           np.column_stack([blo + 100 * np.arange(nb), bins]),
           header="bin_lo n kept S735_all S735_kept S230_all S230_kept  (HITEMP 2024 CO2, own-isotopologue Q)",
           fmt=["%.0f", "%.0f", "%.0f", "%.4e", "%.4e", "%.4e", "%.4e"])
