#!/usr/bin/env python3
"""Compares SOCRATES Venus shortwave column runs with published values.

Each SWEEP is a directory written by run_column_sweep (SOCRATES fork,
examples/venus_sw): one subdirectory per solar zenith angle, each holding
the column output as CDL text (new.vflx, new.uflx, new.nflx, new.hrts,
new.t, new.szen). Several sweeps, e.g. two spectral files on the same
inputs, are drawn side by side.

Means over the planet are formed from the sweep by integrating over
mu = cos(zenith angle), with every quantity zero at mu = 0 and linear in mu
between runs:
  global mean        0.5 * integral of Q(mu) over mu from 0 to 1;
  zonal 24-hour mean (1/pi) * integral of Q(cos(lat) cos(h)) over hour
                     angle h from 0 to pi/2 (zero obliquity).
Fluxes over a wavelength range sum the bands inside it, with a band that
straddles an end weighted by the fraction of its width inside.
Altitudes come from hydrostatic balance on the column's own temperatures,
with the gas constant and gravity of the column's planet.nml.

Figures (published values from venus_sw_obs.py):
  sw_val_albedo.png    Bond albedo per band against Moroz et al. (1985)
                       Table 6-1.
  sw_val_heating.png   global and zonal mean heating against the model of
                       Tomasko et al. (1983), Moroz et al. (1985) Table 6-8.
  sw_val_venera.png    downward and net flux at 45 degrees, per wavelength
                       range, against Moroz et al. (1985) Table 6-7.
  sw_val_pioneer.png   flux at the Pioneer Venus site (65.6 degrees) and the
                       global-mean net flux against Tomasko et al. (1980)
                       Table 8.
  sw_val_surface.png   downward flux at the surface against zenith angle,
                       with Moroz et al. (1985) Table 6-4(b).
With --spectral-file (one per sweep) and --venus-sw:
  sw_val_cloud.png     cloud optical depth at 0.63 um accumulated from the
                       top, per mode, and each mode's total, against the
                       retrieval of Perez-Hoyos et al. (2018), Table 2, and
                       its cloud top (optical depth 1 from above).
  sw_val_absorber.png  spectral shape of the UV absorber, each scaled to
                       its own peak: Crisp (1986) Table IV absorption
                       efficiency, the "soluble" slot of each spectral file,
                       and the Gaussian imaginary refractive index of
                       Perez-Hoyos et al. (2018) and of their prior (Pollack
                       et al. 1980). Absorption efficiency and imaginary
                       index are not proportional for particles of this
                       size, so only the shapes are compared.
The cloud profiles are the mk_haus_cloud files (haus_mode1.soot,
haus_mode2.sulph, haus_modep.oceanic, haus_mode3.ash) in the --venus-sw
directory (SOCRATES fork, examples/venus_sw), on the pressure layers of
the column. A layer's optical depth is its mixing ratio times its mass per
unit area times the slot's mass extinction in the band holding 0.63 um.

Printed: Bond albedo, global-mean absorbed flux, and the fraction of it
absorbed above 64 km (about half, by Tomasko et al. 1980); with the
spectral files, each mode's optical depth and the cloud top.

Usage:
  sw_column_validation.py SWEEP [SWEEP ...] [--label NAME ...]
      [--spectral-file FILE ...] [--venus-sw DIR] [--savedir DIR]
      [--suffix TEXT]
--suffix is added to every figure name (sw_val_albedo_TEXT.png), so that
figures from different runs sit side by side.
Needs numpy and matplotlib.
"""
import argparse
import glob
import math
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import venus_sw_obs as obs

# planet.nml of the column runs.
R_GAS = 191.354   # J kg-1 K-1
GRAV = 8.87       # m s-2

TRAPZ = getattr(np, 'trapezoid', None) or np.trapz

# Cloud modes: name, profile file and its variable, and the aerosol slot of
# the spectral file that holds the mode.
CLOUD_SLOTS = [
    ('1', 'haus_mode1.soot', 'soot', 'Soot'),
    ('2', 'haus_mode2.sulph', 'sulph', 'Sulphuric Acid'),
    ("2'", 'haus_modep.oceanic', 'ocn', 'Oceanic'),
    ('3', 'haus_mode3.ash', 'ash', 'Volcanic Ash'),
]
ABSORBER_SLOT = 'Water soluble'

# Added to every figure name; set from --suffix.
FIGURE_SUFFIX = ''


def read_cdl(path):
    """Variables of the data section of a CDL file, as numpy arrays."""
    text = open(path).read()
    data = text[text.index('data:') + 5:]
    return {name: np.array([float(x) for x in body.replace(',', ' ').split()])
            for name, body in re.findall(r'(\w+)\s*=\s*([^;]*);', data)}


def load_run(directory):
    run = {}
    for ext in ('vflx', 'uflx', 'nflx', 'hrts'):
        cdl = read_cdl(os.path.join(directory, 'new.' + ext))
        n_band = len(cdl['channel'])
        run[ext] = cdl[ext].reshape(n_band, -1)
        run[ext + '_p'] = cdl['plev']
        run['wl'] = np.column_stack([cdl['wl_short'], cdl['wl_long']]) * 1e6
    run['t'] = read_cdl(os.path.join(directory, 'new.t'))['t']
    run['szen'] = float(read_cdl(os.path.join(directory, 'new.szen'))['szen'][0])
    return run


def heights(p_lev, p_lay, t_lay):
    """Altitude (km) of levels and layers, up from the bottom level."""
    n = len(p_lay)
    z_lev = np.zeros(n + 1)
    z_lay = np.zeros(n)
    for i in range(n - 1, -1, -1):
        h = R_GAS * t_lay[i] / GRAV
        z_lay[i] = z_lev[i + 1] + h * math.log(p_lev[i + 1] / p_lay[i])
        z_lev[i] = z_lev[i + 1] + h * math.log(p_lev[i + 1] / p_lev[i])
    return z_lev / 1e3, z_lay / 1e3


def read_spectral_aerosols(path):
    """Band limits (um) and, per aerosol slot, mass absorption and
    scattering (m2 kg-1) per band, from a SOCRATES spectral file."""
    lines = open(path).read().splitlines()
    i = lines.index('Band        Lower limit         Upper limit') + 1
    wl = []
    while lines[i].split() and lines[i].split()[0].isdigit():
        _, lo, hi = lines[i].split()
        wl.append((float(lo) * 1e6, float(hi) * 1e6))
        i += 1
    slots = {}
    for j, line in enumerate(lines):
        m = re.match(r'Index of species =\s+\d+\s+(.*\S)', line)
        if not m:
            continue
        absorption, scattering = [], []
        for row in lines[j + 4:]:
            f = row.split()
            if not f or not f[0].isdigit():
                break
            absorption.append(float(f[1]))
            scattering.append(float(f[2]))
        slots[m.group(1)] = (np.array(absorption), np.array(scattering))
    return np.array(wl), slots


def band_weights(wl, lo, hi):
    """Fraction of each band's width inside [lo, hi] micrometres."""
    inside = np.clip(np.minimum(wl[:, 1], hi) - np.maximum(wl[:, 0], lo),
                     0.0, None)
    return inside / (wl[:, 1] - wl[:, 0])


class Sweep:
    """Column runs of one sweep, sorted by mu."""

    def __init__(self, directory, label):
        self.label = label
        runs = [load_run(d) for d in sorted(glob.glob(
            os.path.join(directory, 'sz*'))) if os.path.isdir(d)]
        if not runs:
            raise SystemExit('No sz* runs in %s' % directory)
        runs.sort(key=lambda r: math.cos(math.radians(r['szen'])))
        self.runs = runs
        self.mu = np.array([math.cos(math.radians(r['szen'])) for r in runs])
        r0 = runs[0]
        self.wl = r0['wl']
        self.z_lev, self.z_lay = heights(r0['vflx_p'], r0['hrts_p'], r0['t'])

    def field(self, ext, weights=None):
        """[run, level] band sum of a field, optionally band-weighted."""
        w = np.ones(len(self.wl)) if weights is None else weights
        return np.array([w @ r[ext] for r in self.runs])

    def field_per_band(self, ext, level):
        """[run, band] of a field at one level."""
        return np.array([r[ext][:, level] for r in self.runs])

    def at(self, zenith):
        """The run at a zenith angle (degrees)."""
        for r in self.runs:
            if abs(r['szen'] - zenith) < 0.01:
                return r
        raise SystemExit('%s has no run at %.2f degrees' % (self.label, zenith))

    def _with_zero(self, q):
        return np.concatenate([[0.0], self.mu]), \
            np.vstack([np.zeros((1,) + q.shape[1:]), q])

    def global_mean(self, q):
        """Global mean of q[run, ...] (zero on the night side)."""
        mu, q = self._with_zero(q)
        return 0.5 * TRAPZ(q, mu, axis=0)

    def zonal_mean(self, q, lat):
        """24-hour mean of q[run, level] at a latitude (degrees)."""
        mu, q = self._with_zero(q)
        h = np.linspace(0.0, 0.5 * math.pi, 181)
        mu_h = math.cos(math.radians(lat)) * np.cos(h)
        qh = np.array([np.interp(mu_h, mu, q[:, k]) for k in range(q.shape[1])])
        return TRAPZ(qh, h, axis=1) / math.pi


def save(fig, savedir, name):
    if FIGURE_SUFFIX:
        stem, ext = os.path.splitext(name)
        name = '%s_%s%s' % (stem, FIGURE_SUFFIX, ext)
    path = os.path.join(savedir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)


STYLES = ['-', '--', ':', '-.']


def plot_albedo(sweeps, savedir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s, ls in zip(sweeps, STYLES):
        up = s.global_mean(s.field_per_band('uflx', 0))
        down = s.global_mean(s.field_per_band('vflx', 0))
        a = np.where(down > 0, up / np.where(down > 0, down, 1.0), np.nan)
        total = up.sum() / down.sum()
        edges = np.append(s.wl[:, 0], s.wl[-1, 1])
        ax.stairs(a, edges, linestyle=ls, label='%s (Bond albedo %.3f)'
                  % (s.label, total))
    wl, alb = zip(*obs.MOROZ_SPECTRAL_ALBEDO)
    ax.plot(wl, alb, 'ko', ms=4, label='Moroz et al. 1985, Table 6-1 '
            '(Bond albedo %.2f)' % obs.MOROZ_BOND_ALBEDO)
    ax.set_xlim(0.175, 4.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Bond albedo')
    ax.legend(fontsize=8)
    save(fig, savedir, 'sw_val_albedo.png')


def plot_heating(sweeps, savedir):
    lats = obs.MOROZ_HEATING_LAT
    fig, axes = plt.subplots(1, len(lats) + 1, figsize=(16, 5), sharey=True)
    z_obs = [row[0] for row in obs.MOROZ_HEATING]
    panels = [('Global mean', None, [row[2] for row in obs.MOROZ_HEATING])]
    panels += [('%d°' % lat, lat, [row[1][i] for row in obs.MOROZ_HEATING])
               for i, lat in enumerate(lats)]
    for ax, (title, lat, q_obs) in zip(axes, panels):
        for s, ls in zip(sweeps, STYLES):
            hr = s.field('hrts')
            q = s.global_mean(hr) if lat is None else s.zonal_mean(hr, lat)
            ax.plot(np.where(q > 0, q, np.nan), s.z_lay, ls, label=s.label)
        ax.plot(q_obs, z_obs, 'ko', ms=4,
                label='Tomasko et al. 1983 model\n(Moroz et al. 1985, Table 6-8)')
        ax.set_xscale('log')
        ax.set_xlim(1e-3, 1e3)
        ax.set_title(title)
        ax.set_xlabel('Heating (K/day)')
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel('Altitude (km)')
    axes[0].legend(fontsize=7, loc='lower right')
    save(fig, savedir, 'sw_val_heating.png')


def plot_venera(sweeps, savedir):
    zen = obs.MOROZ_FLUX_ZENITH
    ranges = {}
    for col, rng in obs.MOROZ_FLUX_COLUMNS.items():
        ranges.setdefault(rng, []).append(col)
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    for (rng, cols), c in zip(ranges.items(), colours):
        name = 'all wavelengths' if math.isinf(rng[1]) \
            else '%.2f-%.2f µm' % rng
        for s, ls in zip(sweeps, STYLES):
            r = s.at(zen)
            w = band_weights(s.wl, *rng)
            for ax, ext in zip(axes, ('vflx', 'nflx')):
                ax.plot(w @ r[ext], s.z_lev, ls, color=c,
                        label='%s, %s' % (s.label, name))
        for col, m in zip(cols, markers):
            for ax, table in zip(axes, (obs.MOROZ_FLUX_DOWN,
                                        obs.MOROZ_FLUX_NET)):
                pts = [(v, z) for v, z in zip(table[col], obs.MOROZ_FLUX_ALT)
                       if v is not None]
                if pts:
                    v, z = zip(*pts)
                    ax.plot(v, z, m, color=c, mfc='none', label=col)
    axes[0].set_title('Downward flux, zenith angle 45°')
    axes[1].set_title('Net flux, zenith angle 45°')
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel('Flux (W m$^{-2}$)')
    axes[0].set_ylim(0, 90)
    axes[0].set_ylabel('Altitude (km)')
    # One legend for both panels: some columns have values in one only.
    entries = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(l, h)
    axes[1].legend(entries.values(), entries.keys(), fontsize=6,
                   loc='upper left')
    fig.suptitle('Moroz et al. 1985, Table 6-7 (open symbols)')
    save(fig, savedir, 'sw_val_venera.png')


def plot_pioneer(sweeps, savedir):
    z, down, up, net_site, net_glob = zip(*obs.TOMASKO_FLUX)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    for s, ls in zip(sweeps, STYLES):
        r = s.at(obs.TOMASKO_ZENITH)
        w = band_weights(s.wl, *obs.TOMASKO_BAND)
        axes[0].plot(w @ r['vflx'], s.z_lev, ls, color='C0',
                     label='%s, down' % s.label)
        axes[0].plot(w @ r['uflx'], s.z_lev, ls, color='C1',
                     label='%s, up' % s.label)
        axes[1].plot(r['nflx'].sum(axis=0), s.z_lev, ls, color='C0',
                     label='%s, probe site' % s.label)
        axes[1].plot(s.global_mean(s.field('nflx')), s.z_lev, ls,
                     color='C2', label='%s, global mean' % s.label)
    axes[0].plot(down, z, 'o', color='C0', mfc='none', label='Tomasko, down')
    axes[0].plot(up, z, 's', color='C1', mfc='none', label='Tomasko, up')
    axes[1].plot(net_site, z, 'o', color='C0', mfc='none',
                 label='Tomasko, probe site')
    axes[1].plot(net_glob, z, 's', color='C2', mfc='none',
                 label='Tomasko, global average')
    axes[0].set_title('%.3f-%.3f µm, zenith angle %.1f°'
                      % (obs.TOMASKO_BAND + (obs.TOMASKO_ZENITH,)))
    axes[1].set_title('Net flux, all wavelengths')
    for ax in axes:
        ax.set_xlabel('Flux (W m$^{-2}$)')
        ax.legend(fontsize=7)
    axes[0].set_ylim(0, 90)
    axes[0].set_ylabel('Altitude (km)')
    fig.suptitle('Tomasko et al. 1980, Table 8 (open symbols)')
    save(fig, savedir, 'sw_val_pioneer.png')


def plot_surface(sweeps, savedir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, ls in zip(sweeps, STYLES):
        zen = [r['szen'] for r in s.runs]
        ax.plot(zen, s.field('vflx')[:, -1], ls, label=s.label)
    markers = ['o', 's', '^', 'D', 'v', 'P']
    for (probe, (_, f0, f45)), m in zip(obs.MOROZ_SURFACE_FLUX.items(),
                                        markers):
        ax.plot([0.0, 45.0], [f0, f45], m, mfc='none', ls='none', label=probe)
    for zen, (mean, spread) in obs.MOROZ_SURFACE_FLUX_MEAN.items():
        ax.errorbar(zen, mean, yerr=spread, fmt='k*', ms=10, capsize=4,
                    label='Average' if zen == 0.0 else None)
    ax.set_xlabel('Solar zenith angle (°)')
    ax.set_ylabel('Downward flux at the surface (W m$^{-2}$)')
    ax.set_title('Moroz et al. 1985, Table 6-4(b), whole spectrum')
    ax.legend(fontsize=7)
    save(fig, savedir, 'sw_val_surface.png')


def cloud_optical_depth(sweep, wl, slots, venus_sw):
    """Optical depth per layer [mode, layer] at the band holding the
    Perez-Hoyos wavelength, and the band's limits."""
    band = int(np.flatnonzero((wl[:, 0] <= obs.PEREZ_HOYOS_WAVELENGTH)
                              & (wl[:, 1] > obs.PEREZ_HOYOS_WAVELENGTH))[0])
    p_lev = sweep.runs[0]['vflx_p']
    mass = np.diff(p_lev) / GRAV
    tau = []
    for _, fname, var, slot in CLOUD_SLOTS:
        q = read_cdl(os.path.join(venus_sw, fname))[var]
        if len(q) != len(mass):
            raise SystemExit('%s has %d layers, the column %d'
                             % (fname, len(q), len(mass)))
        absorption, scattering = slots[slot]
        tau.append(q * mass * (absorption[band] + scattering[band]))
    return np.array(tau), wl[band]


def cloud_top(z_lev, cum):
    """Altitude (km) where the accumulated optical depth reaches 1."""
    k = int(np.argmax(cum >= 1.0))
    f = (1.0 - cum[k - 1]) / (cum[k] - cum[k - 1])
    return z_lev[k - 1] + f * (z_lev[k] - z_lev[k - 1])


def plot_cloud(sweeps, spectral, venus_sw, savedir):
    modes = [m for m, _, _, _ in CLOUD_SLOTS]
    colours = ['C0', 'C1', 'C2', 'C3']
    fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                             gridspec_kw={'width_ratios': [3, 2]})
    ax = axes[0]
    width = 0.8 / (len(sweeps) + 1)
    for n, (s, (wl, slots), ls) in enumerate(zip(sweeps, spectral, STYLES)):
        tau, limits = cloud_optical_depth(s, wl, slots, venus_sw)
        cum = np.vstack([np.zeros(len(modes)), np.cumsum(tau.T, axis=0)]).T
        total = np.concatenate([[0.0], np.cumsum(tau.sum(axis=0))])
        for m, c, row in zip(modes, colours, cum):
            ax.plot(np.where(row > 0, row, np.nan), s.z_lev, ls, color=c,
                    label='%s, mode %s' % (s.label, m))
        ax.plot(np.where(total > 0, total, np.nan), s.z_lev, ls, color='k',
                label='%s, all modes' % s.label)
        top = cloud_top(s.z_lev, total)
        ax.plot(1.0, top, 'k' + 'oxs+'[n], ms=8,
                label='%s, cloud top %.1f km' % (s.label, top))
        axes[1].bar(np.arange(len(modes)) + n * width, tau.sum(axis=1),
                    width, label=s.label, hatch=['', '//', '..', 'xx'][n],
                    color=colours, edgecolor='k')
        print('%s: optical depth at %.3f-%.3f um by mode %s, total %.1f; '
              'cloud top %.1f km' % (s.label, limits[0], limits[1],
              ', '.join('%s %.2f' % (m, t) for m, t in
                        zip(modes, tau.sum(axis=1))), tau.sum(), top))
    z_top, dz_top = obs.PEREZ_HOYOS_CLOUD_TOP
    ax.axhspan(z_top - dz_top, z_top + dz_top, color='0.85',
               label='Perez-Hoyos et al. 2018, cloud top')
    for m, c in zip(modes, colours):
        z_b, dz_b, _, _ = obs.PEREZ_HOYOS_MODES[m]
        ax.errorbar(1e-3, z_b, yerr=dz_b, fmt='>', color=c, mfc='none',
                    capsize=3)
    ax.set_xscale('log')
    ax.set_xlim(1e-3, 1e2)
    ax.set_ylim(40, 90)
    ax.set_xlabel('Optical depth from the top, %.2f µm'
                  % obs.PEREZ_HOYOS_WAVELENGTH)
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Accumulated from the top (arrows: retrieved mode bases)')
    ax.legend(fontsize=6, loc='upper right')
    x = np.arange(len(modes)) + len(sweeps) * width
    tau_ph = [obs.PEREZ_HOYOS_MODES[m][2] for m in modes]
    err_ph = [obs.PEREZ_HOYOS_MODES[m][3] for m in modes]
    axes[1].bar(x, tau_ph, width, yerr=err_ph, capsize=3, color='w',
                edgecolor='k', label='Perez-Hoyos et al. 2018 (total %g)'
                % obs.PEREZ_HOYOS_TOTAL_TAU)
    axes[1].set_xticks(np.arange(len(modes)) + 0.4 - width / 2)
    axes[1].set_xticklabels(['mode %s' % m for m in modes])
    axes[1].set_ylabel('Optical depth, %.2f µm' % obs.PEREZ_HOYOS_WAVELENGTH)
    axes[1].set_title('Total per mode')
    axes[1].legend(fontsize=7)
    fig.suptitle('Cloud against Perez-Hoyos et al. 2018, Table 2')
    save(fig, savedir, 'sw_val_cloud.png')


def plot_absorber(sweeps, spectral, savedir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lo, hi, q = zip(*obs.CRISP_UV_QABS)
    edges = np.append(lo, hi[-1])
    ax.stairs(np.array(q) / max(q), edges, color='k', lw=2,
              label='Crisp 1986, Table IV: absorption efficiency')
    for s, (wl, slots), ls in zip(sweeps, spectral, STYLES):
        absorption = slots[ABSORBER_SLOT][0]
        if absorption.max() <= 0.0:
            ax.plot([], [], linestyle=ls, color='0.5',
                    label='%s spectral file: soluble slot, zero in every '
                    'band' % s.label)
            continue
        e = np.append(wl[:, 0], wl[-1, 1])
        ax.stairs(absorption / absorption.max(), e, linestyle=ls,
                  label='%s spectral file: soluble slot, absorption'
                  % s.label)
    lam = np.linspace(0.2, 0.8, 601)
    for (c, fwhm), name, style in (
            (obs.PEREZ_HOYOS_ABSORBER, 'Perez-Hoyos et al. 2018', 'C3-'),
            (obs.POLLACK_1980_ABSORBER, 'Pollack et al. 1980 (their prior)',
             'C3:')):
        sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        ax.plot(lam, np.exp(-0.5 * ((lam - c) / sigma) ** 2), style,
                label='%s: imaginary index, centre %.2f µm, FWHM %.2f µm'
                % (name, c, fwhm))
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Scaled to its own peak')
    ax.set_title('UV absorber: spectral shapes only')
    ax.legend(fontsize=7)
    save(fig, savedir, 'sw_val_absorber.png')


def report(sweeps):
    for s in sweeps:
        down = s.global_mean(s.field('vflx'))
        up = s.global_mean(s.field('uflx'))
        net = s.global_mean(s.field('nflx'))
        k64 = int(np.argmin(abs(s.z_lev - 64.3)))
        print('%s: Bond albedo %.3f; global-mean absorbed %.1f W m-2; '
              'fraction absorbed above %.1f km %.2f'
              % (s.label, up[0] / down[0], net[0], s.z_lev[k64],
                 (net[0] - net[k64]) / net[0]))


def default_savedir():
    try:
        import config
        return config.savedir
    except ImportError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('sweeps', nargs='+', help='run_column_sweep directories')
    ap.add_argument('--label', nargs='+', help='one name per sweep')
    ap.add_argument('--savedir', help='directory for the figures '
                    '(default: SAVEDIR from .env)')
    ap.add_argument('--spectral-file', nargs='+',
                    help='the spectral file of each sweep, for the cloud '
                    'and absorber figures')
    ap.add_argument('--venus-sw', help='SOCRATES fork examples/venus_sw, '
                    'holding the cloud profiles')
    ap.add_argument('--suffix', default='', help='added to figure names')
    args = ap.parse_args()
    global FIGURE_SUFFIX
    FIGURE_SUFFIX = args.suffix
    labels = args.label or [os.path.basename(os.path.normpath(d))
                            for d in args.sweeps]
    if len(labels) != len(args.sweeps):
        ap.error('give one label per sweep')
    savedir = args.savedir or default_savedir()
    if not savedir:
        ap.error('no --savedir, and no SAVEDIR from .env')
    os.makedirs(savedir, exist_ok=True)
    sweeps = [Sweep(d, l) for d, l in zip(args.sweeps, labels)]
    report(sweeps)
    plot_albedo(sweeps, savedir)
    plot_heating(sweeps, savedir)
    plot_venera(sweeps, savedir)
    plot_pioneer(sweeps, savedir)
    plot_surface(sweeps, savedir)
    if args.spectral_file:
        if len(args.spectral_file) != len(sweeps):
            ap.error('give one spectral file per sweep')
        spectral = [read_spectral_aerosols(f) for f in args.spectral_file]
        plot_absorber(sweeps, spectral, savedir)
        if args.venus_sw:
            plot_cloud(sweeps, spectral, args.venus_sw, savedir)


if __name__ == '__main__':
    main()
