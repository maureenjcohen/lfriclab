""" Impose an observed deep superrotation on an LFRic-Venus checkpoint.

Copies a checkpoint and replaces the zonal-mean zonal wind below BLEND_BOT
with a Pioneer Venus probe profile, blending linearly into the original winds
between BLEND_BOT and BLEND_TOP. Everything else in the checkpoint (theta,
rho, exner, vertical wind, eddies and meridional winds) is left unchanged.

A third case, 'probe_vpcm', takes the zonal-mean zonal wind from VPCM output
above BLEND_TOP instead of keeping the original, so the zonal mean is
replaced over the whole column.

Usage:  python3 haus_deep_wind_init.py <directory>

<directory> holds the checkpoint, the mesh, the diagnostic file written at
the checkpoint's time step, and the probe CSVs. One edited copy of the
checkpoint is written per latitude shape, to <directory>/<shape>/, under the
original file name so that a restart reads it without configuration changes.

How the winds are stored: horizontal W2 values are volume fluxes through the
vertical faces of the cells, one column of NLAYERS per mesh edge. A zonal
wind u = U(z) f(lat) passes a flux

    U(z) * [F(lat2) - F(lat1)] * (r_top**2 - r_bot**2) / 2

through an edge whose end nodes sit at lat1 and lat2, with F the integral of
f. That flux depends only on node latitudes, so the added flow has exactly
zero discrete divergence and needs no change to the mass or temperature
fields. Each edge's positive direction is fixed in the model, not by node
order, so its sign is found by fitting the checkpoint fluxes against the
diagnosed u and v at the same time step.
"""

# %%
import glob
import os
import shutil
import sys
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

## Model geometry, from the &extrusion and &planet namelists of the run.
RADIUS = 6.0518e6
DOMAIN_HEIGHT = 9.0e4
NLAYERS = 60

## Blend layer, in m: imposed winds below BLEND_BOT, original winds above
## BLEND_TOP, linear in height between.
BLEND_BOT = 55.0e3
BLEND_TOP = 62.0e3

## Pioneer Venus probe entry latitudes, in degrees.
PROBE_LAT = {'day': -31.3, 'night': -28.7}

## Latitude below which the 'flat60' shape is uniform, in degrees.
FLAT_LAT = 60.0

## Width of the latitude bins for the original zonal mean, in degrees.
BIN_WIDTH = 3.75

## VPCM output for the 'probe_vpcm' case: zonal wind from VPCM above the
## probes, joined to the 'flat60' probe winds across the blend layer.
## VPCM's vitu is westward positive and its heights come from geop / g.
VPCM_FILE = os.path.expanduser('~/repos/make_waves_2026/data/Xins4.nc')
VPCM_G = 8.87


def xyz(lon, lat):
    """ Unit vectors for longitude and latitude in degrees. """
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def F_flat60(phi):
    """ Integral of f = 1 below FLAT_LAT and cos(lat)/cos(FLAT_LAT) above. """
    p0 = np.deg2rad(FLAT_LAT)
    a = np.abs(phi)
    out = np.where(a <= p0, a, p0 + (np.sin(a) - np.sin(p0)) / np.cos(p0))
    return np.sign(phi) * out


def F_solidbody(phi):
    """ Integral of f = cos(lat). """
    return np.sin(phi)


SHAPES = {'flat60': F_flat60, 'solidbody': F_solidbody}


def probe_layer_means(path, zc):
    """ Westward wind of one probe averaged over each model layer, m/s.

    Layers holding no samples take the linear interpolation between the
    layers that do; layers above the probe's highest sample are NaN. """
    d = pd.read_csv(path)
    z = d['ALT(KM)'].to_numpy() * 1e3
    w = d['WEST'].to_numpy()
    dz = DOMAIN_HEIGHT / NLAYERS
    out = np.full(NLAYERS, np.nan)
    for k, c in enumerate(zc):
        m = np.abs(z - c) <= dz / 2
        if m.any():
            out[k] = w[m].mean()
    have = ~np.isnan(out)
    inside = (zc >= zc[have].min()) & (zc <= zc[have].max())
    out[inside] = np.interp(zc[inside], zc[have], out[have])
    return out


def target_profiles(directory, zc):
    """ U(z) for each shape, westward positive, m/s.

    'flat60' uses the probe winds as they are, since both probes lie inside
    the uniform band. 'solidbody' divides each by cos of its latitude to get
    the equatorial value. Each is the mean over the probes that cover the
    layer. """
    raw = {p: probe_layer_means(os.path.join(directory,
                                             f'cleaned_{p}_probe.csv'), zc)
           for p in PROBE_LAT}
    flat = np.stack(list(raw.values()))
    sb = np.stack([raw[p] / np.cos(np.deg2rad(lat))
                   for p, lat in PROBE_LAT.items()])
    covered = ~np.isnan(flat).all(0)
    out = {}
    for shape, a in (('flat60', flat), ('solidbody', sb)):
        out[shape] = np.full(NLAYERS, np.nan)
        out[shape][covered] = np.nanmean(a[:, covered], 0)
    return out


def vpcm_zonal_mean(zc):
    """ VPCM time- and zonal-mean zonal wind on model layer centres.

    Returns latitudes in degrees and u (NLAYERS, nlat), eastward positive.
    Each latitude is interpolated in its own zonal-mean geopotential
    height. """
    d = nc.Dataset(VPCM_FILE)
    lat = d['lat'][:].data
    u = -d['vitu'][:].data.mean(axis=(0, 3))
    z = d['geop'][:].data.mean(axis=(0, 3)) / VPCM_G
    out = np.stack([np.interp(zc, z[:, j], u[:, j])
                    for j in range(len(lat))], 1)
    return lat, out


def flux_integral(lat_deg, ubar, phi1, phi2):
    """ Integral over latitude of a zonal-mean wind between two latitudes.

    ubar is (NLAYERS, nlat) on lat_deg; the result is (nedge, NLAYERS), the
    flux through each edge per unit of (r_top**2 - r_bot**2) / 2. """
    fine = np.linspace(-np.pi / 2, np.pi / 2, 4001)
    ufine = np.stack([np.interp(fine, np.deg2rad(lat_deg), ubar[k])
                      for k in range(NLAYERS)])
    G = np.concatenate([np.zeros((NLAYERS, 1)),
                        np.cumsum(0.5 * (ufine[:, 1:] + ufine[:, :-1])
                                  * np.diff(fine), 1)], 1)
    return np.stack([np.interp(phi2, fine, G[k]) - np.interp(phi1, fine, G[k])
                     for k in range(NLAYERS)], 1)


def main(directory):
    ckfile = [f for f in glob.glob(os.path.join(directory, '*_*.nc'))
              if os.path.basename(f) not in ('lfric_diag.nc',
                                             'mesh_C24_MG.nc')]
    if len(ckfile) != 1:
        sys.exit(f'expected one checkpoint in {directory}, found {ckfile}')
    ckfile = ckfile[0]
    ck = nc.Dataset(ckfile)
    me = nc.Dataset(os.path.join(directory, 'mesh_C24_MG.nc'))
    dg = nc.Dataset(os.path.join(directory, 'lfric_diag.nc'))

    ## Vertical grid: layer interfaces, centres, and the integral of r dr
    ## over each layer.
    zf = np.linspace(0.0, DOMAIN_HEIGHT, NLAYERS + 1)
    zc = 0.5 * (zf[1:] + zf[:-1])
    rf = RADIUS + zf
    area = (rf[1:] ** 2 - rf[:-1] ** 2) / 2
    blend = np.clip((BLEND_TOP - zc) / (BLEND_TOP - BLEND_BOT), 0.0, 1.0)

    ## Horizontal dof columns: runs of NLAYERS values at one location.
    lat = ck['lat_checkpoint_W2'][:].data
    lon = ck['lon_checkpoint_W2'][:].data
    u = ck['u'][:].data.copy()
    key = np.round(lat, 6) * 1e4 + np.round(lon, 6)
    start = np.r_[0, np.flatnonzero(np.diff(key) != 0) + 1]
    length = np.diff(np.r_[start, len(u)])
    hstart = start[length == NLAYERS]
    idx = hstart[:, None] + np.arange(NLAYERS)[None, :]
    cols = u[idx]

    ## Mesh edge for each column, by edge midpoint.
    nx = me['C24_node_x'][:].data
    ny = me['C24_node_y'][:].data
    en = me['C24_edge_nodes'][:].data - me['C24_edge_nodes'].start_index
    mid = xyz(nx[en[:, 0]], ny[en[:, 0]]) + xyz(nx[en[:, 1]], ny[en[:, 1]])
    mid /= np.linalg.norm(mid, axis=1)[:, None]
    dist, ie = cKDTree(mid).query(xyz(lon[hstart], lat[hstart]))
    if dist.max() > 1e-5 or len(np.unique(ie)) != len(ie):
        sys.exit('checkpoint columns do not map one-to-one onto mesh edges')
    phi1 = np.deg2rad(ny[en[ie, 0]])
    phi2 = np.deg2rad(ny[en[ie, 1]])
    lam1 = np.deg2rad(nx[en[ie, 0]])
    lam2 = np.deg2rad(nx[en[ie, 1]])
    dlam = (lam2 - lam1 + np.pi) % (2 * np.pi) - np.pi

    ## Diagnosed u, v at the checkpoint's time step, averaged onto edges.
    step = int(os.path.basename(ckfile).split('_')[-1].split('.')[0])
    tlast = float(dg['time'][-1])
    print(f'checkpoint step {step}; last diagnostic time {tlast:.0f} s '
          f'= step x {tlast / step:.3f} s')
    dex = dg['Mesh2d_edge_x'][:].data
    dey = dg['Mesh2d_edge_y'][:].data
    efl = dg['Mesh2d_edge_face_links'][:].data \
        - dg['Mesh2d_edge_face_links'].start_index
    _, ide = cKDTree(xyz(dex, dey)).query(mid[ie])
    fa, fb = efl[ide, 0], efl[ide, 1]
    ud = dg['u'][-1].data
    vd = dg['v'][-1].data
    ue = 0.5 * (ud[:, fa] + ud[:, fb]).T
    ve = 0.5 * (vd[:, fa] + vd[:, fb]).T
    cphi = np.cos(0.5 * (phi1 + phi2))
    pred = (ue * (phi2 - phi1)[:, None]
            - ve * (cphi * dlam)[:, None]) * area[None, :]

    ## Sign of each edge's positive direction relative to pred.
    sign = np.sign((cols * pred).sum(1))
    big = np.abs(pred) > np.percentile(np.abs(pred), 50)
    agree = (np.sign(cols) == sign[:, None] * np.sign(pred))[big].mean()
    resid = np.sqrt(((cols - sign[:, None] * pred) ** 2).mean()
                    / (cols ** 2).mean())
    print(f'edge signs: agreement {agree:.5f}, relative rms residual '
          f'{resid:.3f}, {int((sign == 0).sum())} undetermined')
    if agree < 0.999 or (sign == 0).any():
        sys.exit('edge signs not determined reliably; stopping')

    ## Original zonal-mean zonal wind, binned by latitude, and its integral
    ## G(lat) on a fine grid, so its flux through an edge is
    ## area * [G(lat2) - G(lat1)].
    fy = dg['Mesh2d_face_y'][:].data
    edges = np.arange(-90.0, 90.0 + BIN_WIDTH, BIN_WIDTH)
    centres = 0.5 * (edges[1:] + edges[:-1])
    which = np.digitize(fy, edges) - 1
    ubar = np.stack([[ud[k, which == b].mean() for b in range(len(centres))]
                     for k in range(NLAYERS)])
    fine = np.linspace(-np.pi / 2, np.pi / 2, 4001)
    ufine = np.stack([np.interp(fine, np.deg2rad(centres), ubar[k])
                      for k in range(NLAYERS)])
    G = np.concatenate([np.zeros((NLAYERS, 1)),
                        np.cumsum(0.5 * (ufine[:, 1:] + ufine[:, :-1])
                                  * np.diff(fine), 1)], 1)
    dG = np.stack([np.interp(phi2, fine, G[k]) - np.interp(phi1, fine, G[k])
                   for k in range(NLAYERS)], 1)

    targets = target_profiles(directory, zc)
    for shape, Ffun in SHAPES.items():
        U = targets[shape]
        if np.isnan(U[blend > 0]).any():
            sys.exit(f'{shape}: probe profile misses layers below BLEND_TOP')
        U = np.where(blend > 0, U, 0.0)
        dF = Ffun(phi2) - Ffun(phi1)
        ## Model u is eastward positive, so westward superrotation is -U.
        new_flux = -U[None, :] * dF[:, None] * area[None, :]
        old_flux = dG * area[None, :]
        incr = sign[:, None] * blend[None, :] * (new_flux - old_flux)
        unew = u.copy()
        unew[idx] = cols + incr

        outdir = os.path.join(directory, shape)
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, os.path.basename(ckfile))
        shutil.copy(ckfile, outfile)
        with nc.Dataset(outfile, 'a') as out:
            out['u'][:] = unew
        changed = np.flatnonzero(unew != u)
        print(f'{shape}: wrote {outfile}; {len(changed)} of {len(u)} values '
              f'changed, all horizontal: '
              f'{np.isin(changed, idx.ravel()).all()}')
        np.savez(os.path.join(outdir, 'imposed_profile.npz'), z=zc, U=U,
                 blend=blend, ubar_old=ubar, lat_bins=centres)

    ## 'probe_vpcm': flat60 probe winds below BLEND_BOT, VPCM zonal mean
    ## above BLEND_TOP, linear between; replaces the original zonal mean at
    ## every level.
    vlat, uv = vpcm_zonal_mean(zc)
    phiv = np.deg2rad(vlat)
    f = np.where(np.abs(phiv) <= np.deg2rad(FLAT_LAT), 1.0,
                 np.cos(phiv) / np.cos(np.deg2rad(FLAT_LAT)))
    Up = np.where(blend > 0, targets['flat60'], 0.0)
    up = -Up[:, None] * f[None, :]
    ut = blend[:, None] * up + (1 - blend[:, None]) * uv
    new_flux = flux_integral(vlat, ut, phi1, phi2) * area[None, :]
    old_flux = dG * area[None, :]
    unew = u.copy()
    unew[idx] = cols + sign[:, None] * (new_flux - old_flux)
    outdir = os.path.join(directory, 'probe_vpcm')
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, os.path.basename(ckfile))
    shutil.copy(ckfile, outfile)
    with nc.Dataset(outfile, 'a') as out:
        out['u'][:] = unew
    changed = np.flatnonzero(unew != u)
    print(f'probe_vpcm: wrote {outfile}; {len(changed)} of {len(u)} values '
          f'changed, all horizontal: {np.isin(changed, idx.ravel()).all()}')
    np.savez(os.path.join(outdir, 'imposed_profile.npz'), z=zc, U=Up,
             blend=blend, ubar_target=ut, lat_target=vlat, ubar_old=ubar,
             lat_bins=centres)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
