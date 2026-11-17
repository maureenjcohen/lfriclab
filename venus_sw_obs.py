"""Published Venus shortwave values for checking the SOCRATES column model.

Each table is transcribed from the printed page. Units are those of the
source: wavelength in micrometres, altitude in km, flux in W m-2, heating
in K per Earth day. None marks a value the source leaves blank or one left
out, with the reason beside it.

Sources:
  Moroz et al. (1985), "Solar and thermal radiation in the Venus
    atmosphere", Adv. Space Res. 5(11), 197-232 (VIRA, chapter VI).
  Tomasko et al. (1980), "Measurements of the flux of sunlight in the
    atmosphere of Venus", J. Geophys. Res. 85(A13), 8167-8186.
"""

# Moroz et al. (1985) Table 6-1: monochromatic Bond albedo of Venus,
# normalised so that its integral over the solar spectrum is 0.76.
MOROZ_BOND_ALBEDO = 0.76
MOROZ_SPECTRAL_ALBEDO = [
    (0.32, 0.41), (0.34, 0.50), (0.36, 0.50), (0.38, 0.53), (0.40, 0.61),
    (0.42, 0.71), (0.44, 0.77), (0.46, 0.79), (0.48, 0.82), (0.50, 0.83),
    (0.55, 0.85), (0.60, 0.85), (0.80, 0.85), (1.00, 0.85), (1.20, 0.85),
    (1.50, 0.82), (1.75, 0.78), (2.20, 0.70), (2.40, 0.64), (2.85, 0.080),
    (3.00, 0.055), (3.20, 0.035), (3.40, 0.040), (3.60, 0.0555),
    (3.80, 0.064),
]

# Moroz et al. (1985) Table 6-4(b): solar flux at the surface, over the
# whole spectrum, from each probe's measurement reduced to a solar zenith
# angle of 0 and of 45 degrees.
# probe: (local flux, flux at 0 deg, flux at 45 deg)
MOROZ_SURFACE_FLUX = {
    'Venera 9': (95.0, 129.0, 73.0),
    'Venera 10': (60.0, 73.0, 41.0),
    'Venera 11': (78.0, 87.0, 49.0),
    'Pioneer Venus LP': (24.5, 104.0, 58.0),
    'Venera 13': (33.0, 47.0, 27.0),
    'Venera 14': (72.0, 102.0, 57.0),
}
# Average and its spread, at 0 and at 45 degrees.
MOROZ_SURFACE_FLUX_MEAN = {0.0: (90.0, 12.0), 45.0: (51.0, 7.0)}

# Moroz et al. (1985) Table 6-7: downward flux F and net flux dF (W m-2)
# against altitude, reduced to a solar zenith angle of 45 degrees.
# Columns and their wavelength ranges (micrometres):
MOROZ_FLUX_COLUMNS = {
    'Venera 11': (0.45, 1.20),
    'Venera 13': (0.45, 1.20),
    'Venera 14': (0.45, 1.20),
    'Venera average': (0.45, 1.20),
    'Pioneer Venus 0.4-1.0': (0.4, 1.0),
    'Pioneer Venus 0.4-1.8': (0.4, 1.8),
    'Model': (0.0, float('inf')),
}
MOROZ_FLUX_ZENITH = 45.0
MOROZ_FLUX_ALT = [80, 60, 55, 50, 48, 40, 30, 20, 10, 5, 0]
# Downward flux, one list per column, in the order of MOROZ_FLUX_ALT.
MOROZ_FLUX_DOWN = {
    'Venera 11': [None, 754, 573, 436, 343, 328, 289, 225, 144, 100, 48],
    'Venera 13': [None, 638, 500, 297, 257, 212, 183, 148, 95, 61, 24],
    'Venera 14': [None, 792, 658, 465, 460, 416, 351, 282, 183, 123, 51],
    'Venera average': [None, 728, 577, 399, 353, 315, 274, 218, 141, 95, 41],
    'Pioneer Venus 0.4-1.0': [None, 771, 638, 531, 467, 436, 387, 307, 189,
                              122, 48],
    'Pioneer Venus 0.4-1.8': [None, 960, 730, 565, 461, None, None, None,
                              None, None, None],
    'Model': [1810, 1000, 760, 590, 470, 450, 391, 290, 180, 120, 60],
}
# Net flux. The model value at 50 km is printed as 734, out of line with
# 150 at 55 km and 131 at 48 km, and is left out.
MOROZ_FLUX_NET = {
    'Venera 11': [None] * 11,
    'Venera 13': [None, 78, 70, 65, 63, 60, 56, 47, 32, 25, None],
    'Venera 14': [None, 95, 75, 64, 63, 54, 52, 52, 48, 44, None],
    'Venera average': [None, 86, 72, 64, 63, 57, 54, 50, 40, 35, None],
    'Pioneer Venus 0.4-1.0': [None, 110, 84, 86, 85, 85, 73, 60, 44, 41,
                              None],
    'Pioneer Venus 0.4-1.8': [None, 133, 125, 111, 107, None, None, None,
                              None, None, None],
    'Model': [440, 200, 150, None, 131, 122, 102, 76, 60, 56, 55],
}

# Moroz et al. (1985) Table 6-8: zonal, 24-hour mean solar heating (K per
# Earth day) of the baseline model of Tomasko et al. (1983), a model
# fitted to the Pioneer Venus flux measurements, not a measurement itself.
MOROZ_HEATING_LAT = [0, 20, 40, 60, 70, 80]
MOROZ_HEATING = [
    # altitude, heating at each latitude above, global mean
    (94.4, [76.100, 73.504, 65.526, 51.256, 40.606, 25.612], 65.772),
    (85.3, [18.812, 17.941, 15.391, 11.367, 8.914, 6.328], 15.773),
    (78.9, [8.844, 8.542, 7.579, 5.642, 4.030, 1.795], 7.491),
    (75.4, [6.190, 5.846, 4.760, 2.878, 1.749, 0.693], 4.847),
    (72.2, [3.004, 2.745, 2.015, 1.042, 0.597, 0.221], 2.181),
    (68.9, [7.937, 7.484, 6.063, 3.620, 2.154, 0.835], 6.180),
    (65.7, [7.183, 6.583, 4.893, 2.576, 1.481, 0.594], 5.243),
    (63.9, [6.208, 5.633, 4.083, 2.124, 1.231, 0.497], 4.454),
    (62.9, [5.178, 4.679, 3.370, 1.755, 1.020, 0.413], 3.697),
    (62.0, [4.273, 3.857, 2.775, 1.449, 0.843, 0.341], 3.048),
    (60.4, [2.336, 2.107, 1.515, 0.792, 0.461, 0.186], 1.665),
    (58.5, [1.437, 1.298, 0.936, 0.491, 0.286, 0.116], 1.027),
    (55.3, [0.278, 0.251, 0.180, 0.094, 0.055, 0.022], 0.198),
    (51.4, [0.140, 0.126, 0.091, 0.048, 0.028, 0.011], 0.100),
    (48.9, [0.073, 0.066, 0.048, 0.025, 0.015, 0.006], 0.059),
    (36.1, [0.011, 0.010, 0.007, 0.004, 0.002, 0.001], 0.008),
    (12.1, [0.001, 0.001, 0.001, 0.000, 0.000, 0.000], 0.001),
]

# Tomasko et al. (1980) Table 8: Pioneer Venus Large Probe fluxes, scaled
# according to their cloud model F. The probe entered at a solar zenith
# angle of 65.6 degrees.
# altitude, down and up flux at the probe site over 0.325-1.905 um,
# bolometric net flux at the probe site, and its global average.
TOMASKO_ZENITH = 65.6
TOMASKO_BAND = (0.325, 1.905)
TOMASKO_FLUX = [
    (64.3, 564.6, 490.9, 76.9, 60.3),
    (61.3, 467.1, 398.9, 74.2, 57.9),
    (57.5, 354.7, 303.2, 53.5, 41.6),
    (49.7, 256.3, 213.4, 43.3, 33.5),
    (47.9, 209.9, 167.1, 42.4, 32.7),
    (39.9, 194.1, 153.0, 41.9, 32.4),
    (31.4, 175.5, 140.9, 34.3, 26.5),
    (24.0, 151.9, 122.2, 29.4, 22.7),
    (16.2, 113.7, 89.6, 23.8, 18.4),
    (10.1, 84.4, 63.2, 21.4, 16.5),
    (5.0, 55.7, 35.5, 21.0, 16.3),
    (2.5, 40.0, 19.9, 21.3, 16.5),
    (0.1, 24.5, 3.7, 22.6, 16.7),
]

# Perez-Hoyos et al. (2018), "Venus upper clouds and the UV absorber from
# MESSENGER/MASCS observations", J. Geophys. Res. Planets 123, 145-162.
# Retrieval from equatorial spectra, 0.3-1.5 um, phase angle near 90 deg.
# Table 2, best fit: base altitude (km) and optical depth at 0.63 um of
# each cloud mode, with their uncertainties. Mode 2 is retrieved as
# negligible (optical depth 1e-5).
PEREZ_HOYOS_WAVELENGTH = 0.63
PEREZ_HOYOS_MODES = {
    # mode: (base altitude, its error, optical depth, its error)
    '1': (65.0, 2.0, 3.2, 0.2),
    '2': (56.0, 1.0, 1.0e-5, 0.0),
    "2'": (46.0, 2.0, 6.8, 0.4),
    '3': (55.0, 2.0, 7.5, 0.4),
}
# Cloud top: altitude where the optical depth from above reaches 1 at
# 0.63 um, mean over all spectra.
PEREZ_HOYOS_CLOUD_TOP = (75.0, 2.0)
# Total aerosol optical depth at 0.63 um, given as "around 25".
PEREZ_HOYOS_TOTAL_TAU = 25.0
# Imaginary refractive index of the UV absorber in mode 1 particles,
# described as a Gaussian in wavelength: centre and full width at half
# maximum (um). Theirs, and that of the prior they started from (Pollack
# et al. 1980), whose width is given as "slightly below 100 nm".
PEREZ_HOYOS_ABSORBER = (0.34, 0.14)
POLLACK_1980_ABSORBER = (0.33, 0.10)

# Crisp (1986), "Radiative forcing of the Venus mesosphere. I. Solar fluxes
# and heating rates", Icarus 67, 484-514, Table IV: absorption efficiency
# Q_abs of the UV absorber per wavelength interval (lower and upper edge,
# um). Zero outside 0.31-0.78 um.
CRISP_UV_QABS = [
    (0.31, 0.32, 0.333), (0.32, 0.33, 0.356), (0.33, 0.34, 0.300),
    (0.34, 0.35, 0.247), (0.35, 0.36, 0.223), (0.36, 0.37, 0.202),
    (0.37, 0.38, 0.184), (0.38, 0.39, 0.155), (0.39, 0.40, 0.111),
    (0.40, 0.42, 0.080), (0.42, 0.44, 0.053), (0.44, 0.46, 0.034),
    (0.46, 0.48, 0.031), (0.48, 0.50, 0.028), (0.50, 0.52, 0.028),
    (0.52, 0.54, 0.023), (0.54, 0.56, 0.018), (0.56, 0.58, 0.014),
    (0.58, 0.60, 0.008), (0.60, 0.62, 0.004), (0.62, 0.78, 0.003),
]
