""" Data organisation for LFRic-Venus output """

# %%
## Import packages
import os
import re
import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from lfric_wp1_diag import calc_temperature, calc_static_stability
from config import savedir

# %%
## Definition of LFRic-Venus model constants
## These mirror the &planet and &extrusion namelists of the Venus
## configuration, so that derived quantities stay consistent with the
## model that produced the output.
## Radius in m, gravity in m/s2, rotation rate in s-1 (negative: Venus is
## retrograde), specific heat and gas constant in SI units, reference
## pressure in Pa, domain height in m.
lfricdict = {'radius': 6.0518e6, 'g': 8.87, 'rotrate': -2.9924e-07,
             'cp': 900.0, 'rd': 191.4, 'p_zero': 9.2e6,
             'domain_height': 9.0e4, 'nlayers': 60,
             'name': 'Venus'}

## Names of the mesh topology variables in LFRic UGRID output.
meshdict = {'face_x': 'Mesh2d_face_x', 'face_y': 'Mesh2d_face_y',
            'edge_x': 'Mesh2d_edge_x', 'edge_y': 'Mesh2d_edge_y',
            'node_x': 'Mesh2d_node_x', 'node_y': 'Mesh2d_node_y',
            'face_nodes': 'Mesh2d_face_nodes',
            'face_edges': 'Mesh2d_face_edges'}

## Candidate names for each physical field. LFRic names diagnostics after
## the XIOS file definition, so a time-meaned run and an instantaneous run
## carry different names for the same quantity.
fielddict = {'u': ['u_mean', 'u_in_w2h', 'u_inst','u'],
             'v': ['v_mean', 'v_in_w2h', 'v_inst', 'v'],
             'theta': ['theta_mean', 'theta', 'theta_inst'],
             'exner': ['exner_mean', 'exner', 'exner_inst'],
             'rho': ['rho_mean', 'rho', 'rho_inst'],
             'dtheta_force': ['dtheta_force_mean', 'dtheta_force']}


### LFRicPlanet class object definition ###
### Manages planet configuration data and LFRic simulation output data.

# %%
class LFRicPlanet:
    """ A Planet object which contains the output data for an LFRic simulation.

    LFRic writes UGRID output on an unstructured cubed-sphere mesh, so the
    horizontal dimensions are cell indices (nMesh2d_face, nMesh2d_edge) rather
    than longitude and latitude. Geolocation is carried alongside in
    Mesh2d_face_x/y and Mesh2d_edge_x/y, both in degrees. This object exposes
    those coordinates and the mesh connectivity so that zonal means and global
    integrals can be taken on the native mesh without regridding. """

    def __init__(self, planetdict, model, run):
        """ Initiates an LFRicPlanet object using the input dictionary of
        planet constants, the name of the model, and the name of the run.
        Model names: lfric
        Run names: free text, e.g. idealised, C24_MG """

        self.name = planetdict['name']
        self.model = model
        self.run = run
        print(f'Welcome to Venus. Your lander will melt in 57 minutes.')
        print(f'This is the {self.run} dataset created by {self.model.upper()}')
        for key, value in planetdict.items():
            setattr(self, key, value)
        self.kappa = self.rd / self.cp

    def identify(self):
        print(f'This is the {self.run} dataset created by {self.model.upper()}')

    def load_file(self, fn):
        """ Loads a netCDF file using the xarray package and stores in object.
            Lists dictionary key, name, dimensions, and shape of each data cube
            and stores text in a reference list. """
        if isinstance(fn, str):
            ds = xr.open_dataset(fn, decode_cf=False)
        elif isinstance(fn, list):
            ds = xr.open_mfdataset(fn, combine='nested', concat_dim='time',
                                   decode_cf=False)
        else:
            print('Improper filename input, must be string or list')
        reflist = []
        str1 = 'File contains:'
        print(str1)
        reflist.append(str1)
        for key in ds.data_vars:
            if 'long_name' in ds[key].attrs:
                keystring = key + ': ' + ds[key].long_name + ', ' + \
                      str(ds[key].dims) + ', ' + \
                      str(ds[key].shape)
            else:
                keystring = key + ': ' + str(ds[key].dims) + ', ' \
                      + str(ds[key].shape)
            print(keystring)
            reflist.append(keystring)
        self.data = ds
        self.reflist = reflist

    def close(self):
        """ Closes netCDF file packaged in LFRicPlanet data object """
        self.data.close()
        print('LFRicPlanet object associated dataset has been closed')

    def contents(self):
        """ Prints reference list for easy formatted oversight of file contents"""
        print(*self.reflist, sep='\n')

    def key(self, field):
        """ Resolves a physical field name onto the variable actually present
        in the file, since XIOS names diagnostics after the file definition. """
        for candidate in fielddict[field]:
            if candidate in self.data.variables:
                return candidate
        raise KeyError(f'No variable in file for {field}; '
                       f'tried {fielddict[field]}')

    def set_resolution(self):
        """ Reads the mesh geolocation and connectivity from the file and
        derives the vertical coordinate from the extrusion namelist values.

        Face and edge latitudes are the scattered cell-centre and edge-midpoint
        positions of the cubed sphere, not a regular axis. """
        self.face_lons = self.data[meshdict['face_x']].values
        self.face_lats = self.data[meshdict['face_y']].values
        self.edge_lons = self.data[meshdict['edge_x']].values
        self.edge_lats = self.data[meshdict['edge_y']].values
        self.node_lons = self.data[meshdict['node_x']].values
        self.node_lats = self.data[meshdict['node_y']].values
        self.face_nodes = self.data[meshdict['face_nodes']].values
        self.face_edges = self.data[meshdict['face_edges']].values

        self.nface = self.face_lats.size
        self.nedge = self.edge_lats.size
        self.time = self.data['time'].values
        if self.time.size > 1:
            self.tinterval = float(np.diff(self.time[0:2])[0])
        else:
            self.tinterval = np.nan

        self.set_vertical()
        self.face_areas()
        print('Resolution is ' + str(self.nface) + ' faces, '
              + str(self.nedge) + ' edges, ' + str(self.nlayers) + ' levs')
        print(f'Vertical axis is height above surface, uniform '
              f'{self.dz:.1f} m layers')

    def set_vertical(self):
        """ Builds height coordinates for the uniform extrusion.

        full_levels and half_levels in the file are level indices, not heights.
        A uniform extrusion places full levels (Wtheta) at k*dz and half levels
        (W3) at (k+0.5)*dz. Heights in km are kept for plotting. """
        self.dz = self.domain_height / self.nlayers
        self.z_full = np.arange(self.nlayers + 1) * self.dz
        self.z_half = (np.arange(self.nlayers) + 0.5) * self.dz
        self.heights_full = self.z_full * 1e-3
        self.heights = self.z_half * 1e-3
        self.vert = self.nlayers
        self.vert_unit = 'km'
        self.vert_axis = 'Height'

    def face_areas(self):
        """ Calculates the spherical area of each mesh face.

        Cubed-sphere cells are not equal-area, so a global integral needs the
        true areas. Each quadrilateral face is split into two spherical
        triangles and the area found from the spherical excess. """
        lon = np.deg2rad(self.node_lons)
        lat = np.deg2rad(self.node_lats)
        xyz = np.stack([np.cos(lat) * np.cos(lon),
                        np.cos(lat) * np.sin(lon),
                        np.sin(lat)], axis=-1)
        corners = xyz[self.face_nodes]
        excess = (self._triangle_excess(corners[:, 0], corners[:, 1], corners[:, 2])
                  + self._triangle_excess(corners[:, 0], corners[:, 2], corners[:, 3]))
        self.areas = excess * self.radius**2
        return self.areas

    @staticmethod
    def _triangle_excess(a, b, c):
        """ Spherical excess of a triangle of unit vectors, by the formula
        E = 2*atan2(|a.(b x c)|, 1 + a.b + b.c + c.a), which stays accurate
        for the small triangles a cubed sphere produces. """
        numer = np.abs(np.einsum('ij,ij->i', a, np.cross(b, c)))
        denom = (1.0 + np.einsum('ij,ij->i', a, b)
                 + np.einsum('ij,ij->i', b, c)
                 + np.einsum('ij,ij->i', c, a))
        return 2.0 * np.arctan2(numer, denom)

    def edges_to_faces(self, field):
        """ Maps an edge-located field onto face centres by averaging the
        edges of each face.

        Winds live in W2h on cell edges while mass lives in W3 at cell centres,
        so any product of the two must be co-located first. The mesh's own
        face-edge connectivity does this without regridding.

        field: array with the edge index as its last dimension. """
        idx = self.face_edges
        valid = idx >= 0
        safe = np.where(valid, idx, 0)
        gathered = field[..., safe]
        gathered = np.where(valid, gathered, np.nan)
        return np.nanmean(gathered, axis=-1)

    def zonal_mean(self, field, lats, nbins=None):
        """ Bins a mesh field into latitude bands and averages within each.

        This is the zonal mean on the native mesh. Face and edge positions are
        scattered in latitude, so a binned average replaces the axis average
        that a regular grid would allow, and avoids interpolating first.

        field: array with the mesh index as its last dimension
        lats:  latitude in degrees for that mesh index
        Returns (bin centres in degrees, binned field). """
        if nbins is None:
            nbins = int(np.sqrt(lats.size / 6.0) * 2)
        edges = np.linspace(-90.0, 90.0, nbins + 1)
        which = np.clip(np.digitize(lats, edges) - 1, 0, nbins - 1)
        counts = np.bincount(which, minlength=nbins)
        flat = field.reshape(-1, lats.size)
        binned = np.stack([np.bincount(which, weights=row, minlength=nbins)
                           for row in flat])
        with np.errstate(invalid='ignore'):
            binned = binned / counts
        binned = binned.reshape(field.shape[:-1] + (nbins,))
        return 0.5 * (edges[:-1] + edges[1:]), binned

    def calc_density(self, time_slice):
        """ Derives air density at cell centres on half levels.

        The Venus diagnostic output carries exner and theta but not rho, so
        density comes from the model's own equation of state:
        p = p_zero*exner**(1/kappa), T = theta*exner, rho = p/(rd*T).
        theta lives on full levels and is averaged onto half levels to meet
        exner. """
        exner = self.data[self.key('exner')][time_slice].values
        theta = self.data[self.key('theta')][time_slice].values
        if theta.shape[0] == exner.shape[0] + 1:
            theta_half = 0.5 * (theta[:-1, :] + theta[1:, :])
        else:
            theta_half = theta
        temperature = theta_half * exner
        pressure = self.p_zero * exner**(1.0 / self.kappa)
        return pressure / (self.rd * temperature)

    def calc_relative_am(self, trange=(0, None)):
        """ Calculates the relative angular momentum of the atmosphere.

        M_r = integral of u * r * cos(lat) dm, with r = radius + z for the
        deep atmosphere the configuration uses (shallow=.false.) and
        dm = rho * area * (r/radius)**2 * dz. The planetary term is excluded,
        so the result is the wind contribution alone and starts near zero for
        a run from rest.

        Returns (time in seconds, M_r in kg m2 s-1). """
        if not hasattr(self, 'areas'):
            self.set_resolution()

        ukey = self.key('u')
        nt = self.data[ukey].shape[0]
        stop = nt if trange[1] is None else trange[1]
        indices = np.arange(trange[0], stop)

        radius = self.radius + self.z_half
        coslat = np.cos(np.deg2rad(self.face_lats))
        # Moment arm r*cos(lat) and the cell volume element, both fixed in time.
        # Cell areas are on the surface sphere; deep-atmosphere cells widen
        # with height as (r/radius)**2.
        arm = radius[:, None] * coslat[None, :]
        volume = (self.areas[None, :] * (radius[:, None] / self.radius)**2
                  * self.dz)

        am = np.zeros(indices.size)
        for n, i in enumerate(indices):
            u_field = self.data[ukey][i]
            u_face = (u_field.values if 'nMesh2d_face' in u_field.dims
                      else self.edges_to_faces(u_field.values))
            rho = self.calc_density(int(i))
            am[n] = np.nansum(u_face * arm * rho * volume)

        self.am_time = self.time[indices]
        self.am_relative = am
        return self.am_time, self.am_relative

    def calc_superrotation_ke(self, trange=(0, None)):
        """ Calculates the globally integrated superrotation index and the
        global total kinetic energy, as in Lee et al. (2007), Fig. 3.

        S = M_r / M_omega (Read, 1986), where
        M_r     = integral of u * r * cos(lat) dm, the wind angular momentum
        M_omega = integral of omega * r**2 * cos(lat)**2 dm, the angular
                  momentum of the same atmosphere at rest relative to the
                  surface
        with r = radius + z and dm = rho * area * (r/radius)**2 * dz. S = 0
        for an atmosphere at rest; the total angular momentum is (1 + S)
        times M_omega. For a run from rest, 1 + S matches the total angular
        momentum normalised to its initial value, as in Lebonnois et al.
        (2010), Fig. 1, only while the mass distribution, and so M_omega,
        stays as it was at the start. Retrograde omega and
        retrograde wind are both negative, so superrotation gives S > 0.

        KE = integral of 0.5 * (u**2 + v**2) dm; vertical wind is omitted.

        Returns (time in seconds, S, KE in J). """
        if not hasattr(self, 'areas'):
            self.set_resolution()

        ukey, vkey = self.key('u'), self.key('v')
        nt = self.data[ukey].shape[0]
        stop = nt if trange[1] is None else trange[1]
        indices = np.arange(trange[0], stop)

        radius = self.radius + self.z_half
        coslat = np.cos(np.deg2rad(self.face_lats))
        arm = radius[:, None] * coslat[None, :]
        # Cell areas are on the surface sphere; deep-atmosphere cells widen
        # with height as (r/radius)**2.
        volume = (self.areas[None, :] * (radius[:, None] / self.radius)**2
                  * self.dz)

        index = np.zeros(indices.size)
        ke = np.zeros(indices.size)
        for n, i in enumerate(indices):
            u_field = self.data[ukey][i]
            v_field = self.data[vkey][i]
            u_face = (u_field.values if 'nMesh2d_face' in u_field.dims
                      else self.edges_to_faces(u_field.values))
            v_face = (v_field.values if 'nMesh2d_face' in v_field.dims
                      else self.edges_to_faces(v_field.values))
            mass = self.calc_density(int(i)) * volume
            m_rel = np.nansum(u_face * arm * mass)
            m_omega = np.nansum(self.rotrate * arm**2 * mass)
            index[n] = m_rel / m_omega
            ke[n] = np.nansum(0.5 * (u_face**2 + v_face**2) * mass)

        return self.time[indices], index, ke


# %%
def zmzw_and_am(plobject, meaning=True, trange=(0, None), time_slice=-1,
                nbins=None, plot=True, savepath=savedir,
                save=False, saveformat='png', savename='zmzw_and_am.png'):

    """ Input: LFRicPlanet object holding LFRic UGRID output
        Output: two-panel figure, zonal mean zonal wind on the left and
        relative angular momentum against time on the right

        meaning (default True) time-means the wind over trange
        time_slice (default -1) selects the time if meaning=False
        nbins (default None) sets the number of latitude bands, chosen from
        the mesh size if not given

        Eastward wind is plotted with its physical sign. Venus superrotates
        retrograde, so an established superrotation appears as negative u. """

    if not hasattr(plobject, 'areas'):
        plobject.set_resolution()

    zonal = plobject.data[plobject.key('u')]
    if meaning is True:
        wind = np.mean(zonal[trange[0]:trange[1], :, :].values, axis=0)
    else:
        wind = zonal[time_slice, :, :].values

    lats = plobject.face_lats if 'nMesh2d_face' in zonal.dims else plobject.edge_lats
    lats, zmean = plobject.zonal_mean(wind, lats, nbins=nbins)
    times, am = plobject.calc_relative_am(trange=trange)
    days = times / 86400.0

    if plot is not True:
        return lats, zmean, days, am

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    levels = ax[0].contourf(lats, plobject.heights, zmean,
                            cmap='RdBu_r', norm=TwoSlopeNorm(0))
    ax[0].set_title('Zonal mean zonal wind')
    ax[0].set_xlabel('Latitude [deg]')
    ax[0].set_ylabel('Height [km]')
    cbar = fig.colorbar(levels, ax=ax[0])
    cbar.ax.set_title('m/s')

    ax[1].plot(days, am, color='k')
    ax[1].axhline(0.0, color='gray', linewidth=0.8, linestyle='--')
    ax[1].set_title('Relative angular momentum')
    ax[1].set_xlabel('Time [Earth days]')
    ax[1].set_ylabel('$M_r$ [kg m$^2$ s$^{-1}$]')

    fig.suptitle(f'{plobject.name}, {plobject.run}')
    fig.tight_layout()

    if save is True:
        plt.savefig(os.path.join(savepath, savename), format=saveformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def atmosphere_state(plobject, meaning=True, trange=(0, None), time_slice=-1,
                     nbins=None, constant_cp=False, plot=True,
                     save=False, saveformat='png', savepath=savedir,
                     savename='atmosphere_state.png'):

    """ Input: LFRicPlanet object holding LFRic UGRID output
        Output: four-panel figure of the state of the atmosphere: zonal mean
        zonal wind, mean meridional mass streamfunction, and global
        area-weighted mean profiles of temperature and static stability

        meaning (default True) time-means every panel over trange
        time_slice (default -1) selects the time if meaning=False
        nbins (default None) sets the number of latitude bands, chosen from
        the mesh size if not given
        constant_cp (default False) evaluates the static stability with the
        c_p(T) law; True uses the namelist cp of a constant heat capacity run
        savepath (default SAVEDIR from .env) is the directory the figure is
        written to, created if missing

        The streamfunction is psi = 2*pi*r*cos(lat) * integral from z to the
        model top of [rho*v] dz, with r = radius + z. Positive values
        circulate clockwise in the latitude-height plane, as in the northern
        Hadley cell. """

    if not hasattr(plobject, 'areas'):
        plobject.set_resolution()

    if meaning is True:
        stop = plobject.time.size if trange[1] is None else trange[1]
        indices = np.arange(trange[0], stop)
    else:
        indices = np.atleast_1d(np.arange(plobject.time.size)[time_slice])

    ukey, vkey = plobject.key('u'), plobject.key('v')
    weights = plobject.areas / np.sum(plobject.areas)
    u_sum, rhov_sum, temp_sum, stab_sum = 0.0, 0.0, 0.0, 0.0
    for i in indices:
        u_field = plobject.data[ukey][i]
        v_field = plobject.data[vkey][i]
        u_face = (u_field.values if 'nMesh2d_face' in u_field.dims
                  else plobject.edges_to_faces(u_field.values))
        v_face = (v_field.values if 'nMesh2d_face' in v_field.dims
                  else plobject.edges_to_faces(v_field.values))
        u_sum = u_sum + u_face
        # rho*v is meaned before the zonal mean so that eddy mass fluxes
        # are kept in the streamfunction.
        rhov_sum = rhov_sum + plobject.calc_density(int(i)) * v_face
        temp_sum = temp_sum + np.sum(calc_temperature(plobject, int(i))
                                     * weights[None, :], axis=1)
        stab_sum = stab_sum + np.sum(
            calc_static_stability(plobject, int(i), constant_cp)
            * weights[None, :], axis=1)
    nt = indices.size

    # Zonal means are area-weighted, since cubed-sphere cells in one
    # latitude band differ in size; this keeps the column mass budget
    # closer to zero at the surface.
    _, band_area = plobject.zonal_mean(plobject.areas, plobject.face_lats,
                                       nbins)
    lats, zmzw = plobject.zonal_mean(u_sum / nt * plobject.areas,
                                     plobject.face_lats, nbins)
    zmzw = zmzw / band_area
    lats, zm_rhov = plobject.zonal_mean(rhov_sum / nt * plobject.areas,
                                        plobject.face_lats, nbins)
    zm_rhov = zm_rhov / band_area
    temp_profile = temp_sum / nt
    stab_profile = stab_sum / nt

    # Integrate rho*v downward from the model top onto full levels.
    column = np.cumsum(zm_rhov[::-1] * plobject.dz, axis=0)[::-1]
    column = np.concatenate([column, np.zeros((1, lats.size))], axis=0)
    radius = plobject.radius + plobject.z_full
    psi = 2 * np.pi * radius[:, None] * np.cos(np.deg2rad(lats))[None, :] \
        * column

    if plot is not True:
        return lats, zmzw, psi, temp_profile, stab_profile

    days = plobject.time[indices] / 86400.0
    if meaning is True:
        when = f'mean over Earth days {days[0]:.0f} to {days[-1]:.0f}'
    else:
        when = f'Earth day {days[0]:.0f}'

    fig, ax = plt.subplots(2, 2, figsize=(13, 10), layout='constrained')

    # Symmetric whole-number levels, so equal colour intensity means equal
    # speed in either direction. The step keeps about 20 levels.
    step = max(1, np.ceil(np.nanmax(np.abs(zmzw)) / 10))
    vmax = step * np.ceil(np.nanmax(np.abs(zmzw)) / step)
    cf = ax[0, 0].contourf(lats, plobject.heights, zmzw,
                           levels=np.arange(-vmax, vmax + step, step),
                           cmap='RdBu_r', norm=TwoSlopeNorm(0))
    ax[0, 0].set_title('Zonal mean zonal wind')
    cbar = fig.colorbar(cf, ax=ax[0, 0])
    cbar.ax.set_title('m/s')

    psi_scaled = psi * 1e-10
    step = max(1, np.ceil(np.nanmax(np.abs(psi_scaled)) / 10))
    vmax = step * np.ceil(np.nanmax(np.abs(psi_scaled)) / step)
    cf = ax[0, 1].contourf(lats, plobject.heights_full, psi_scaled,
                           levels=np.arange(-vmax, vmax + step, step),
                           cmap='RdBu_r', norm=TwoSlopeNorm(0))
    ax[0, 1].set_title('Mean meridional mass streamfunction')
    cbar = fig.colorbar(cf, ax=ax[0, 1])
    cbar.ax.set_title('$10^{10}$ kg/s')

    for a in ax[0]:
        a.set_xlabel('Latitude [deg]')
        a.set_ylabel('Height [km]')

    ax[1, 0].plot(temp_profile, plobject.heights, color='k')
    ax[1, 0].set_title('Global mean temperature')
    ax[1, 0].set_xlabel('Temperature [K]')

    if constant_cp is True:
        adiabat = f'constant $c_p$ = {plobject.cp:.0f} J/kg/K'
    else:
        adiabat = '$c_p(T)$'
    ax[1, 1].plot(stab_profile, plobject.heights_full[1:-1], color='k')
    ax[1, 1].axvline(0.0, color='gray', linewidth=0.8, linestyle='--')
    ax[1, 1].set_title(f'Global mean static stability, {adiabat}')
    ax[1, 1].set_xlabel('$dT/dz + g/c_p$ [K/km]')

    for a in ax[1]:
        a.set_ylabel('Height [km]')
        a.grid(alpha=0.5)

    fig.suptitle(f'{plobject.name}, {plobject.run}, {when}')

    if save is True:
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, savename), format=saveformat,
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def find_segments(rundir, nsegments=None, steps_per_day=16812):

    """ Lists the segments of a chained run to read, in time order.

        rundir holds segment subdirectories named
        <candidate>_<start day>_<end day>, each with one lfric_diag.nc.
        nsegments (default None) is the number of segments to read, counted
        from the first, between 1 and the number of segment directories.
        If None, every segment whose end-of-segment checkpoint exists and is
        not empty is read, stopping at the first segment without one. The
        checkpoint is written only when a segment finishes, so this skips a
        segment whose lfric_diag.nc is still being written. Give nsegments
        explicitly if checkpoints have been deleted.
        steps_per_day (default 16812, a Venus solar day at dt = 600 s) gives
        the step number of each segment's end checkpoint

        A segment that does not start where the previous one ended raises
        an error.

        Returns a list of (start day, end day, path). """

    pattern = re.compile(r'^(.+)_(\d+)_(\d+)$')
    segments = []
    for name in os.listdir(rundir):
        match = pattern.match(name)
        path = os.path.join(rundir, name)
        if match and os.path.isfile(os.path.join(path, 'lfric_diag.nc')):
            segments.append((int(match.group(2)), int(match.group(3)), path))
    segments.sort()
    if not segments:
        raise FileNotFoundError(f'No segment directories in {rundir}')

    if nsegments is None:
        complete = []
        for start, end, path in segments:
            ckpt = [f for f in glob.glob(os.path.join(
                        path, f'*_{end * steps_per_day:010d}.nc'))
                    if os.path.getsize(f) > 0]
            if not ckpt:
                break
            complete.append((start, end, path))
        if len(complete) < len(segments):
            print(f'Reading {len(complete)} complete segments of '
                  f'{len(segments)}; {segments[len(complete)][2]} has no '
                  f'non-empty end checkpoint')
        segments = complete
    else:
        if not 1 <= nsegments <= len(segments):
            raise ValueError(f'nsegments must be between 1 and '
                             f'{len(segments)}, the number of segments '
                             f'in {rundir}')
        segments = segments[:nsegments]
    if not segments:
        raise FileNotFoundError(f'No complete segments in {rundir}')
    for previous, current in zip(segments[:-1], segments[1:]):
        if current[0] != previous[1]:
            raise ValueError(f'Gap in the chain: {previous[2]} ends at day '
                             f'{previous[1]} but the next segment with '
                             f'lfric_diag.nc, {current[2]}, starts at day '
                             f'{current[0]}')

    return segments


# %%
def spinup_index(rundir, nsegments=None, steps_per_day=16812,
                 solar_day=1.00872e7, plot=True, save=False,
                 saveformat='png', savepath=savedir, savename=None):

    """ Input: top-level directory of a chained run, e.g. forced_haus, whose
        segment subdirectories are named <candidate>_<start day>_<end day>
        and each hold one lfric_diag.nc
        Output: two-panel figure after Lee et al. (2007), Fig. 3: globally
        integrated superrotation index S above global total kinetic energy,
        against time in Venus solar days

        nsegments (default None) and steps_per_day (default 16812) select
        the segments to read, as in find_segments, which also stops at a gap
        in the chain
        solar_day (default 1.00872e7 s) converts model time to Venus days

        Each segment is read on its own, so no file is held open beside the
        others and the mesh variables are never concatenated along time.
        The model time axis runs from the start of the chain, so segment
        times are used as they are; a time that fails to increase across
        segments raises an error.

        Returns (time in Venus days, S, KE in J). """

    segments = find_segments(rundir, nsegments, steps_per_day)

    run = os.path.basename(os.path.normpath(rundir))
    times, index, ke = [], [], []
    plobject = None
    for start, end, path in segments:
        print(f'Reading {path}')
        if plobject is None:
            plobject = LFRicPlanet(lfricdict, 'lfric', run)
            plobject.load_file(os.path.join(path, 'lfric_diag.nc'))
            plobject.set_resolution()
        else:
            plobject.data = xr.open_dataset(os.path.join(path, 'lfric_diag.nc'),
                                            decode_cf=False)
            plobject.time = plobject.data['time'].values
        t, s, k = plobject.calc_superrotation_ke()
        plobject.close()
        if times and t[0] <= times[-1][-1]:
            raise ValueError(f'Time in {path} does not follow on from the '
                             f'previous segment')
        times.append(t)
        index.append(s)
        ke.append(k)

    days = np.concatenate(times) / solar_day
    index = np.concatenate(index)
    ke = np.concatenate(ke)

    if plot is not True:
        return days, index, ke

    # Each quantity has its own panel and y-axis on a shared time axis, so
    # neither curve's apparent slope depends on the other's scale.
    fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                           layout='constrained')
    ax[0].plot(days, index, color='k', linewidth=2)
    ax[0].set_title('Globally integrated superrotation index')
    ax[0].set_ylabel('S')
    ax[1].plot(days, ke, color='k', linewidth=2)
    ax[1].set_title('Global total kinetic energy')
    ax[1].set_ylabel('KE [J]')
    ax[1].set_xlabel('Time [Venus solar days]')
    for a in ax:
        a.grid(alpha=0.5)
        a.set_xlim(0, days[-1])
    fig.suptitle(f'{plobject.name}, {run}')

    if save is True:
        if savename is None:
            savename = f'spinup_{run}.{saveformat}'
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, savename), format=saveformat,
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return days, index, ke

# %%
def zonal_wind_history(rundir, heights=(5, 15, 25, 35, 45, 55, 65, 75),
                       latband=30.0, nsegments=None, steps_per_day=16812,
                       solar_day=1.00872e7, plot=True, save=False,
                       saveformat='png', savepath=savedir, savename=None):

    """ Input: top-level directory of a chained run, e.g. forced_haus, as
        for spinup_index
        Output: one panel per height of the zonal mean zonal wind against
        time in Venus solar days

        heights (default 5 to 75 km in 10 km steps) are in km; each is
        taken at the nearest model half level
        latband (default 30) is the half-width in degrees of the band about
        the equator over which the wind is averaged, weighted by cell area.
        Because the zonal mean is a function of latitude, the band average
        reduces it to one value per height and time.
        nsegments and steps_per_day select the segments, as in
        find_segments; solar_day converts model time to Venus days

        Each panel has its own y-axis, so a wind of order 1 m/s in the deep
        atmosphere shows its trend beside a jet of order 50 m/s. Eastward
        wind has its physical sign; Venus superrotates retrograde, so
        superrotation is negative.

        Returns (time in Venus days, heights of the levels used in km,
        band-mean wind in m/s with shape (time, height)). """

    segments = find_segments(rundir, nsegments, steps_per_day)
    run = os.path.basename(os.path.normpath(rundir))
    times, winds = [], []
    plobject = None
    for start, end, path in segments:
        print(f'Reading {path}')
        if plobject is None:
            plobject = LFRicPlanet(lfricdict, 'lfric', run)
            plobject.load_file(os.path.join(path, 'lfric_diag.nc'))
            plobject.set_resolution()
            levels = np.array([np.argmin(np.abs(plobject.heights - h))
                               for h in heights])
            inband = np.abs(plobject.face_lats) <= latband
            weights = np.where(inband, plobject.areas, 0.0)
            weights = weights / weights.sum()
        else:
            plobject.data = xr.open_dataset(os.path.join(path, 'lfric_diag.nc'),
                                            decode_cf=False)
            plobject.time = plobject.data['time'].values
        u_field = plobject.data[plobject.key('u')]
        # Only the selected levels are read from the file.
        u_sel = u_field[:, levels].values
        if 'nMesh2d_face' not in u_field.dims:
            u_sel = plobject.edges_to_faces(u_sel)
        t = plobject.time
        plobject.close()
        if times and t[0] <= times[-1][-1]:
            raise ValueError(f'Time in {path} does not follow on from the '
                             f'previous segment')
        times.append(t)
        winds.append(np.sum(u_sel * weights, axis=-1))

    days = np.concatenate(times) / solar_day
    wind = np.concatenate(winds)
    used = plobject.heights[levels]

    if plot is not True:
        return days, used, wind

    ncol = 2
    nrow = int(np.ceil(used.size / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(11, 2.4 * nrow),
                           sharex=True, layout='constrained', squeeze=False)
    # Highest level in the top-left panel, so the grid reads top-down like
    # the atmosphere.
    order = np.argsort(used)[::-1]
    for a, j in zip(ax.flat, order):
        a.plot(days, wind[:, j], color='k', linewidth=1.5)
        a.set_title(f'{used[j]:.2f} km')
        a.set_ylabel('u [m/s]')
        a.grid(alpha=0.5)
        a.set_xlim(0, days[-1])
    for a in ax.flat[used.size:]:
        a.set_visible(False)
    for a in ax[-1]:
        a.set_xlabel('Time [Venus solar days]')
    fig.suptitle(f'{plobject.name}, {run}: zonal mean zonal wind, '
                 f'area mean over {latband:.0f}S to {latband:.0f}N')

    if save is True:
        if savename is None:
            savename = f'zonal_wind_history_{run}.{saveformat}'
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, savename), format=saveformat,
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return days, used, wind

# %%
