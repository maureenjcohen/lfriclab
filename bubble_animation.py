""" Data organisation and animation for LFRic-Venus vertical slice runs """

# %%
## Import packages
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm
from config import savedir

# %%
## Constants of the slice configuration, mirroring its &planet and
## &extrusion namelists. Gravity in m/s2, specific heat and gas constant in
## SI units, reference pressure in Pa, domain height in m.
slicedict = {'g': 8.87, 'cp': 900.0, 'rd': 191.4, 'p_zero': 9.2e6,
             'domain_height': 1.0e4, 'name': 'Venus slice'}

## Names of the mesh topology variables in LFRic UGRID output.
slicemeshdict = {'node_x': 'Mesh2d_node_x', 'node_y': 'Mesh2d_node_y',
                 'face_nodes': 'Mesh2d_face_nodes'}

## Candidate names for each physical field in the slice output.
slicefielddict = {'theta': ['theta', 'theta_inst'],
                  'exner': ['exner', 'exner_inst'],
                  'rho': ['rho', 'rho_inst'],
                  'w': ['w_in_wth', 'w'],
                  'u': ['u_in_w2h', 'u']}


### LFRicSlice class object definition ###
### Manages configuration data and LFRic output on a biperiodic planar mesh.

# %%
class LFRicSlice:
    """ An object which contains the output data for an LFRic vertical slice.

    The slice runs on a biperiodic planar mesh (BiP), with x and y in metres
    rather than longitude and latitude. The mesh is a regular grid of nx by
    ny square cells, so fields are reordered onto that grid and averaged
    over the short y direction to give an x-z section. """

    def __init__(self, slicedict, model, run):
        """ Initiates an LFRicSlice object using the input dictionary of
        constants, the name of the model, and the name of the run.
        Run names: free text, e.g. hot_off, hot_on """

        self.name = slicedict['name']
        self.model = model
        self.run = run
        print(f'This is the {self.run} dataset created by {self.model.upper()}')
        for key, value in slicedict.items():
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
        """ Closes netCDF file packaged in LFRicSlice data object """
        self.data.close()
        print('LFRicSlice object associated dataset has been closed')

    def contents(self):
        """ Prints reference list for easy formatted oversight of file contents"""
        print(*self.reflist, sep='\n')

    def key(self, field):
        """ Resolves a physical field name onto the variable in the file. """
        for candidate in slicefielddict[field]:
            if candidate in self.data.variables:
                return candidate
        raise KeyError(f'No variable in file for {field}; '
                       f'tried {slicefielddict[field]}')

    def _scaled(self, name):
        """ Mesh coordinate in metres. With decode_cf=False the stored
        scale_factor is not applied on load. """
        var = self.data[name]
        return var.values * var.attrs.get('scale_factor', 1.0)

    def _periodic_centres(self, node_coord, face_nodes):
        """ Face centres along one periodic direction, from the corner nodes.

        The Mesh2d_face_x/y written by LFRic are wrong for the cells that
        straddle the periodic boundary, since their corners are averaged
        without unwrapping. Here each corner is unwrapped relative to the
        first corner before averaging.
        Returns (centres, cell width, domain length). """
        unique = np.unique(np.round(node_coord, 3))
        width = np.min(np.diff(unique))
        length = unique.size * width
        corners = node_coord[face_nodes]
        offset = corners - corners[:, :1]
        offset -= length * np.round(offset / length)
        centres = corners[:, 0] + offset.mean(axis=1)
        low = unique.max() - length
        return low + np.mod(centres - low, length), width, length

    def set_resolution(self):
        """ Rebuilds the regular grid of the planar mesh and the vertical
        coordinate of the uniform extrusion. """
        face_nodes = self.data[slicemeshdict['face_nodes']]
        start = int(face_nodes.attrs.get('start_index', 0))
        face_nodes = face_nodes.values - start

        x, self.dx, self.lx = self._periodic_centres(
            self._scaled(slicemeshdict['node_x']), face_nodes)
        y, self.dy, self.ly = self._periodic_centres(
            self._scaled(slicemeshdict['node_y']), face_nodes)
        self.nx = int(round(self.lx / self.dx))
        self.ny = int(round(self.ly / self.dy))
        # Face order that lays the faces out row by row, x varying fastest.
        self.grid_order = np.lexsort((x, y))
        self.x = x[self.grid_order][:self.nx]
        self.y = y[self.grid_order][::self.nx]
        self.xkm = self.x * 1e-3

        self.nlayers = self.data.sizes['half_levels']
        self.dz = self.domain_height / self.nlayers
        self.z_full = np.arange(self.nlayers + 1) * self.dz
        self.z_half = (np.arange(self.nlayers) + 0.5) * self.dz
        self.heights_full = self.z_full * 1e-3
        self.heights = self.z_half * 1e-3
        self.time = self.data['time'].values
        print(f'Grid is {self.nx} x {self.ny} cells of {self.dx:.0f} m, '
              f'{self.nlayers} layers of {self.dz:.0f} m')

    def to_section(self, field):
        """ Reorders a face field onto the regular grid and averages over y,
        giving an x-z section.

        field: array with the face index as its last dimension. """
        gridded = field[..., self.grid_order]
        gridded = gridded.reshape(field.shape[:-1] + (self.ny, self.nx))
        return gridded.mean(axis=-2)

    def calc_temperature(self, time_slice=-1):
        """ Temperature at cell centres on half levels, T = theta*exner, as
        an x-z section. Full-level theta is averaged onto half levels. """
        theta = self.data[self.key('theta')][time_slice].values
        exner = self.data[self.key('exner')][time_slice].values
        if theta.shape[0] == exner.shape[0] + 1:
            theta = 0.5 * (theta[:-1] + theta[1:])
        return self.to_section(theta * exner)

    def calc_temperature_anomaly(self, time_slice=-1):
        """ Temperature excess over the environment at the same height.

        The environment is the median over x at each level. The bubble fills
        a minority of the columns, so the median ignores it. At a given
        height the pressure is nearly uniform, so this excess is the buoyancy
        of the bubble whichever heat capacity the model uses. """
        temperature = self.calc_temperature(time_slice)
        return temperature - np.median(temperature, axis=1, keepdims=True)

    def calc_background_temperature(self, time_slice=0):
        """ Temperature profile of the environment on half levels, the
        median over x at each level. """
        return np.median(self.calc_temperature(time_slice), axis=1)


# %%
def bubble_animation(plobjects, anomaly=True, trange=(0, None), levels=None,
                     interval=150, save=False, saveformat='gif',
                     savepath=savedir, savename='bubble_animation.gif'):

    """ Input: list of two LFRicSlice objects, e.g. the hot bubble with the
        variable_cp switch off and on
        Output: animation of the bubble, one run per panel, labelled by run
        name

        anomaly (default True) plots the temperature excess over the
        environment at each height, False plots the temperature itself
        trange (default all) selects the output times to animate
        levels (default None) sets the contour levels, chosen from the data
        range over all frames of both runs if not given
        saveformat 'gif' uses the pillow writer, 'mp4' needs ffmpeg
        savepath (default SAVEDIR from .env) is the directory the file is
        written to, created if missing """

    for plobject in plobjects:
        if not hasattr(plobject, 'grid_order'):
            plobject.set_resolution()

    calc = 'calc_temperature_anomaly' if anomaly is True else 'calc_temperature'
    nt = plobjects[0].time.size
    frames = range(trange[0], nt if trange[1] is None else trange[1])
    # Every frame is held in memory so that all panels share one colour scale.
    fields = [np.stack([getattr(plobject, calc)(i) for i in frames])
              for plobject in plobjects]

    if levels is None:
        if anomaly is True:
            vmax = np.ceil(max(np.abs(f).max() for f in fields))
            levels = np.linspace(-vmax, vmax, 21)
        else:
            levels = np.linspace(min(f.min() for f in fields),
                                 max(f.max() for f in fields), 21)
    if anomaly is True:
        cmap, norm, title, unit = 'RdBu_r', TwoSlopeNorm(0), \
            'Temperature anomaly', 'K'
    else:
        cmap, norm, title, unit = 'inferno', None, 'Temperature', 'K'

    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True,
                           layout='constrained')

    for a, plobject in zip(ax, plobjects):
        a.set_title(f'{plobject.run}')
        a.set_xlabel('x [km]')
    ax[0].set_ylabel('Height [km]')

    # The right-hand panel's outer axis labels each height with the
    # environmental temperature of the first frame. The panels share heights,
    # so one temperature axis serves both. Temperature falls monotonically
    # with height, so the profile is inverted by reversing it.
    background = plobjects[-1].calc_background_temperature(frames[0])
    heights = plobjects[-1].heights
    secax = ax[-1].secondary_yaxis(
        'right', functions=(lambda h: np.interp(h, heights, background),
                            lambda t: np.interp(t, background[::-1],
                                                heights[::-1])))
    # Ticks stay inside the profile, beyond which np.interp clamps.
    secax.set_yticks(np.arange(np.ceil(background.min() / 10) * 10,
                               background.max(), 10))
    secax.set_ylabel('Background temperature [K]')

    contours = []

    def draw(n):
        for cs in contours:
            cs.remove()
        contours.clear()
        for a, plobject, field in zip(ax, plobjects, fields):
            contours.append(a.contourf(plobject.xkm, plobject.heights,
                                       field[n], levels=levels, cmap=cmap,
                                       norm=norm, extend='both'))
            a.grid(alpha=0.5)
        fig.suptitle(f'{title}, {plobjects[0].name}, '
                     f't = {plobjects[0].time[frames[n]]:.0f} s')
        return contours

    draw(0)
    cbar = fig.colorbar(contours[-1], ax=ax)
    cbar.ax.set_title(unit)

    ani = animation.FuncAnimation(fig, draw, frames=len(frames),
                                  interval=interval, repeat=False)

    if save is True:
        writer = 'pillow' if saveformat == 'gif' else 'ffmpeg'
        os.makedirs(savepath, exist_ok=True)
        ani.save(os.path.join(savepath, savename), writer=writer)
        plt.close()
    else:
        plt.show()
    return ani

# %%
# hot_off = LFRicSlice(slicedict, 'lfric', 'hot_off')
# hot_off.load_file('data/slice_test/lfric_diag_hot_off.nc')
# hot_on = LFRicSlice(slicedict, 'lfric', 'hot_on')
# hot_on.load_file('data/slice_test/lfric_diag_hot_on.nc')
# bubble_animation([hot_off, hot_on], save=True, savepath=savedir)
