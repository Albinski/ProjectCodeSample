import os
import xarray as xr
import numpy as np
import sys
from datetime import datetime, timedelta
from parcels import (Field, FieldSet, ParticleSet, JITParticle,
                     AdvectionRK4, ErrorCode, Variable)
from glob import glob
from scipy.ndimage import distance_transform_edt

year, month = int(sys.argv[1]), int(sys.argv[2])

# Start Time
t0 = datetime(year=year, month=month, day=14, hour=12)

# Duration, calculation timestep, output interval
run_time = timedelta(weeks = 31)
run_dt = timedelta(minutes = 15)
output_dt = timedelta(days = 1)

# Sinking and Beaching timescales
ts = 1/timedelta(days=365).total_seconds() # Decay timescale
tb = 1/timedelta(days = 30).total_seconds()

### DIRECTORIES ###
dirs = {}
dirs['root'] = os.getcwd() + '/../../'
dirs['script'] = dirs['root'] + 'ocean_personal_data/albinski/'
dirs['traj'] = dirs['root'] + 'ocean_personal_data/albinski/Trajs/'
dirs['ocean'] = dirs['root'] + 'snapdragon/swio/WINDS-M/'
dirs['grid'] = dirs['root'] + 'ocean_personal_data/albinski/'

#Automation of file name creation
start_year_month = f"{t0.year}_{t0.month:02d}_{t0.day}"
duration_weeks = int(run_time.days / 7)
traj_filename = f"a_validation_{start_year_month}_{duration_weeks}w.zarr"

# File handles
fh = {}
fh['ocean'] = sorted(glob(dirs['ocean'] + 'WINDS_SFC_*.nc'))
fh['grid'] = dirs['grid'] + 'croco_grd.nc'
fh['traj'] = dirs['traj'] + traj_filename
fh['clustering'] = dirs['grid'] + 'updated_clustering_labels_psi.npy'

ds = xr.open_dataset(fh['grid'])

mask_psi = ds['mask_psi'].values
lon_psi = ds['lon_psi'].values
lat_psi = ds['lat_psi'].values

# Calculate the distance transform using mask_psi
distance_to_coast_psi = distance_transform_edt(mask_psi)  # Calculating distance from the grid cells to the land cells

clustering_labels = np.load(fh['clustering'])

# Assigning the clustering labels to land points
labels_grid = np.full(mask_psi.shape, -1)
labels_grid[np.where(mask_psi == 0)] = clustering_labels

# Calculate distances and indices to nearest land
distances, indices = distance_transform_edt(mask_psi != 0, return_indices=True)
nearest_land_indices = tuple(indices[i, mask_psi != 0] for i in range(len(indices)))

ocean_labels = labels_grid[nearest_land_indices]
final_labels_grid_psi = np.full(mask_psi.shape, -1)
final_labels_grid_psi[mask_psi != 0] = ocean_labels

# Masking the placeholder values (-1) before plotting
final_labels_grid_masked_psi = np.ma.masked_where(final_labels_grid_psi == -1, final_labels_grid_psi)

### DEFINING THE FIELDS ###
island_id_field = Field('Island_ID', data=final_labels_grid_masked_psi,
                       lon=lon_psi, lat=lat_psi, mesh='spherical', allow_time_extrapolation=True, interp_method='nearest')
distance_field = Field('distance_to_coast', data=distance_to_coast_psi, # Initialise the distance_to_coast variable into a field
                       lon=lon_psi, lat=lat_psi, mesh='spherical', allow_time_extrapolation=True, interp_method= 'nearest')

fset = FieldSet.from_nemo(filenames={'U': fh['ocean'], 'V': fh['ocean']},
                          variables={'U': 'u_surf', 'V': 'v_surf'},
                          dimensions={'lon': 'nav_lon_u', 'lat': 'nav_lat_v', 'time': 'time_counter'},
                          mesh='spherical', allow_time_extrapolation=False) # Do not want static current extrapolation at end time
fset.add_constant('ts', ts) # Add the mass decay timescale to the FieldSet so the particle can read it
fset.add_constant('tb', tb)
fset.add_field(distance_field) # Add the distance field generated above so that the particle can read it
fset.add_field(island_id_field)

### DEFINING THE PARTICLE CLASS ###
class CoastDebrisDistanceParticle(JITParticle): #Islands need to be specified manually, as there is limited functionality within OceanParcels kernels
    existance = Variable('existance', dtype = np.int8, initial = 1, to_write = False)
    distance = Variable('distance', dtype=np.float32, initial = 0, to_write = False)
    mass = Variable('mass', dtype=np.float32, initial=1.0)
    nearest_id = Variable('nearest_id', dtype=np.int32, initial=-3, to_write = False)
    lon_initial = Variable('lon_initial', dtype=np.float32, initial=0.0)
    lat_initial = Variable('lat_initial', dtype=np.float32, initial=0.0)
    island2 = Variable('island2', dtype=np.float32, initial=0, to_write = False)
    island3 = Variable('island3', dtype=np.float32, initial=0, to_write = False)
    island4 = Variable('island4', dtype=np.float32, initial=0, to_write = False)
    island5 = Variable('island5', dtype=np.float32, initial=0, to_write = False)
    island6 = Variable('island6', dtype=np.float32, initial=0, to_write = False)
    island7 = Variable('island7', dtype=np.float32, initial=0, to_write = False)
    island8 = Variable('island8', dtype=np.float32, initial=0, to_write = False)
    island9 = Variable('island9', dtype=np.float32, initial=0, to_write = False)
    island10 = Variable('island10', dtype=np.float32, initial=0, to_write = False)
    island11 = Variable('island11', dtype=np.float32, initial=0, to_write = False)
    island12 = Variable('island12', dtype=np.float32, initial=0, to_write = False)
    island13 = Variable('island13', dtype=np.float32, initial=0, to_write = False)
    island14 = Variable('island14', dtype=np.float32, initial=0, to_write = False)
    island15 = Variable('island15', dtype=np.float32, initial=0, to_write = False)
    island16 = Variable('island16', dtype=np.float32, initial=0, to_write = False)
    island17 = Variable('island17', dtype=np.float32, initial=0, to_write = False)
    island18 = Variable('island18', dtype=np.float32, initial=0, to_write = False)
    island19 = Variable('island19', dtype=np.float32, initial=0, to_write = False)
    island20 = Variable('island20', dtype=np.float32, initial=0, to_write = False)
    island21 = Variable('island21', dtype=np.float32, initial=0, to_write = False)
    island22 = Variable('island22', dtype=np.float32, initial=0, to_write = False)
    island23 = Variable('island23', dtype=np.float32, initial=0, to_write = False)
    island24 = Variable('island24', dtype=np.float32, initial=0, to_write = False)
    island25 = Variable('island25', dtype=np.float32, initial=0, to_write = False)
    island26 = Variable('island26', dtype=np.float32, initial=0, to_write = False)
    island27 = Variable('island27', dtype=np.float32, initial=0, to_write = False)
    island28 = Variable('island28', dtype=np.float32, initial=0, to_write = False)
    island29 = Variable('island29', dtype=np.float32, initial=0, to_write = False)
    island30 = Variable('island30', dtype=np.float32, initial=0, to_write = False)
    island31 = Variable('island31', dtype=np.float32, initial=0, to_write = False)
    island33 = Variable('island33', dtype=np.float32, initial=0, to_write = False)
    island34 = Variable('island34', dtype=np.float32, initial=0, to_write = False)
    island35 = Variable('island35', dtype=np.float32, initial=0, to_write = False)
    island36 = Variable('island36', dtype=np.float32, initial=0, to_write = False)
    island37 = Variable('island37', dtype=np.float32, initial=0, to_write = False)
    island38 = Variable('island38', dtype=np.float32, initial=0, to_write = False)
    island39 = Variable('island39', dtype=np.float32, initial=0, to_write = False)
    island40 = Variable('island40', dtype=np.float32, initial=0, to_write = False)
    island41 = Variable('island41', dtype=np.float32, initial=0, to_write = False)
    island42 = Variable('island42', dtype=np.float32, initial=0, to_write = False)
    island43 = Variable('island43', dtype=np.float32, initial=0, to_write = False)
    island44 = Variable('island44', dtype=np.float32, initial=0, to_write = False)
    island45 = Variable('island45', dtype=np.float32, initial=0, to_write = False)
    island46 = Variable('island46', dtype=np.float32, initial=0, to_write = False)
    island47 = Variable('island47', dtype=np.float32, initial=0, to_write = False)
    island48 = Variable('island48', dtype=np.float32, initial=0, to_write = False)
    island49 = Variable('island49', dtype=np.float32, initial=0, to_write = False)
    island50 = Variable('island50', dtype=np.float32, initial=0, to_write = False)
    island51 = Variable('island51', dtype=np.float32, initial=0, to_write = False)
    island52 = Variable('island52', dtype=np.float32, initial=0, to_write = False)
    island53 = Variable('island53', dtype=np.float32, initial=0, to_write = False)
    island55 = Variable('island55', dtype=np.float32, initial=0, to_write = False)
    island56 = Variable('island56', dtype=np.float32, initial=0, to_write = False)
    island57 = Variable('island57', dtype=np.float32, initial=0, to_write = False)
    island58 = Variable('island58', dtype=np.float32, initial=0, to_write = False)
    island59 = Variable('island59', dtype=np.float32, initial=0, to_write = False)
    island60 = Variable('island60', dtype=np.float32, initial=0, to_write = False)
    island100 = Variable('island100', dtype=np.float32, initial=0, to_write = False)
    island105 = Variable('island105', dtype=np.float32, initial=0, to_write = False)
    island140 = Variable('island140', dtype=np.float32, initial=0, to_write = False)
    island150 = Variable('island150', dtype=np.float32, initial=0, to_write = False)
    island200 = Variable('island200', dtype=np.float32, initial=0, to_write = False)
    island220 = Variable('island220', dtype=np.float32, initial=0, to_write = False)
    island269 = Variable('island269', dtype=np.float32, initial=0, to_write = False)
    island300 = Variable('island300', dtype=np.float32, initial=0, to_write = False)
    island330 = Variable('island330', dtype=np.float32, initial=0, to_write = False)
    island350 = Variable('island350', dtype=np.float32, initial=0, to_write = False)


particle_lon = np.load('/home/ocean_personal_data/albinski/pre_work_for_RUNS/pre_scripts/validation_particle_lon.npy')
particle_lat = np.load('/home/ocean_personal_data/albinski/pre_work_for_RUNS/pre_scripts/validation_particle_lat.npy')
particle_lon = particle_lon.flatten()
particle_lat = particle_lat.flatten()

### CREATING THE PARTICLE SET WITH CLASS AND DENSITY FROM ABOVE ###
pset = ParticleSet.from_list(fieldset=fset,
                             pclass=CoastDebrisDistanceParticle,
                             lonlatdepth_dtype=np.float64,
                             lon=particle_lon, lat=particle_lat, time=t0)
pset.set_variable_write_status('depth', False) # Don't output depth, since this is a 2D simulation

# Set up the ParticleFile
pfile = pset.ParticleFile(name=fh['traj'], outputdt=output_dt)

### SET UP THE KERNELS ###
def SetInitialPosition(particle, fieldset, time):
    if particle.existance == 1:
        if particle.lon_initial == 0.0 and particle.lat_initial == 0.0:
            particle.lon_initial = particle.lon
            particle.lat_initial = particle.lat

def IslandMassTransfer(particle, fieldset, time):
    if particle.existance == 1:
        particle.distance = fieldset.distance_to_coast[time, particle.depth, particle.lat, particle.lon]
        particle.nearest_id = fieldset.Island_ID[time, particle.depth, particle.lat, particle.lon]   
        if particle.distance <= 3:
            transferred_mass = fieldset.tb  * particle.mass * particle.dt
            mass_loss_factor = 1 - (fieldset.ts + fieldset.tb) * particle.dt
            particle.mass *= mass_loss_factor
            if particle.nearest_id == 2.0:
                particle.island2 += transferred_mass
            elif particle.nearest_id == 3.0:
                particle.island3 += transferred_mass
            elif particle.nearest_id == 4.0:
                particle.island4 += transferred_mass
            elif particle.nearest_id == 5.0:
                particle.island5 += transferred_mass
            elif particle.nearest_id == 6.0:
                particle.island6 += transferred_mass
            elif particle.nearest_id == 7.0:
                particle.island7 += transferred_mass
            elif particle.nearest_id == 8.0:
                particle.island8 += transferred_mass
            elif particle.nearest_id == 9.0:
                particle.island9 += transferred_mass
            elif particle.nearest_id == 10.0:
                particle.island10 += transferred_mass
            elif particle.nearest_id == 11.0:
                particle.island11 += transferred_mass
            elif particle.nearest_id == 12.0:
                particle.island12 += transferred_mass
            elif particle.nearest_id == 13.0:
                particle.island13 += transferred_mass
            elif particle.nearest_id == 14.0:
                particle.island14 += transferred_mass
            elif particle.nearest_id == 15.0:
                particle.island15 += transferred_mass
            elif particle.nearest_id == 16.0:
                particle.island16 += transferred_mass
            elif particle.nearest_id == 17.0:
                particle.island17 += transferred_mass
            elif particle.nearest_id == 18.0:
                particle.island18 += transferred_mass
            elif particle.nearest_id == 19.0:
                particle.island19 += transferred_mass
            elif particle.nearest_id == 20.0:
                particle.island20 += transferred_mass
            elif particle.nearest_id == 21.0:
                particle.island21 += transferred_mass
            elif particle.nearest_id == 22.0:
                particle.island22 += transferred_mass
            elif particle.nearest_id == 23.0:
                particle.island23 += transferred_mass
            elif particle.nearest_id == 24.0:
                particle.island24 += transferred_mass
            elif particle.nearest_id == 25.0:
                particle.island25 += transferred_mass
            elif particle.nearest_id == 26.0:
                particle.island26 += transferred_mass
            elif particle.nearest_id == 27.0:
                particle.island27 += transferred_mass
            elif particle.nearest_id == 28.0:
                particle.island28 += transferred_mass
            elif particle.nearest_id == 29.0:
                particle.island29 += transferred_mass
            elif particle.nearest_id == 30.0:
                particle.island30 += transferred_mass
            elif particle.nearest_id == 31.0:
                particle.island31 += transferred_mass
            elif particle.nearest_id == 33.0:
                particle.island33 += transferred_mass
            elif particle.nearest_id == 34.0:
                particle.island34 += transferred_mass
            elif particle.nearest_id == 35.0:
                particle.island35 += transferred_mass
            elif particle.nearest_id == 36.0:
                particle.island36 += transferred_mass
            elif particle.nearest_id == 37.0:
                particle.island37 += transferred_mass
            elif particle.nearest_id == 38.0:
                particle.island38 += transferred_mass
            elif particle.nearest_id == 39.0:
                particle.island39 += transferred_mass
            elif particle.nearest_id == 40.0:
                particle.island40 += transferred_mass
            elif particle.nearest_id == 41.0:
                particle.island41 += transferred_mass
            elif particle.nearest_id == 42.0:
                particle.island42 += transferred_mass
            elif particle.nearest_id == 43.0:
                particle.island43 += transferred_mass
            elif particle.nearest_id == 44.0:
                particle.island44 += transferred_mass
            elif particle.nearest_id == 45.0:
                particle.island45 += transferred_mass
            elif particle.nearest_id == 46.0:
                particle.island46 += transferred_mass
            elif particle.nearest_id == 47.0:
                particle.island47 += transferred_mass
            elif particle.nearest_id == 48.0:
                particle.island48 += transferred_mass
            elif particle.nearest_id == 49.0:
                particle.island49 += transferred_mass
            elif particle.nearest_id == 50.0:
                particle.island50 += transferred_mass
            elif particle.nearest_id == 51.0:
                particle.island51 += transferred_mass
            elif particle.nearest_id == 52.0:
                particle.island52 += transferred_mass
            elif particle.nearest_id == 53.0:
                particle.island53 += transferred_mass
            elif particle.nearest_id == 55.0:
                particle.island55 += transferred_mass
            elif particle.nearest_id == 56.0:
                particle.island56 += transferred_mass
            elif particle.nearest_id == 57.0:
                particle.island57 += transferred_mass
            elif particle.nearest_id == 58.0:
                particle.island58 += transferred_mass
            elif particle.nearest_id == 59.0:
                particle.island59 += transferred_mass
            elif particle.nearest_id == 60.0:
                particle.island60 += transferred_mass
            elif particle.nearest_id == 100.0:
                particle.island100 += transferred_mass
            elif particle.nearest_id == 105.0:
                particle.island105 += transferred_mass
            elif particle.nearest_id == 140.0:
                particle.island140 += transferred_mass
            elif particle.nearest_id == 150.0:
                particle.island150 += transferred_mass
            elif particle.nearest_id == 200.0:
                particle.island200 += transferred_mass
            elif particle.nearest_id == 220.0:
                particle.island220 += transferred_mass
            elif particle.nearest_id == 269.0:
                particle.island269 += transferred_mass
            elif particle.nearest_id == 300.0:
                particle.island300 += transferred_mass
            elif particle.nearest_id == 330.0:
                particle.island330 += transferred_mass
            elif particle.nearest_id == 350.0:
                particle.island350 += transferred_mass

        else:
            particle.mass = particle.mass * (1 - fieldset.ts * particle.dt)

kernel = pset.Kernel(SetInitialPosition) + pset.Kernel(IslandMassTransfer) + pset.Kernel(AdvectionRK4) 

# Kernel to delete particles (e.g. particle leaves the domain)
def deactivate_move_land(particle, fieldset, time):
    particle.existance = 0
    particle.lon = 35
    particle.lat = -5

### Run the simulation ###
pset.execute(kernel,
             runtime=run_time,
             dt=run_dt,
             recovery={ErrorCode.ErrorOutOfBounds: deactivate_move_land,
                       ErrorCode.ErrorInterpolation: deactivate_move_land},
                       output_file=pfile)
