import numpy as np
import h5py
import os


def create_YY_grid(datadir, gridfile):
    # tau values [Gyr]:
    tau = np.arange(0.1, 15.1, 0.1)

    # Number of mass values per feh/alpha/tau combination:
    nm = 140

    # Get list of isochrone data files
    isofiles = [file for file in os.listdir(datadir) if file.endswith('.dat')]

    # Number of lines from the start of one data block to the next
    line_jump = nm + 2

    # Names and units of parameters in the columns.
    # The data is saved under these names in the hdf5 file
    # and the units are added as attributes (metadata).
    params = [('Mini', 'solar'),
              ('logT', 'log10(K)'),
              ('logL', 'log10(solar)'),
              ('logg', 'log10(cm/s2)'),
              ('U', 'mag'),
              ('B', 'mag'),
              ('V', 'mag'),
              ('R', 'mag'),
              ('I', 'mag'),
              ('J', 'mag'),
              ('H', 'mag'),
              ('Ks', 'mag'),
              ('G', 'mag')]

    print("Building YY isochrone grid from files in directory '"
          + datadir + "'...\n")

    with h5py.File(gridfile) as YYgrid:
        for datafile in isofiles:
            print("Processing '" + datafile + "'...")

            # Get alphaFe and FeH from the name of the datafile
            alpha_data = float(datafile[7:10])
            feh_data = float(datafile[1:6])

            for i, age in enumerate(tau):
                gridpath = 'alphaFe=' + format(alpha_data, '.4f') + '/' +\
                           'FeH=' + format(feh_data, '.4f') + '/' +\
                           'age=' + format(age, '.4f') + '/'

                # Read nm lines from the datafile which contain the data for
                # the relevant feh/alpha/tau combination
                data_i = np.genfromtxt(os.path.join(datadir, datafile),
                                       skip_header=3+i*line_jump, max_rows=nm)

                # Make a new array and fill in the data to be stored e.g. the
                # magnitudes as calculated from the colors given in the
                # datafile.
                data_mags = np.zeros((data_i.shape[0], len(params)))
                V = data_i[:, 4]
                VmI = data_i[:, 8]
                for i in range(4):
                    data_mags[:, i] = data_i[:, i]
                data_mags[:, 4] = V + data_i[:, 5] + data_i[:, 6] # U = V+(U-B)+(B-V)
                data_mags[:, 5] = V + data_i[:, 6] # B = V+(B-V)
                data_mags[:, 6] = V # V = V
                data_mags[:, 7] = V - data_i[:, 7] # R = V-(V-R)
                data_mags[:, 8] = V - VmI # I = V-(V-I)
                data_mags[:, 9] = V - data_i[:, 9] # J = V-(V-J)
                data_mags[:, 10] = V - data_i[:, 10] # H = V-(V-H)
                data_mags[:, 11] = V - data_i[:, 11] - 0.044 # Ks = V-(V-K)-0.044
                G = V - 0.0354 - 0.0561*VmI - 0.1767*VmI**2 - 0.0108*VmI**3
                data_mags[:, 12] = G

                # Store the data in the grid file
                for ind, (pname, punit) in enumerate(params):
                    YYgrid[gridpath + pname] = data_mags[:, ind]
                    YYgrid[gridpath + pname].attrs.create('unit', np.string_(punit))

    print("\nBuild done!")
