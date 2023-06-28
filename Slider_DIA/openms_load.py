import pyopenms as oms
import numpy as np
import source.Spytrometer as Spytrometer
from datetime import datetime
import torch
import torch.optim as optim
import source.models as models
import source.parameters as params

def discretize_spectrum(spectrum, mass_proton, bin_width, bin_offset):
    peaks = spectrum.get_peaks()
    peak_bins = list(map(lambda p: mass2bin(p, mass_proton, bin_width, bin_offset), peaks[0]))
    bins = np.zeros(peak_bins[-1] + 1, dtype='float64')
    for peak_idx, peak in enumerate(peak_bins):
        if peak < (peak_bins[-1] + 1) and bins[peak] < peaks[1][peak_idx]:
            bins[peak] = peaks[1][peak_idx]
    return bins

def undiscretize_spectrum(bins, mass_proton, bin_width, bin_offset):
    mz = []
    intensity = []
    peak_bin_idx = np.nonzero(bins)
    bin_to_peak = [bin2mass(peak_bin, mass_proton, bin_width, bin_offset)
                    for peak_bin in peak_bin_idx[0]]
    for i, peak in enumerate(bin_to_peak):
        peak_intensity = bins[peak_bin_idx[0][i]]
        if peak > 0:
            mz.append(peak)
            intensity.append(peak_intensity)
    return mz, intensity

def create_oms_spectrum(mz, intensity, spectrum):
    oms_spectrum = oms.MSSpectrum()
    oms_spectrum.set_peaks([mz, intensity])
    oms_spectrum.setMSLevel(spectrum.getMSLevel())
    oms_spectrum.setRT(spectrum.getRT())
    oms_spectrum.setPrecursors(spectrum.getPrecursors())
    oms_spectrum.setInstrumentSettings(spectrum.getInstrumentSettings())
    return oms_spectrum

def mass2bin(mass, mass_proton, bin_width, bin_offset,  charge=1):
    return ((mass + (charge - 1) * mass_proton)/
            (charge*bin_width) + 1.0 - bin_offset).astype(int)

def bin2mass(bin, mass_proton, bin_width, bin_offset, charge=1):
        """Convert bin to mass"""
        return ((bin - 1.0 + bin_offset) * (charge * bin_width)
                 + (charge - 1) * mass_proton)

#set parameters
mass_proton = 1.00727646688
bin_width = 0.01
bin_offset = 0.0

#create oms experiment and load the data
exp = oms.MSExperiment()
oms.MzXMLFile().load("data\e01306.mzXML", exp)

#create another oms experiment for data export
exp_outp = oms.MSExperiment()

for cnt, _ in enumerate(exp):
    if exp[cnt].getMSLevel() == 1:
        exp_outp.addSpectrum(exp[cnt])
        continue
    
    bins = discretize_spectrum(exp[cnt], mass_proton, bin_width, bin_offset)

    #binned spectrum to mz&intensity
    mz, intensity = undiscretize_spectrum(bins, mass_proton, bin_width, bin_offset)

    #recreate oms spectrum
    oms_spectrum = create_oms_spectrum(mz, intensity, exp[cnt])
    exp_outp.addSpectrum(oms_spectrum)

    if cnt == 1:
        print(exp[0].get_peaks()[0])
        print(exp_outp[0].get_peaks()[0])
        print(exp[1].get_peaks()[0])
        print(exp_outp[1].get_peaks()[0])
oms.MzXMLFile().store('output.mzXML', exp_outp)

'''
spy = Spytrometer.Spytrometer(max_peak=2000, bin_width=0.01, bin_offset=0.0, remove_precursor_peak=False, skip_preprocessing=True)

#spy.bin_width = 0.2
#spy.bin_offset = 0.0
spy.tolarence_window = 10
spy.tolerance_type = 'PPM'
spy.max_theo_pept_peak_charge = 5
spy.remove_precursor_peak = False
spy.remove_precursor_tolerance = 1.5
spy.missed_cleavages = 0
spy.min_pept_len = 7
spy.unique_peptides = 1

# Load dia data
start_time = datetime.now()
spy.load_openms(exp)
for spectrum_id, spectrum in enumerate(spy.spectrum_collection):
    spy.discretize_spectrum(spectrum_id)
    #spy.normalize_regions(spectrum_id)

# Make a dict with precursor masses as keys and spectrum index lists with these precursor masses as values

prec_masses = {}
for i in range(len(spy.spectrum_collection)):
    if spy.spectrum_collection[i].precursor_mass not in prec_masses.keys():
        prec_masses[spy.spectrum_collection[i].precursor_mass] = [i]
    else:
        prec_masses[spy.spectrum_collection[i].precursor_mass].append(i)
for key, value in prec_masses.items():
    print(key, len(value))

model = torch.load("C:/Users/poder/lab/Slider_DIA/Slider_DIA.pth")

# Run model for each precursor mass separately and rewrite spectrum array
for key, value in prec_masses.items():
    spectrum_collection_concat = []
    for index in value:
        spectrum_collection_concat.append(spy.spectrum_collection[index].spectrum_array)
    X = torch.tensor(spectrum_collection_concat, dtype=torch.float)
    X = X.unsqueeze(0) # 1 x scans x 1999
    X = X.unsqueeze(0) # 1 x 1 x scans x 1999
    spectrum_collection_concat = torch.squeeze(model(X)).detach().numpy()
    for i in range(len(spectrum_collection_concat)):
        spy.spectrum_collection[value[i]].spectrum_array = spectrum_collection_concat[i].tolist()
'''
# Save new data as .mzxml file
#spy.export_spectra_mzxml('output.mzXML', exp)
