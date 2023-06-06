import pyopenms as oms
import numpy as np
import source.Spytrometer as Spytrometer
from datetime import datetime
import torch
import torch.optim as optim
import source.models as models
import source.parameters as params

exp = oms.MSExperiment()
oms.MzXMLFile().load("data\e01306.mzXML", exp)

spy = Spytrometer.Spytrometer(max_peak=2000)
bin_width = 0.02
bin_offset = 0
spy.tolarence_window = 20
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
    spy.normalize_regions(spectrum_id)

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

# Save new data as .mzxml file
spy.export_spectra_mzxml('output.mzXML', exp)