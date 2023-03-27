import numpy as np
import source.Spytrometer as Spytrometer
from datetime import datetime
import torch
import torch.optim as optim
import source.models as models
import source.parameters as params

spy = Spytrometer.Spytrometer(max_peak=2000)

# Load dia data
start_time = datetime.now()
spy.load_dia_data('c:\\Users\\poder\\lab\\Slider_DIA\\data\\e01306.mzML')
spy.load_fasta('c:\\Users\\poder\\crux-demo\\cerevisiae_orf_trans_all.fasta')

for spectrum_id, spectrum in enumerate(spy.spectrum_collection):
    spy.discretize_spectrum(spectrum_id)
    spy.normalize_regions(spectrum_id)
   
print("Protein digestion done. Time (h:m:s):\t"+str(datetime.now() - start_time))
print("Number of spectra:\t%d"%(len(spy.spectrum_collection)))

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

# Save new data as .ms2 file
spy.export_spectra_ms2('e01306.ms2')