import pyopenms as oms
import numpy as np
import source.Spytrometer as Spytrometer
import source.utils as utils
import torch
import torch.optim as optim
import source.models as models
import source.parameters as params

torch_device = torch.device("cuda:{}".format(2) if torch.cuda.is_available() else "cpu")

#set parameters
mass_proton = 1.00727646688
bin_width = 1.0005079
bin_offset = 0.0

#create oms experiment and load the data
exp = oms.MSExperiment()
oms.MzXMLFile().load("/home/apoderni/data/e01306.mzXML", exp)
'''
for spectrum in exp:
    if spectrum.getMSLevel() > 1:
        filtered_mz = []
        filtered_int = []
        bins = discretize_spectrum(spectrum, mass_proton, bin_width, bin_offset)
        filtered_mz, filtered_int = undiscretize_spectrum(bins, mass_proton, bin_width, bin_offset)
        spectrum.set_peaks((filtered_mz, filtered_int))
'''

# create a dictionary with precusrsor masses as keys and a list of 
# all spectra corresponding to each precursor mass as values
spectrum_cnt = 0
prec_masses = {}
for spectrum in exp:
    if spectrum.getMSLevel() > 1:
        if spectrum.getPrecursors()[0].getMZ() not in prec_masses.keys():
            prec_masses[spectrum.getPrecursors()[0].getMZ()] = [spectrum_cnt]
        else:
            prec_masses[spectrum.getPrecursors()[0].getMZ()].append(spectrum_cnt)
    spectrum_cnt += 1

# print a value of the precursor mass and the length of the list of all corresponding spectra
for key, value in prec_masses.items():
    print(key, len(value))

# load model
model = torch.load("/home/apoderni/Slider_DIA.pth").to(torch_device)

# run model for each precursor mass separately and rewrite spectrum array
for key, value in prec_masses.items():
    # discretize all spectra with selected precursor mass and store them in spectrum_collection_concat
    spectrum_collection_concat = []
    for index in value:
        bins = utils.discretize_spectrum(exp[index], mass_proton, bin_width, bin_offset)
        spectrum_collection_concat.append(bins)
    print(key)
    # convert the list to a tensor, split it and fit each part of a tensor to Slider separately, 
    # then concatenate back into np.array
    X = torch.tensor(np.array(spectrum_collection_concat), dtype=torch.float)
    spectrum_collection_concat = []
    for x in X.split(10):
        x = x.unsqueeze(0).to(torch_device) # 1 x scans x 1999
        x = x.unsqueeze(0).to(torch_device) # 1 x 1 x scans x 1999
        if len(spectrum_collection_concat) == 0:
            spectrum_collection_concat = torch.squeeze(model(x)).detach().cpu().numpy()
        else:
            trained_spectrum = torch.squeeze(model(x)).detach().cpu().numpy()
            spectrum_collection_concat = np.concatenate((spectrum_collection_concat, trained_spectrum), axis=0)
    cnt = 0
    for spectrum in exp:
        if spectrum.getMSLevel() > 1 and spectrum.getPrecursors()[0].getMZ() == key:
            bins = spectrum_collection_concat[cnt].tolist()
            filtered_mz = []
            filtered_int = []
            filtered_mz, filtered_int = utils.undiscretize_spectrum(bins, mass_proton, bin_width, bin_offset)
            spectrum.set_peaks((np.array(filtered_mz), np.array(filtered_int)))
            exp[value[cnt]] = spectrum
            cnt += 1
    print(exp[value[1]].get_peaks()[0])

oms.MzXMLFile().store('output.mzXML', exp)
'''
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
'''
--------------------------------------------------------------—
Layer (type) Output Shape Param #
================================================================
BatchNorm2d-1 [-1, 1, 5, 20000] 2
Conv2d-2 [-1, 20, 5, 20000] 6,080
BatchNorm2d-3 [-1, 20, 5, 20000] 40
Sigmoid-4 [-1, 20, 5, 20000] 0
Conv2d-5 [-1, 1, 5, 20000] 21
BatchNorm2d-6 [-1, 1, 5, 20000] 2
Sigmoid-7 [-1, 1, 5, 20000] 0
================================================================
Total params: 6,145
Trainable params: 6,145
Non-trainable params: 0
--------------------------------------------------------------—
Input size (MB): 0.38
Forward/backward pass size (MB): 48.83
Params size (MB): 0.02
Estimated Total Size (MB): 49.23
----------------------------------------------------------------
'''