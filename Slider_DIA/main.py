import pandas as pd
import torch
import torch.optim as optim
import numpy as np
import source.Spytrometer as Spytrometer
import source.models as models
import source.parameters as params
import source.visualization as visual
import time 
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(4)
torch_type = params.types['float_type']

log_softmax = torch.nn.LogSoftmax(dim=0)

def Info(text, end='\n', flush=True, show=True):
    if show:
        print(text, end=end, flush=flush)

torch_device = torch.device("cpu")

training_parameters = params.training

spy = Spytrometer.Spytrometer(max_peak=2000)
bin_width = 0.02
bin_offset = 0
spy.tolarence_window = 20
spy.tolerance_type = 'PPM'
spy.max_theo_pept_peak_charge = 3
spy.remove_precursor_peak = True
spy.remove_precursor_tolerance = 1.5
spy.missed_cleavages = 0
spy.min_pept_len = 7
spy.unique_peptides = 1

#Loading mzml and fasta files
spy.load_dia_data('c:\\Users\\poder\\crux-demo\\Slider_DIA\\e01306.mzML', precursor_mass = 410.4365)
spy.load_fasta('c:\\Users\\poder\\crux-demo\\cerevisiae_orf_trans_all.fasta')

#Spectrum discreditation and normalization
for spectrum_id, spectrum in enumerate(spy.spectrum_collection):
    spy.discretize_spectrum(spectrum_id)
    spy.normalize_regions(spectrum_id)
for i in range(len(spy.protein_collection)):
    try:
        spy.generate_peptides(i)
    except:
        continue
spy.sort_peptides()    
spy.sort_spectra()
spy.set_candidate_peptides()      
print("Number of proteins:\t%d"%(len(spy.protein_collection)))
print("Number of spectra:\t%d"%(len(spy.spectrum_collection)))
print("Number of peptides:\t%d"%(len(spy.peptide_collection)))

# Load train_data obtained by percolator
data_percolator = pd.read_csv('c:\\Users\\poder\\crux-demo\\Slider_DIA\\train_data_percolator.csv')

peptides = data_percolator['sequence'].unique()
target_peptide = []
target_binary = []

for i in range(len(spy.peptide_collection)):
    if spy.peptide_collection[i].peptide_seq in peptides:
        spy.calculate_peptide_fragmentation(i)
        spy.get_teo_binary(i)
        target_peptide.append(spy.peptide_collection[i].peptide_seq)
        target_binary.append(spy.peptide_collection[i].spectrum_array)

#Delete sequences from training data, obtained by percolator, if they are not in spy.peptide_collection
for sequence in data_percolator['sequence']:
    if sequence not in target_peptide:
        data_percolator.drop(data_percolator.loc[data_percolator['sequence']==sequence].index, inplace=True)

#y - target values (binary vectors of theoretical peaks)
y = []
for sequence in data_percolator['sequence']:
    for i in range(len(target_peptide)):
        if sequence == target_peptide[i]:
            y.append(target_binary[i])
#x - trainable data (binary vectors of experimental spectra)
x = []
for scan in data_percolator['scan']:
    spectrum_idx = spy.get_spectrum_by_scan(scan - 1)
    x.append(spy.spectrum_collection[spectrum_idx].spectrum_array)

X = torch.tensor(np.stack(x, axis=0), dtype=torch.float)
Y = torch.tensor(np.stack(y, axis=0), dtype=torch.float)
X = X.unsqueeze(1)
Y = Y.unsqueeze(1)


#Training parameters
window_width = training_parameters['window_width']
batch_size = training_parameters['batch_size'] #16
kernel_num = training_parameters['kernel_num'] #20
num_epochs = training_parameters['epochs'] #20
clip_value = training_parameters['clip_value'] #1
print_info = training_parameters['print_info'] #True
printing_tick = training_parameters['printing_tick'] #1
spectrum_size = spy.max_bin #1999
print(spectrum_size)

train_data = TensorDataset(X, Y)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

conv_net = models.DeepConv(window_width=window_width, kernel_num=kernel_num, spectrum_size=spectrum_size, torch_device=torch_device, torch_type=torch_type)
conv_net = conv_net.to(torch_device)

train_accuracies = []

optimizer = optim.Adam(filter(lambda p: p.requires_grad, conv_net.parameters()),
        lr=training_parameters['learning_rate'],
        weight_decay=0)

xent = models.BCELossWeight(pos_weight=torch.tensor(20, dtype=torch.float, device=torch_device))

start_time = time.time()
learning_curve = []

for epoch in range(num_epochs):
    partial_loss = 0
    batch_cnt = 0
    for x_train, y_train in train_loader:
        optimizer.zero_grad()
        output = conv_net(x_train)
        loss = xent(output, y_train)
        loss.backward()
        conv_net.clip_grads(clip_value=clip_value)
        optimizer.step()
        partial_loss += loss
        batch_cnt += 1
    epoch_error = (partial_loss/batch_cnt).data.cpu().numpy()
    learning_curve.append(epoch_error)
    if (epoch+1)%printing_tick == 0 or epoch == 0:
        print("Epoch: {}/{}. Time: {}, loss:{}".format(epoch+1, num_epochs, round(time.time()-start_time, 2), epoch_error))

Info("Learning done. Time: {} sec.".format(round(time.time()-start_time, 2)), show=print_info)
filename_curve = str(num_epochs) + '_ep_batch_' + str(batch_size)
visual.plot_learning_curve(learning_curve, filename_curve)
