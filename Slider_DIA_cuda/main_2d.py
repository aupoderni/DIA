import pandas as pd
import torch
import torch.optim as optim
import numpy as np
import source.Spytrometer as Spytrometer
import source.models as models
import source.parameters as params
import source.visualization as visual
import time 
import random
torch.set_num_threads(4)
torch_type = params.types['float_type']

log_softmax = torch.nn.LogSoftmax(dim=0)

def Info(text, end='\n', flush=True, show=True):
    if show:
        print(text, end=end, flush=flush)

torch_device = torch.device("cuda:{}".format(2) if torch.cuda.is_available() else "cpu")

training_parameters = params.training

#Training parameters
window_width = training_parameters['window_width']
kernel_num = training_parameters['kernel_num'] #20
num_epochs = training_parameters['epochs'] #40
clip_value = training_parameters['clip_value'] #1
print_info = training_parameters['print_info'] #True
printing_tick = training_parameters['printing_tick'] #1
bin_width = training_parameters['bin_width']
batch_size = training_parameters['batch_size']

spy = Spytrometer.Spytrometer(max_peak=2000, bin_width=bin_width, bin_offset=0)

#Loading mzml and fasta files
spy.load_dia_data('/home/apoderni/data/e01306.mzML')
spy.load_fasta('/home/apoderni/data/cerevisiae_orf_trans_all.fasta')

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
#spy.sort_spectra()
spy.set_candidate_peptides()      
print("Number of proteins:\t%d"%(len(spy.protein_collection)))
print("Number of spectra:\t%d"%(len(spy.spectrum_collection)))
print("Number of peptides:\t%d"%(len(spy.peptide_collection)))

# Load train_data obtained by percolator
# Load train_data obtained by percolator
data_percolator = pd.read_csv('/home/apoderni/data/train_data_percolator.csv')

peptides = data_percolator[['sequence', 'charge']].drop_duplicates(subset=['sequence'])
peptides_seq = peptides['sequence'].tolist()
target_peptide = []
target_binary = []
# Get list of unique peptides and their binaries
for i in range(len(spy.peptide_collection)):
    if spy.peptide_collection[i].peptide_seq in peptides_seq:
        spy.calculate_peptide_fragmentation(i)
        idx_charge = peptides.index[peptides['sequence'] == spy.peptide_collection[i].peptide_seq][0]
        spy.get_teo_binary(i, charge=peptides['charge'][idx_charge])
        target_peptide.append(spy.peptide_collection[i].peptide_seq)
        target_binary.append(spy.peptide_collection[i].spectrum_array)
print('Length of training data: ', len(target_peptide))
# Delete sequences from training data, obtained by percolator, if they are not in spy.peptide_collection
for sequence in data_percolator['sequence']:
    if sequence not in target_peptide:
        data_percolator.drop(data_percolator.loc[data_percolator['sequence']==sequence].index, inplace=True)

spectrum_size = spy.max_bin
print(spectrum_size)

conv_net = models.DeepConv(window_width=window_width, kernel_num=kernel_num, bin_width=bin_width, 
                           spectrum_size=spectrum_size, torch_device=torch_device, 
                           torch_type=torch_type)
conv_net = conv_net.to(torch_device)

train_accuracies = []

optimizer = optim.Adam(filter(lambda p: p.requires_grad, conv_net.parameters()),
        lr=training_parameters['learning_rate'],
        weight_decay=0)

xent = models.BCELossWeight(pos_weight=torch.tensor(20, dtype=torch.float, device=torch_device))

start_time = time.time()
learning_curve = []

for epoch in range(num_epochs):
    # Shuffle peptide indices to train a model on all theoretical spectra in random order
    random_index = random.sample(range(len(target_peptide)), len(target_peptide))
    partial_loss = 0
    batch_cnt = 0
    # For each theoretical spectrum
    for idx in random_index:
        # y - one theoretical spectrum
        y = np.array(target_binary[idx])
        # x - all experimental spectra that corresponds to y
        train_data = data_percolator.loc[data_percolator['sequence'] == target_peptide[idx]]
        train_data = train_data.sample(frac=1)
        x = []
        for scan in train_data['scan']:
            spectrum_idx = spy.get_spectrum_by_scan(scan)
            x.append(spy.spectrum_collection[spectrum_idx].spectrum_array)
        X = torch.tensor(np.array(x), dtype=torch.float).to(torch_device) # scans x 1999
        Y = torch.tensor(y, dtype=torch.float).to(torch_device) # 1 x 1999
        X = X.unsqueeze(0).to(torch_device) # 1 x scans x 1999
        Y = Y.unsqueeze(0).to(torch_device) # 1 x 1 x 1999
        X = X.unsqueeze(0).to(torch_device) # 1 x 1 x scans x 1999
        Y = Y.unsqueeze(0).to(torch_device) # 1 x 1 x 1 x 1999
        #print(X.size())
        if batch_cnt % 1000 == 0:
            print('batch_cnt = ', batch_cnt)
        output = conv_net(X)
        loss = xent(output, Y)
        loss.backward()
        if batch_cnt % batch_size == 0:
            conv_net.clip_grads(clip_value=clip_value)
            optimizer.step()
            optimizer.zero_grad()
        #
        partial_loss += loss.detach().data.cpu().numpy()
        batch_cnt += 1
    print(X.size(), ' - training data size')
    print(output.size(), ' - size of output')
    epoch_error = (partial_loss/batch_cnt)
    learning_curve.append(epoch_error)
    if (epoch+1)%printing_tick == 0 or epoch == 0:
        print("Epoch: {}/{}. Time: {}, loss:{}".format(epoch+1, num_epochs, round(time.time()-start_time, 2), epoch_error))
torch.save(conv_net, 'Slider_DIA.pth')
Info("Learning done. Time: {} sec.".format(round(time.time()-start_time, 2)), show=print_info)
filename_curve = str(num_epochs) + '_ep_' + str(window_width) + '_ww_' + str(kernel_num) + '_kn_' + str(training_parameters['learning_rate']) + '_lr'
visual.plot_learning_curve(learning_curve, filename_curve)