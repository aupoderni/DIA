import math
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO
import seaborn
from pyteomics import mzml, parser  # mass, mgf

spytrometer_proton = 1.00727646688


class Spectrum:  # peptide arrays
    def __init__(
        self,
        path_to_file,
        scan_id,
        mz_array,
        intensity_array,
        charge,
        precursor_mass,
        max_peak,
        remove_precursor_peak,
        remove_precursor_tolerance,
    ):

        # spectrum info
        name_start = path_to_file.rfind("/") + 1
        self.id = 0
        self.path_to_file = path_to_file[name_start:]
        self.scan_id = scan_id

        self.charge = charge
        self.precursor_mass = precursor_mass
        self.neutral_mass = (self.precursor_mass - spytrometer_proton) * self.charge

        mask = np.all([intensity_array > 1e-10], axis=0)
        mask3 = np.all([mz_array < max_peak], axis=0)
        mask2 = np.all([mz_array < self.neutral_mass + 50], axis=0)
        if remove_precursor_peak:
            mask1 = np.all(
                [abs(mz_array - precursor_mass) > remove_precursor_tolerance], axis=0
            )
        else:
            mask1 = True

        self.mz_array = mz_array[mask & mask1 & mask2 & mask3]  # List of mz peaks
        self.intensity_array = intensity_array[
            mask & mask1 & mask2 & mask3
        ]  # List of the intensities of the mz peaks

        self.spectrum_array = (
            []
        )  # Vector for the discretized spectrum, depending on bin_width and max_mz parameters
        self.peak_bins = []

        # peptide-spectrum-match info
        self.threshold = 0  # some metrics for sorting
        self.peak_list = []  # candidate peaks for self supervision
        self.quintile = -1000000  # value of 20%-percentile score
        self.raw_score = -1000000  # raw score for any scoring method
        self.score = -1000000  # some matching score, like hyperscore or XCORR, etc.
        self.confidence = 1000000  # some statistical confidence value, such as E-value, or (exact) p-value
        self.qvalue = 1000000  # statistical q-value value from TDC or exact-pvalue
        self.n_candidates = 0
        self.peptide = None
        self.isotope = 0
        self.start_pept = 0
        self.end_pept = 0
        #    self.fdr = 1.0
        self.bias = 0

    def print_spectrum(self, spectrum_id):
        if self.peptide == None:
            return

        modifications = ""
        result_list = [
            self.path_to_file,
            self.scan_id,
            spectrum_id,
            self.charge,
            self.precursor_mass,
            self.neutral_mass,
            float(self.peptide.neutral_mass),
            float(self.score),
            self.confidence,
            self.qvalue,
            self.n_candidates,
            int(self.peptide.target),
            self.peptide.protein_id,
            self.peptide.peptide_id,
            self.peptide.peptide_seq,
            len(self.peptide.peptide_seq),
            modifications,
            self.peptide.cleavage_info,
            self.peptide.missed_cleavages,
            self.peptide.original_peptide_seq,
            float(self.quintile),
        ]

        return result_list


# Adding data from fasta file
class ProteinObj:  # peptide information
    def __init__(self, protein_id, protein_header, protein_seq, protein_flag=0):

        self.id = protein_id
        self.header = protein_header
        self.seq = protein_seq
        self.flag = protein_flag  # can be used to indicate something, which can be used to filter protein sequences


#### protein and peptide sequence related functions
# Adding data from fasta file
class PeptideObj(object):  # peptide information
    def __init__(
        self,
        neutral_mass,
        peptide_seq,
        protein_id,
        target,
        original_peptide_seq,
        cleavage_info,
        missed_cleavages,
        start_end_pos,
    ):

        self.neutral_mass = neutral_mass
        self.peptide_seq = peptide_seq
        self.protein_id = [protein_id]
        self.peptide_id = 0
        self.aa_mass = []
        self.target = target  # indicates if the peptide is target: True/False
        self.original_peptide_seq = original_peptide_seq
        self.cleavage_info = cleavage_info  # indicates the type of the cleavage which generated the peptide: full/semi
        self.missed_cleavages = missed_cleavages
        self.peaks = []
        self.weight = -1
        self.bias = 0
        self.spectrum_array = []
        self.start_end_pos = start_end_pos


class Spytrometer:
    def __init__(
        self,
        bin_width=1.0005079,
        bin_offset=0.4,
        remove_precursor_peak=True,
        remove_precursor_tolerance=1.5,
        max_peak=2000,
        skip_preprocessing=False,
        enzyme="trypsin",
        missed_cleavages=0,
        min_pept_len=7,
        max_pept_len=30,
        min_pept_mass=200,
        max_pept_mass=7200,
        max_mods=1,
        min_mods=0,
        decoy_format=2,  # 0 for reverse, 1 for shuffle, anything else means no decoy generation.
        semi_cleavage=0,  # 0 for full tryptic, 1 for semi tryptic, (2 for non-tryptic not supported yet)
        decoys_only=False,  # Generate only decoy peptides, without generating target peptides.
        modifications={},
        static_mods={"C+57.02146"},
        theo_pept_peaks="by",  # possible peaks: 'abcxyz'
        max_theo_pept_peak_charge=2,  # Should be renamed to max_theo_frag_peak_charge
        unique_peptides=1,  # 0 no unique peptides, much more memory and redundancy,
        # 1 for unique peptides, less memory, less peptides, but slower peptide generation
        tolarence_type="PPM",
        tolarence_window=10,
        intensity_cutoff_coefficient=0.05,
    ):

        # constants
        self.__version__ = "v1.0"
        self.elts_mono = {
            "H": 1.007825035,
            "C": 12.0,
            "N": 14.003074,
            "O": 15.99491463,
            "P": 30.973762,
            "S": 31.9720707,
        }
        self.B = 0.0
        self.mono_h2o = 2 * self.elts_mono["H"] + self.elts_mono["O"]
        self.Y = self.mono_h2o
        self.h2o = self.Y
        self.nh3 = 3 * self.elts_mono["H"] + self.elts_mono["N"]

        # spectrum processing related parameters
        self.skip_preprocessing = skip_preprocessing
        self.remove_precursor_peak = remove_precursor_peak
        self.remove_precursor_tolerance = remove_precursor_tolerance
        self.intensity_cutoff_coefficient = intensity_cutoff_coefficient
        self.max_xcorr_offset = 75
        self.max_intensity = 50.0
        self.num_spectrum_regions = 10

        # spectrum data related parameters
        self.max_peak = max_peak  # Highest peak considered. Peaks having higher m/z than this will be discarded
        self.bin_offset = bin_offset
        self.bin_width = bin_width
        self.max_bin = self.mass2bin(self.max_peak)  # maximum bin
        self.tolarence_type = tolarence_type
        self.tolarence_window = tolarence_window  # precursor_window

        # peptide generation related parameters
        self.missed_cleavages = missed_cleavages
        self.min_pept_len = min_pept_len
        self.max_pept_len = max_pept_len
        self.min_pept_mass = min_pept_mass
        self.max_pept_mass = max_pept_mass
        self.enzyme = enzyme
        self.max_mods = max_mods
        self.min_mods = min_mods
        self.modifications = modifications
        self.theo_pept_peaks = theo_pept_peaks
        self.max_theo_pept_peak_charge = max_theo_pept_peak_charge
        self.decoy_format = decoy_format
        self.decoys_only = decoys_only
        self.semi_cleavage = semi_cleavage
        self.unique_peptides = unique_peptides

        self.peptide_ids = {}
        self.spectrum_collection = []
        self.peptide_collection = []
        self.protein_collection = []  # protein sequence related data
        self.pattern = re.compile(r"-?\[([^\]]+)\]-?")

        self.peptide_set = set()

        self.aa_mass = {
            "G": 57.02146,
            "A": 71.03711,
            "S": 87.03203,
            "P": 97.05276,
            "V": 99.06841,
            "T": 101.04768,
            "C": 103.00919,
            "L": 113.08406,
            "I": 113.08406,
            "N": 114.04293,
            "D": 115.02694,
            "Q": 128.05858,
            "K": 128.09496,
            "E": 129.04259,
            "M": 131.04049,
            "H": 137.05891,
            "F": 147.06841,
            "U": 150.95364,
            "R": 156.10111,
            "Y": 163.06333,
            "W": 186.07931,
            "O": 0,  # 255.15829,
            "Z": 0,
            "X": 0,
            "B": 0,
            "J": 0,
        }  # Make this float
        self.aa_idx = {
            "G": 0,
            "A": 1,
            "S": 2,
            "P": 3,
            "V": 4,
            "T": 5,
            "C": 6,
            "L": 7,
            "I": 8,
            "N": 9,
            "D": 10,
            "Q": 11,
            "K": 12,
            "E": 13,
            "M": 14,
            "H": 15,
            "F": 16,
            "U": 17,
            "R": 18,
            "Y": 19,
            "W": 20,
            "O": 21,
            "Z": 22,
            "X": 23,
            "B": 24,
            "J": 25,
        }

        # Make this float
        for stat_mod in static_mods:
            if stat_mod[1] == "t":  # for terminal mods: e.g. : Nt+229.99
                if stat_mod[0] == "C":
                    self.Y += float(stat_mod[2:])
                if stat_mod[0] == "N":
                    self.B += float(stat_mod[2:])
                continue
            self.aa_mass[stat_mod[0]] += float(stat_mod[1:])

        self.aa_mass_idx = {}
        for AA, mass in self.aa_mass.items():
            idx = self.aa_idx[AA]
            self.aa_mass_idx[AA] = (idx, mass)

        self.aa_num = len(self.aa_mass)
        self.aa2_idx = np.zeros((self.aa_num, self.aa_num), dtype=int)
        index = self.aa_num
        for AA1, (idx1, mass1) in self.aa_mass_idx.items():
            for AA2, (idx2, mass2) in self.aa_mass_idx.items():
                self.aa2_idx[idx1, idx2] = index
                index += 1
        index = 0
        self.aa3_idx = np.zeros((self.aa_num, self.aa_num, self.aa_num), dtype=int)
        for AA1, (idx1, mass1) in self.aa_mass_idx.items():
            for AA2, (idx2, mass2) in self.aa_mass_idx.items():
                for AA3, (idx3, mass3) in self.aa_mass_idx.items():
                    self.aa3_idx[idx1, idx2, idx3] = index
                    index += 1

        self.w1 = 0
        self.b1 = 0
        self.w2 = 0
        self.b2 = 0

        self.spectrum_margin = 0
        self.max_n_candidates = 0
        self.spytrometer_proton = 1.00727646688

    def mass2bin(self, mass, charge=1):  # convert to bin
        bin = int(
            (mass + (charge - 1) * spytrometer_proton) / (charge * self.bin_width)
            + 1.0
            - self.bin_offset
        )
        return bin

    def AA_mass2bin(self, mass, charge=1):  # convert to bin
        return int(
            (mass + (charge - 1) * spytrometer_proton) / (charge * self.bin_width)
        )
        # return int((mass + (charge - 1) * spytrometer_proton) / (charge) )

    def Dalton2bin(self, mass, charge=1):  # convert to bin
        return int(mass / (charge * self.bin_width))

    def load_dia_data(self, path_to_file, precursor_mass, min_peak_th=10):
        print("Loading spectrum data...")
        start_time = datetime.now()
        self.spectrum_collection = []
        spectra = list(mzml.read(path_to_file))
        for spectrum in spectra:
            try:
                prec_mass = spectrum["precursorList"]["precursor"][0][
                    "selectedIonList"]["selectedIon"][0]["selected ion m/z"]
                if (prec_mass == precursor_mass):
                    spectrum_record = Spectrum(
                                path_to_file,  # path to file
                                int(spectrum["index"]),  # scan_id
                                np.float_(spectrum["m/z array"][:]),  # mz array
                                np.float_(spectrum["intensity array"][:]),  # intensity array
                                int(
                                    1
                                ),  # charge
                                float(
                                    spectrum["precursorList"]["precursor"][0][
                                        "selectedIonList"
                                    ]["selectedIon"][0]["selected ion m/z"]
                                ),  # precursor mass
                                self.max_peak,
                                self.remove_precursor_peak,
                                self.remove_precursor_tolerance,
                            )
                    if len(spectrum_record.intensity_array) >= min_peak_th:
                        self.spectrum_collection.append(spectrum_record)
            except:
                continue
        self.set_spectrum_idx()
        print(
            "Done. Time (h:m:s):\t" + str(datetime.now() - start_time)
        )  # Print out the total number of the proteins
            
    def load_data(self, path_to_file, min_peak_th=10, data_type="humvar"):
        print("Loading spectrum data...")
        start_time = datetime.now()
        self.spectrum_collection = []
        with mzml.read(path_to_file, dtype=dict) as spectra:
            for spectrum_id, spectrum in enumerate(spectra):
                if data_type in ["humvar", "iprg", "malaria", "yeast"]:
                    spectrum_record = Spectrum(
                        path_to_file,  # path to file
                        int(spectrum["id"][5:]),  # scan_id
                        np.float_(spectrum["m/z array"][:]),  # mz array
                        np.float_(spectrum["intensity array"][:]),  # intensity array
                        int(spectrum["ms2 file charge state"][0:1]),  # charge
                        float(spectrum["ms2 file charge state"][2:]),  # precursor mass
                        self.max_peak,
                        self.remove_precursor_peak,
                        self.remove_precursor_tolerance,
                    )
                elif data_type in ["ocean", "ecoli", "human"]:
                    spectrum_record = Spectrum(
                        path_to_file,  # path to file
                        int(spectrum["index"]),  # scan_id
                        np.float_(spectrum["m/z array"][:]),  # mz array
                        np.float_(spectrum["intensity array"][:]),  # intensity array
                        int(spectrum["ms2 file charge state"][0:1]),  # charge
                        float(spectrum["ms2 file charge state"][2:]),  # precursor mass
                        self.max_peak,
                        self.remove_precursor_peak,
                        self.remove_precursor_tolerance,
                    )
                else:
                    spectrum_record = Spectrum(
                        path_to_file,  # path to file
                        spectrum_id,  # scan_id
                        np.float_(spectrum["m/z array"][:]),  # mz array
                        np.float_(spectrum["intensity array"][:]),  # intensity array
                        int(
                            spectrum["precursorList"]["precursor"][0][
                                "selectedIonList"
                            ]["selectedIon"][0]["charge state"]
                        ),  # charge
                        float(
                            spectrum["precursorList"]["precursor"][0][
                                "selectedIonList"
                            ]["selectedIon"][0]["selected ion m/z"]
                        ),  # precursor mass
                        self.max_peak,
                        self.remove_precursor_peak,
                        self.remove_precursor_tolerance,
                    )
                if len(spectrum_record.intensity_array) >= min_peak_th:
                    self.spectrum_collection.append(spectrum_record)
        self.set_spectrum_idx()
        print(
            "Done. Time (h:m:s):\t" + str(datetime.now() - start_time)
        )  # Print out the total number of the proteins


    def sort_spectra(
        self, key="neutral_mass", reverse=False
    ):  # reverse=True => decreaseing; reverse=False => increasing
        self.spectrum_collection = sorted(
            self.spectrum_collection, key=lambda d: getattr(d, key), reverse=reverse
        )  # sort spectra by neutral_mass in increasing manner

    def set_candidate_peptides(self):
        # Spectrum collection must be sorted
        # Peptide collection must be sorted
        start_pept_id = 0
        end_pept_id = 0
        pept_num = len(self.peptide_collection)
        for spect_id in range(len(self.spectrum_collection)):
            spectrum = self.spectrum_collection[spect_id]
            min_mass, max_mass = self.compute_window(
                spectrum.neutral_mass, spectrum.charge
            )

            # finding candidate peptides using rolling window approach. peptides are assumed to be sorted by neutral mass
            if start_pept_id >= pept_num:
                break
            while self.peptide_collection[start_pept_id].neutral_mass < min_mass:
                start_pept_id += 1
                if start_pept_id >= pept_num:
                    break

            if end_pept_id < start_pept_id:
                end_pept_id = start_pept_id

            if end_pept_id >= pept_num:
                break
            while self.peptide_collection[end_pept_id].neutral_mass < max_mass:
                end_pept_id += 1
                if end_pept_id >= pept_num:
                    break
            self.spectrum_collection[spect_id].n_candidates = (
                end_pept_id - start_pept_id
            )
            self.spectrum_collection[spect_id].start_pept = start_pept_id
            self.spectrum_collection[spect_id].end_pept = end_pept_id

            if self.max_n_candidates < self.spectrum_collection[spect_id].n_candidates:
                self.max_n_candidates = self.spectrum_collection[spect_id].n_candidates

                
    def discretize_spectrum(self, spectrum_id):

        spectrum = self.spectrum_collection[spectrum_id]
        spectrum.spectrum_array = np.zeros(self.max_bin)
        spectrum.peak_bins = list(map(self.mass2bin, spectrum.mz_array))
        if not self.skip_preprocessing:
            sqrt_intensity_array = list(map(math.sqrt, spectrum.intensity_array))
        else:
            sqrt_intensity_array = spectrum.intensity_array

        # keep the maximum of the intensities in every bin
        for i, peak_bin in enumerate(spectrum.peak_bins):
            if (
                peak_bin < self.max_bin
                and spectrum.spectrum_array[peak_bin] < sqrt_intensity_array[i]
            ):
                spectrum.spectrum_array[peak_bin] = sqrt_intensity_array[i]

        
    def normalize_regions(self, spectrum_id, N=np.inf, initial=True):
        # discretize_spectrum must be called beforehand
        # Fill peaks
        spectrum = self.spectrum_collection[spectrum_id]
        largest_bin = max(spectrum.peak_bins)
        highest_intensity = max(spectrum.spectrum_array)
        intensity_cutoff = (
            highest_intensity * self.intensity_cutoff_coefficient
        )  # lower intensity cutoff
        region_size = (
            int(largest_bin / self.num_spectrum_regions) + 1
        )  # size of any region
        if initial == True:
            spectrum.spectrum_array[
                np.where(spectrum.spectrum_array <= intensity_cutoff)
            ] = 0
        highest_intensity = [
            max(spectrum.spectrum_array[i * region_size : (i + 1) * region_size])
            for i in range(self.num_spectrum_regions)
        ]

        N = N // self.num_spectrum_regions

        def normalize(id_intens):
            region_start = id_intens[0] * region_size
            if initial == True:
                spectrum.spectrum_array[region_start : region_start + region_size] /= id_intens[1]
                
            if N < region_size:
                idx = spectrum.spectrum_array[
                    region_start : region_start + region_size
                ].argsort()[: (region_size - N)]
                spectrum.spectrum_array[idx + region_start] = 0

        list(
            map(
                normalize,
                filter(lambda id_intens: id_intens[1], enumerate(highest_intensity)),
            )
        )
        
    def get_spectrum_by_scan(self, scan):
        """Getting spectrum by scan_id"""
        for cnt, spectrum in enumerate(self.spectrum_collection):
            if spectrum.scan_id == scan:
                return cnt

    def reset_search_results(self):
        for spectrum in self.spectrum_collection:
            spectrum.score = (
                -1000000
            )  # some matching score, like hyperscore or XCORR, etc. the bigger the better
            spectrum.confidence = 1000000  # some statistical confidence value, such as E-value, or (exact) p-value, smaller the better
            spectrum.qvalue = (
                1000000  # statistical q-value value from TDC or exact-pvalue
            )
            spectrum.peptide = None
            spectrum.isotope = 0


    def tailor_scoring(self):  # tailor_scoring
        # Assume that the protein fasta  and the spectrum data files are loaded
        # Assume that all spectra are discretized, normalized and preprocessed with XCORR_substract_background
        start_time = datetime.now()
        min_candidates = 20

        for spect_id in range(len(self.spectrum_collection)):
            spectrum = self.spectrum_collection[spect_id]

            if spectrum.n_candidates == 0:
                continue

            start_id = spectrum.start_pept
            end_id = spectrum.end_pept

            if spectrum.n_candidates < min_candidates:
                end_id = start_id + min_candidates
            if end_id >= len(self.peptide_collection):
                end_id = len(self.peptide_collection)
            if (end_id - start_id) < min_candidates:
                start_id = end_id - min_candidates
            if start_id < 0:
                start_id = 0

            scores = np.zeros(end_id - start_id)

            for i, pept_id in enumerate(range(start_id, end_id)):
                scores[i] = self.calculate_xcorr_score(spect_id, pept_id) * 50

            top_hits = max(int(len(scores) * 0.05), 5)
            scores_norm = np.sort(scores)[-top_hits]
            norm_scores = scores - scores_norm

            for i, pept_id in enumerate(range(spectrum.start_pept, spectrum.end_pept)):
                if norm_scores[i] > spectrum.score:
                    spectrum.score = norm_scores[i]
                    self.peptide_collection[pept_id].peptide_id = pept_id
                    spectrum.peptide = self.peptide_collection[pept_id]

    def compute_qvalues_tdc(self):

        self.sort_spectra(key="score", reverse=True)

        target_cnt = 0
        decoy_cnt = 1
        for spectrum in self.spectrum_collection:
            if not spectrum.peptide:
                break
            if spectrum.peptide.target == True:
                target_cnt += 1
            else:
                decoy_cnt += 1
            fdr = decoy_cnt / (target_cnt + 1)
            if fdr > 1.0:
                fdr = 1.0
            spectrum.qvalue = fdr

        # convert fdrs to qvalues:
        for i in range(len(self.spectrum_collection) - 2, -1, -1):
            if (
                self.spectrum_collection[i + 1].qvalue
                < self.spectrum_collection[i].qvalue
            ):
                self.spectrum_collection[i].qvalue = self.spectrum_collection[
                    i + 1
                ].qvalue

    def set_spectrum_idx(self):
        id = 0
        for spectrum in self.spectrum_collection:
            spectrum.id = id
            id += 1

    def compute_window(self, mass, charge=1):
        if self.tolarence_type == "MASS":
            out_min = mass - self.tolarence_window
            out_max = mass + self.tolarence_window
        elif self.tolarence_type == "MZ":
            mz_minus_proton = mass - spytrometer_proton  # mass must be precursor_mass
            out_min = (mz_minus_proton - self.tolarence_window) * charge
            out_max = (mz_minus_proton + self.tolarence_window) * charge
        elif self.tolarence_type == "PPM":
            tiny_precursor = self.tolarence_window * 1e-6
            out_min = mass * (1.0 - tiny_precursor)
            out_max = mass * (1.0 + tiny_precursor)
        else:
            out_min = out_max = mass
            print("Uncorrect type of tolerance!")
        return out_min, out_max

    def XCORR_substract_background(self, spectrum_id):
        # operation is as follows: new_observed = observed - average_within_window,
        # but average is computed as if the array extended infinitely: denominator is same throughout array, even
        # near edges (where fewer elements have been summed)

        # discretize_spectrum must be called beforehand
        spectrum = self.spectrum_collection[spectrum_id]

        multiplier = 1.0 / (self.max_xcorr_offset * 2)
        end = len(spectrum.spectrum_array)
        partial_sums = np.zeros(end + 1)

        partial_sums[0:end] = np.add.accumulate(spectrum.spectrum_array, axis=0)
        partial_sums[end] = partial_sums[end - 1]

        l_border = self.max_xcorr_offset
        r_border = end - self.max_xcorr_offset

        partial_sums_left = np.zeros(end)
        partial_sums_left[l_border + 1 : end] = partial_sums[0 : r_border - 1]

        partial_sums_right = np.repeat(partial_sums[end], end)
        partial_sums_right[0:r_border] = partial_sums[l_border:end]

        spectrum.spectrum_array[:] -= multiplier * (
            partial_sums_right[:] - partial_sums_left[:] - spectrum.spectrum_array[:]
        )

    def print_accepted_psms(self):
        count_01_percent = 0
        count_1_percent = 0
        count_5_percent = 0
        count_10_percent = 0
        for spectrum in self.spectrum_collection:
            if spectrum.peptide is None:
                break
            if spectrum.peptide.target == False:
                continue
            if spectrum.qvalue < 0.001:
                count_01_percent += 1
            if spectrum.qvalue < 0.01:
                count_1_percent += 1
            if spectrum.qvalue < 0.05:
                count_5_percent += 1
            if spectrum.qvalue < 0.10:
                count_10_percent += 1
        counts = (count_01_percent, count_1_percent, count_5_percent, count_10_percent)
        print(
            "Number of accepted PSMs at\t0.1%% FDR = %d,\t1%% FDR = %d,\t5%% FDR = %d,\t10%% FDR = %d"
            % counts
        )
        return counts

    def load_fasta(self, path_to_fasta):
        cnt = 0
        for record in SeqIO.parse(path_to_fasta, "fasta"):
            self.protein_collection.append(
                ProteinObj(cnt, str(record.id), str(record.seq))
            )
            cnt += 1

    def generate_peptides(self, protein_id, decoy=False):
        if self.enzyme == "trypsin":
            tide_trypsin = r"([KR](?=[^P]))"
            full_peptides = set(
                parser.cleave(
                    self.protein_collection[protein_id].seq,
                    tide_trypsin,
                    self.missed_cleavages,
                    self.min_pept_len,
                )
            )
        elif self.enzyme == "trypsin/p":  # ignore proline rule for trypsin
            tide_trypsin = r"([KR])"
            full_peptides = set(
                parser.cleave(
                    self.protein_collection[protein_id].seq,
                    tide_trypsin,
                    self.missed_cleavages,
                    self.min_pept_len,
                )
            )
        elif self.enzyme == "no-digestion":
            full_peptides = set()
            full_peptides.add(self.protein_collection[protein_id].seq)
        else:
            full_peptides = set(
                parser.cleave(
                    self.protein_collection[protein_id].seq,
                    parser.expasy_rules[self.enzyme],
                    self.missed_cleavages,
                    self.min_pept_len,
                )
            )

        peptides = []

        for peptide in full_peptides:  # check peptides
            if peptide.find("X") != -1:
                continue
            if peptide.find("Z") != -1:
                continue
            if peptide.find("B") != -1:
                continue
            pept_len = len(peptide)
            if pept_len < self.min_pept_len:
                continue

            if pept_len > self.max_pept_len:
                continue

            peptides.append(peptide)
            if self.semi_cleavage == 1:
                for j in range(1, len(peptide)):  # choose the part of peptide
                    rev_j = len(peptide) - j

                    if self.min_pept_len <= rev_j:
                        peptides.append(peptide[j:])

                    if self.min_pept_len <= j:
                        peptides.append(peptide[:rev_j])
        peptides = set(peptides)
        for peptide in peptides:
            if decoy:
                self.add_peptide_collection2(peptide, protein_id, 0)
                continue

            if not self.decoys_only:
                self.add_peptide_collection2(peptide, protein_id, 1)

            decoy_peptide = self.get_decoy(peptide, self.decoy_format)
            if decoy_peptide != None:
                self.add_peptide_collection2(decoy_peptide, protein_id, 0)

    def add_peptide_collection2(self, peptide, protein_id, target):
        # Find multiple occurances of the peptide in its parent protein sequence
        start_end_pos = [
            [x.start(), x.end() - 1]
            for x in re.finditer(peptide, self.protein_collection[protein_id].seq)
        ]

        if self.unique_peptides == 1 and peptide in self.peptide_ids:
            i = self.peptide_ids[peptide]
            if protein_id not in self.peptide_collection[i].protein_id:
                self.peptide_collection[i].protein_id.append(protein_id)
                self.peptide_collection[i].start_end_pos.append(start_end_pos)
        else:

            modified_peptides = set(
                parser.isoforms(
                    peptide, variable_mods=self.modifications, max_mods=self.max_mods
                )
            )

            peptide_aa_mass = np.array(
                [self.aa_mass[aa] for aa in peptide]
            )  # Make this faster using AA as ubytes

            for mod_pept in modified_peptides:

                mod_cnt = len(re.findall(r"\]", mod_pept))
                if mod_cnt < self.min_mods or mod_cnt > self.max_mods:
                    continue

                mod_peptide_aa_mass = list(
                    peptide_aa_mass
                )  # peptide_aa_mass[:] does not work
                offset = 0
                for mod in re.finditer(
                    self.pattern, mod_pept
                ):  # calc modifications' mass
                    location = mod.start()

                    if mod_pept[location] == "-":
                        mod_peptide_aa_mass[-1] += float(mod.group(1))
                        break
                    mod_peptide_aa_mass[location - offset] += float(mod.group(1))
                    offset += mod.end() - mod.start()

                mod_pep_mass = (
                    np.sum(mod_peptide_aa_mass) + self.Y + self.B
                )  # H and OH and static modes
                if (
                    mod_pep_mass < self.min_pept_mass
                    or mod_pep_mass >= self.max_pept_mass
                ):
                    continue

                pept_obj = PeptideObj(
                    mod_pep_mass,
                    mod_pept,
                    protein_id,
                    target,
                    peptide,
                    "full",
                    self.missed_cleavages,
                    [start_end_pos],
                )

                pept_obj.aa_mass = mod_peptide_aa_mass
                self.peptide_collection.append(pept_obj)
                if mod_cnt == 0:
                    self.peptide_set.add(peptide)
                    self.peptide_ids[peptide] = len(self.peptide_collection) - 1

    def get_decoy(self, peptide, format=0):
        if format == 0:  # reverse
            middle = peptide[1:-1][::-1]
            return peptide[0] + middle + peptide[-1]
        if format == 1:  # shuffle
            middle = list(peptide[1:-1])
            random.shuffle(middle)
            return peptide[0] + "".join(middle) + peptide[-1]
        return None

    def sort_peptides(self, key="neutral_mass", reverse=False):
        self.peptide_collection = sorted(
            self.peptide_collection, key=lambda d: getattr(d, key), reverse=reverse
        )  # sort peptides by
        for i, pep in enumerate(self.peptide_collection):
            self.peptide_ids[pep.peptide_seq] = i
        # # neutral_mass in increasing manner

    def calculate_xcorr_score(self, spect_id, pept_id):
        spectrum = self.spectrum_collection[spect_id]

        if not self.peptide_collection[pept_id].peaks:
            self.calculate_peptide_fragmentation(pept_id)

        peaks = []
        for key, peak_list in (
            self.peptide_collection[pept_id].peaks[0].items()
        ):  # Match sinlge charged theoretical peaks
            peaks += peak_list

        if spectrum.charge > 2 and self.max_theo_pept_peak_charge > 1:
            for key, peak_list in (
                self.peptide_collection[pept_id].peaks[1].items()
            ):  # Match double charged theoretical peaks.
                peaks += peak_list
        peaks = np.unique(peaks)
        peaks = peaks - self.spectrum_margin
        score = np.sum(spectrum.spectrum_array[peaks])

        return score / 200

    def tide_search(self):
        """
        Performing Tide search, python reimplementation of Tide-Search from CRUX. See: crux.ms and https://noble.gs.washington.edu/papers/diamen2011faster.pdf
        ------
        Notice:
        1) Assume that the protein fasta  and the spectrum data files are loaded
        2) Assume that all spectra are discretized, normalized and preprocessed with XCORR_substract_background
        3) Spectrum_collection must be sorted by neutral mass in increasing order
        """
       
        for spect_id in range(len(self.spectrum_collection)):
            spectrum = self.spectrum_collection[spect_id]

            for pept_id in range(spectrum.start_pept, spectrum.end_pept):
                
                xcorr = self.calculate_xcorr_score(spect_id, pept_id)*50
                if xcorr > spectrum.score:
                    spectrum.score = xcorr
                    self.peptide_collection[pept_id].peptide_id = pept_id
                    spectrum.peptide = self.peptide_collection[pept_id]
                    
                    
    def calculate_peptide_fragmentation(
        self, peptide_id
    ):  # calculating masses of peptide fragment ions

        peptide = self.peptide_collection[peptide_id]
        peptide.peaks = [dict() for x in range(self.max_theo_pept_peak_charge)]

        for ion_series in self.theo_pept_peaks:

            if ion_series == "b":  # generate B ions.
                fragment_ions = np.cumsum(peptide.aa_mass[:-1]) + (
                    self.B + spytrometer_proton
                )
            if ion_series == "y":  # generate Y ions.
                fragment_ions = np.cumsum(peptide.aa_mass[1:][::-1]) + (
                    self.Y + spytrometer_proton
                )

            for peak_charge in range(self.max_theo_pept_peak_charge):
                fragment_idx = self.mass2bin_vec(fragment_ions, peak_charge + 1)
                peak_list = list(filter(lambda peak: peak < self.max_bin, fragment_idx))
                peptide.peaks[peak_charge][ion_series] = peak_list

    def mass2bin_vec(self, mass, charge=1):  # convert to bin
        return (
            (mass + (charge - 1) * spytrometer_proton) / (charge * self.bin_width)
            + 1.0
            - self.bin_offset
        ).astype(int)

    def plot_peptide(self, peptide_id, filename=None, font_size=24):

        fg = plt.figure(figsize=(20, 5))
        fg.patch.set_facecolor("xkcd:white")
        plt.rc("font", size=font_size)  # controls default text sizes
        plt.rc("axes", titlesize=font_size)  # fontsize of the axes title
        plt.rc("axes", labelsize=font_size)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=font_size)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=font_size)  # fontsize of the tick labels
        plt.rc("legend", fontsize=font_size)  # legend fontsize
        plt.rc("figure", titlesize=font_size)  # fontsize of the figure title
        line_width = 2
        bar_width = 0.5

        spectrum = self.peptide_collection[peptide_id]
        if self.peptide_collection[peptide_id].target == 1:
            target_decoy = "target"
        else:
            target_decoy = "decoy"
        min_y = np.min((np.min(spectrum.spectrum_array), 0))
        max_y = np.max(spectrum.spectrum_array)
        min_x = 0  # np.min(np.nonzero(spectrum.spectrum_array))-1
        max_x = 2000  # np.max(np.nonzero(spectrum.spectrum_array))+1

        plt.xlim(min_x, max_x)
        range_y = max_y - min_y
        tick_height = range_y * 0.02
        plt.ylim(0, 0.4)
        cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        grey = (0.7, 0.7, 0.7)
        plt.bar(
            np.nonzero(spectrum.spectrum_array)[0],
            spectrum.spectrum_array[np.nonzero(spectrum.spectrum_array)],
            width=bar_width,
            linewidth=bar_width,
            color=grey,
            edgecolor=grey,
        )
        # Line for the base
        plt.plot([min_x, max_x], [0, 0], linewidth=1, color="black")
        plt.ylabel("Intensity")
        plt.xlabel("m/z")
        plt.title(
            "Peptide: {} ({})".format(
                self.peptide_collection[peptide_id].peptide_seq, target_decoy
            )
        )
        plt.grid(True)

        if filename is not None:

            plt.savefig(filename + ".pdf", bbox_inches="tight")
            plt.clf()
        plt.close()

    def plot_spectrum(
        self,
        spectrum_id,
        show_annotation=True,
        peptide_seq=None,
        filename=None,
        font_size=24,
    ):
        fg = plt.figure(figsize=(20, 5))
        fg.patch.set_facecolor("xkcd:white")
        plt.rc("font", size=font_size)  # controls default text sizes
        plt.rc("axes", titlesize=font_size)  # fontsize of the axes title
        plt.rc("axes", labelsize=font_size)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=font_size)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=font_size)  # fontsize of the tick labels
        plt.rc("legend", fontsize=font_size)  # legend fontsize
        plt.rc("figure", titlesize=font_size)  # fontsize of the figure title
        line_width = 2
        bar_width = 0.5

        spectrum = self.spectrum_collection[spectrum_id]
        min_y = np.min((np.min(spectrum.spectrum_array), 0))
        max_y = np.max(spectrum.spectrum_array)
        min_x = 0#np.min(np.nonzero(spectrum.spectrum_array))-1
        max_x = 2000#np.max(np.nonzero(spectrum.spectrum_array))+1

        plt.xlim(min_x, max_x)
        range_y = max_y - min_y
        tick_height = range_y * 0.02
        cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        grey = (0.7, 0.7, 0.7)
        plt.bar(
            np.nonzero(spectrum.spectrum_array)[0],
            spectrum.spectrum_array[np.nonzero(spectrum.spectrum_array)],
            width=bar_width,
            linewidth=bar_width,
            color=grey,
            edgecolor=grey,
        )
        # Line for the base
        plt.plot([min_x, max_x], [0, 0], linewidth=1, color="black")

        plt.ylabel("Relative intensity")
        plt.xlabel("m/z")
        plt.grid(True)
        plt.yticks((0.0, 0.5, 1.0))

        if show_annotation == True:
            peptide = peptide_seq

            print(
                "Seq:{}  Charge:{}   Mass: {}".format(
                    peptide.peptide_seq, spectrum.charge, peptide.neutral_mass
                )
            )
            cnt = 1
            # Line for b fragments
            #plt.plot([min_x, max_x],[max_y+range_y*0.1,max_y+range_y*0.1], linewidth=line_width)
            # Line for y fragments
            #plt.plot([min_x, max_x],[max_y+range_y*0.2,max_y+range_y*0.2], linewidth=line_width)
            for charge in range(len(peptide.peaks)):
                # Use theoretical peaks which were used in scoring
                if spectrum.charge < 3 and charge > 0:
                    continue
                for ion_type in peptide.peaks[charge].keys():
                    # Get the theoretical peaks
                    peaks = np.asarray(peptide.peaks[charge][ion_type], dtype=np.int32)
                    # Line for ion series
                    # plt.plot([min_x, max_x],[max_y+range_y*cnt/10,max_y+range_y*cnt/10], linewidth=line_width, color=cmap[cnt])
                    # Place ticks for the ion fragments in the line
                    peak_prev = min_x
                    for peak in peaks:
                        plt.bar(
                            peak,
                            spectrum.spectrum_array[peak],
                            width=line_width,
                            color=cmap[cnt],
                            edgecolor=cmap[cnt],
                        )
                        if (
                            spectrum.spectrum_array[peak] > 0
                        ):  # take into account only positive matches
                            # Place tick and mark experimental peak
                            plt.plot(
                                [peak_prev, peak],
                                [
                                    max_y + range_y * cnt / 10,
                                    max_y + range_y * cnt / 10,
                                ],
                                linewidth=line_width,
                                color=cmap[cnt],
                            )
                            plt.plot(
                                [peak, peak],
                                [
                                    max_y + range_y * cnt / 10 - tick_height,
                                    max_y + range_y * cnt / 10 + tick_height,
                                ],
                                linewidth=line_width,
                                color=cmap[cnt],
                            )
                        else:
                            # Place small tick if the theoretical peak did not match
                            plt.plot(
                                [peak_prev, peak],
                                [
                                    max_y + range_y * cnt / 10,
                                    max_y + range_y * cnt / 10,
                                ],
                                linestyle="--",
                                linewidth=line_width,
                                color=cmap[cnt],
                            )
                            plt.plot(
                                [peak, peak],
                                [
                                    max_y + range_y * cnt / 10 - tick_height / 2,
                                    max_y + range_y * cnt / 10 + tick_height / 2,
                                ],
                                linewidth=bar_width,
                                color=cmap[0],
                            )
                        peak_prev = peak
                    # Get the position for the amino acids
                    AA_pos = np.concatenate(
                        ([peaks[0] / 2], (peaks[:-1] + peaks[1:]) / 2)
                    )
                    # Get the peptide sequence
                    pept_seq = peptide.peptide_seq
                    try:
                        modification = (
                            "[" + re.search(r"\[(.*?)\]", pept_seq).group(1) + "]"
                        )
                        pept_seq = list(re.sub(r"\[(.*?)\]", "!", pept_seq))
                        pept_seq[pept_seq.index("!") + 1] = (
                            modification + pept_seq[pept_seq.index("!") + 1]
                        )
                        del pept_seq[pept_seq.index("!")]
                    except:
                        pept_seq = list(pept_seq)
                    if ion_type == "b":
                        pept_seq = pept_seq[:-1]
                    # Revers the peptide sequence
                    if ion_type == "y":
                        pept_seq = pept_seq[1:][::-1]
                    plt.text(
                        0,
                        max_y + range_y * 1 / 10 + range_y * 0.02,
                        " b-ion",
                        horizontalalignment="left",
                        verticalalignment="center",
                        fontsize=font_size - 12,
                    )
                    plt.text(
                        0,
                        max_y + range_y * 2 / 10 + range_y * 0.02,
                        " y-ion",
                        horizontalalignment="left",
                        verticalalignment="center",
                        fontsize=font_size - 12,
                    )
                    for i in range(len(AA_pos)):
                        if AA_pos[i] > max_x:
                            break
                        if i == 0:
                            plt.text(
                                AA_pos[i],
                                max_y + range_y * cnt / 10 + range_y * 0.02,
                                pept_seq[i] + " [TMT6plex]",
                                horizontalalignment="center",
                                verticalalignment="center",
                                fontsize=font_size - 12,
                            )
                        else:
                            plt.text(
                                AA_pos[i],
                                max_y + range_y * cnt / 10 + range_y * 0.02,
                                pept_seq[i],
                                horizontalalignment="center",
                                verticalalignment="center",
                                fontsize=font_size - 12,
                            )
                    cnt += 1

        if filename is not None:
            # plt.savefig(filename+".pdf", bbox_inches='tight')
            plt.savefig(filename + ".pdf", bbox_inches="tight")
            plt.clf()
        plt.xlim(min_x, max_x)
        plt.close()


    def plot_intensity_change(self, filename):
        import copy

        font_size = 18
        fg = plt.figure(figsize=(10, 5))
        fg.patch.set_facecolor("xkcd:white")
        plt.rc("font", size=font_size)  # controls default text sizes
        plt.rc("axes", titlesize=font_size)  # fontsize of the axes title
        plt.rc("axes", labelsize=font_size)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=font_size)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=font_size)  # fontsize of the tick labels
        plt.rc("legend", fontsize=font_size)  # legend fontsize
        plt.rc("figure", titlesize=font_size)  # fontsize of the figure title
        line_width = 2
        bar_width = 2
        resolution = 1000
        cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        grey = (0.7, 0.7, 0.7)
        frequency_bins_match = np.zeros((resolution * 3 + 5))
        frequency_bins_nomatch = np.zeros((resolution * 3 + 5))
        for spectrum_id, spectrum in enumerate(self.spectrum_collection):
            # print(spectrum.qvalue)
            if spectrum.qvalue > 0.01:
                break
            spectrum_array = copy.deepcopy(spectrum.spectrum_array)
            self.discretize_spectrum(spectrum_id)
            self.normalize_regions(spectrum_id)

            original_array = copy.deepcopy(spectrum.spectrum_array)
            spectrum.spectrum_array = spectrum_array
            diff = spectrum_array - original_array
            diff = (np.around(np.around(diff * resolution) + resolution + 1)).astype(
                np.int
            )
            diff_match = diff[spectrum.peptide.unique_peaks]
            # [print(i) for i in diff_match]
            diff_nomatch = np.delete(diff, spectrum.peptide.unique_peaks)
            # print(diff)
            for i in diff_match:
                # if i != resolution + 1:
                frequency_bins_match[i] += 1
            for i in diff_nomatch:
                # if i != resolution + 1:
                frequency_bins_nomatch[i] += 1
            # break
        frequency_bins_match /= frequency_bins_match.sum()
        frequency_bins_nomatch /= frequency_bins_nomatch.sum()
        plt.plot(
            range(resolution * 3 + 5),
            frequency_bins_nomatch,
            linewidth=bar_width,
            color=grey,
            label="non-matching peaks",
        )
        plt.plot(
            range(resolution * 3 + 5),
            frequency_bins_match,
            linewidth=bar_width,
            color=cmap[1],
            label="matching peaks",
        )
        # plt.plot(np.nonzero(frequency_bins_nomatch)[0], frequency_bins_nomatch[np.nonzero(frequency_bins_nomatch)],linewidth=bar_width, color=grey, label='non-matching peaks')
        # plt.plot(np.nonzero(frequency_bins_match)[0], frequency_bins_match[np.nonzero(frequency_bins_match)],  linewidth=bar_width, color=cmap[1], label='matching peaks')
        xticks = range(0, resolution * 3, 200)
        plt.xticks(xticks, [(i - resolution) / resolution for i in xticks])
        plt.xlim(400, 1600)
        plt.ylim(0, 0.02)
        plt.box(True)
        plt.ylabel("Frequency")
        plt.xlabel("Peak intensity change")

        plt.legend()
        if filename is not None:
            # plt.savefig(filename+".pdf", bbox_inches='tight')
            plt.savefig(filename + "intensity_change.pdf", bbox_inches="tight")
            plt.clf()

    def plot_qvalues(self, mode="show", filename=None, qval_lim=0.1):
        fig = plt.figure(figsize=(10, 6))

        x = []
        y = []
        qval = 0
        # qval_lim = 0.1
        accepted_psm = 0
        for spectrum in self.spectrum_collection:
            if qval > qval_lim:
                break
            if qval != spectrum.qvalue:
                x.append(qval)
                y.append(accepted_psm)
                qval = spectrum.qvalue
            try:
                if spectrum.peptide.target == True:
                    accepted_psm += 1
            except:
                pass

        x.append(qval)
        y.append(accepted_psm)

        plt.plot(x, y)
        plt.xlim([0, qval_lim])
        # plt.ylim([0, 2500])
        plt.ylabel("Number of accepted spectra")
        plt.xlabel("False Discovery Rate")
        fig.patch.set_facecolor("xkcd:white")
        if mode == "show":
            plt.show()
        elif mode == "save":
            fig.savefig(filename)
            plt.clf()
        plt.close()
        
    def print_results(self, filename):
        fout = open(filename, "w")

        names_of_columns = [
            "file",
            "scan",
            "spectrum_id",
            "charge",
            "spectrum precursor m/z",
            "spectrum neutral mass",
            "peptide mass",
            "score",
            "confidence",
            "qvalue",
            "number of candidates",
            "target",
            "protein id",
            "peptide_id",
            "peptide sequence",
            "peptide length",
            "modifications",
            "cleavage type",
            "missed cleavages",
            "original sequence",
            "quintile",
        ]

        header = "\t".join(names_of_columns) + "\n"
        fout.writelines(header)

        for spectrum_id, spectrum in enumerate(self.spectrum_collection):
            result_string = spectrum.print_spectrum(spectrum_id)
            if result_string:
                fout.writelines(
                    list(
                        "{:d}\t".format(item)
                        if type(item) == int
                        else "{:f}\t".format(item)
                        if type(item) == float
                        else "{:s}\t".format(item)
                        for item in result_string
                    )
                )
                fout.write("\n")

        fout.close()
