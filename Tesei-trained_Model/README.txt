This readme file was generated on 2024-07-23 by Lilianna Houston

GENERAL INFORMATION

Title of Project: PML, Tesei-trained Model 

Principal Investigator Information
Name: Kingshuk Ghosh
Institution: University of Denver
Email: kingshuk.ghosh@du.edu

Author Information
Name: Lilianna Houston
Institution: University of Denver
Email: lili.houston@du.edu

DATA & FILE OVERVIEW

File List: 

"weights" -> Folder containing weights from the Tesei-trained CNN that predicts omega_2 from sequence. We trained the model 10 separate times on all omega_2 calculations from the Tesei 2023 dataset and provide all 10 resulting weights.

"Tesei_w2_Ree_preds" -> CSV containing calculated and ML predicted omega_2 (w2) (predicted using 10 fold cross-validation), as well as reported and predicted R_ee for the Tesei 2023 dataset. Sequences were omega_2 calculation failed are omitted.

"exper_seqs_master" -> CSV of our compiled experimental sequences, including source, sequnces, salt, pH, temperature, reported R_g and our predicted R_g (our value is averaged across the results of all 10 trained models). This is used as the input file for extract_w2.py [Use a different csv if you want to use a different sequence or set of sequences.]

"extract_w2" -> .py file that extracts the omega_2s of a specified list of sequences using a specified set of weights. Make sure you use the correct input file if you want to change the current input file. Also change the output file at the end of the code if you change the input file.

"exper_seqs_w2preds" -> CSV file. Same content as "exper_seqs_master," with the addition of predicted w2s using weights_0 from the "weights" folder. This is used as the input for extract_Rg

"extract_Rg" -> .py file that extracts the x, R_ee, and R_gs of a specified list of sequences using omega_2. Currently w2 is obtained from exper_seqs_w2preds but use a different one if you used a different output above.

"OBfmt_5-1500.npy" -> Helper file for "extract_Rg" containing precalulated terms.
"theory_functions" -> .py helper file for "extract_Rg" containing constants and functions needed for R_g calculation.