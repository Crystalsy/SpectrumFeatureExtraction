# Feature Extraction Using MS/MS Spectra Datatset
Trained for extract significant features of tryptic peptide tandem ms dataset 
  # Introduction
  In this project, we used MS/MS spectra identified by Andromeda search software to extract features of spectra. Since this dataset is sequential, we used various recurrent neural network models-Long Short-Term Memory (LSTM), Bidirectional LSTM (BiLSTM)-and compared the performance of each models.
  
  # Dataset
  The Original dataset is PXD004732 dataset of 9,960,509 spectra represented by tandem ms over 350,000 human tryptic peptides from the ProteomeTools Consortium. We extracted only 791,364 spectra of which used Higher-Energy Collisional Dissociation (HCD) and which normalized collisional energy is over 35%, to study spectra made by same experimental process.
  
![pxd004732](https://user-images.githubusercontent.com/52642328/60783392-6cd2fe00-a185-11e9-9f8b-2819d5c78619.PNG)

  
