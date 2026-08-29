# V14 ECG Encoder

`v14_ecg_encoder.keras` is a one-dimensional convolutional autoencoder encoder
trained from scratch on Lead II ECG signals from MIMIC-IV-ECG v1.0. It produces
a 16-dimensional latent representation (`z1`-`z16`) from a 10-second Lead II
segment resampled to 250 Hz.

The model was trained by the project author and is released under the MIT
License. See the repository `LICENSE` file.

The underlying ECG data are available from PhysioNet under the MIMIC-IV-ECG
terms; the weights themselves are distributed under MIT.
