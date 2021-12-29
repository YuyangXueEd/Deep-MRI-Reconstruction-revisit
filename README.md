# Deep MRI Reconstruction 

## Dataset

* [3T dataset](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release)
    * This 3T fMRI dataset is used for the fMRI courses at the University of Amsterdam.  The complete dataset is hosted under the name "NI-edu-data-complete".
    * Testing the model on 
        * Unseen 3T MRI images
        * Noisy 3T MRI images
    * Use PSNR to evaluate the performance of the reconstructed images

## Brief Introduction on MR Images

A variety of systems are used in medical imaging ranging from open MRI units with magnetic field strength of 0.3 Tesla (T) to extremity MRI systems with field strengths up to 1.0 T and whole-body scanners with field strengths up to 3.0 T (in clinical use).

Tesla is the unit of measuring the quantitative strength of magnetic field of MR images.

High field MR scanners (7T, 11.5T) yielding higher SNR (signal-to-noise ratio) even with smaller voxel (a 3-dimensional patch or a grid) size and are thus preferred for more accurate diagnosis.

### Understanding the Brain MRI 3T Dataset

The brain MRI dataset consists of 3D volumes each volume has in total 207 slices/images of brain MRI's taken at different slices of the brain. Each slice is of dimension `173 x 173`. The images are single channel grayscale images. There are in total 30 subjects, each subject containing the MRI scan of a patient. The image format is not jpeg,png etc. but rather nifti format. You will see in later section how to read the nifti format images.

## CNN and Autoencoder Tutorial