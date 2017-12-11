res_unet based on https://github.com/DLTK/DLTK
-

installation / use:
- Follow installation instructions for DLTK ,
- add the files in the path : /DLTK-master/examples/applications/MRBrainS13_tissue_segmentation/
-you need a .csv file where each line is a data point in the form :
        subject id , path/to/image
the masks should have the same path and name as the images but add the suffix '_mask' in the name.


This implementation fed with .png images and in the deploy phase it exports .nii.gz - I suggest FIJI or ImageJ to open them

PERFORMANCE :
Dice distance of about 0.9
inference speed using GTX TITAN ~ 22fps

TODO - Better Documentation

This is an initial version of the documentation since project is yet unfinished.

Any issues let me know
