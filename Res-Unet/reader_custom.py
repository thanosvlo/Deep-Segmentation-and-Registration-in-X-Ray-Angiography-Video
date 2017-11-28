import SimpleITK as sitk
import os
from dltk.io.augmentation import *
from dltk.io.preprocessing import *
import tensorflow as tf

def read_fn (file_references,mode,params=None):
    def _augment(img, lbl):
        """An image augmentation function"""
        img = add_gaussian_noise(img, sigma=0.1)
        [img, lbl] = flip([img, lbl], axis=1)

        return img, lbl

    for meta_data in file_references:
        subject_id=meta_data[0]
        img_fn=meta_data[1]

        # Read the image - tif - with cv2 
        sitk_img=sitk.ReadImage(str(img_fn))
        img=sitk.GetArrayFromImage(sitk_img)

        img=whitening(img)

        img=img[np.newaxis,...]
      

        if mode==tf.estimator.ModeKeys.PREDICT:
            yield{'features':{'x':img},
                    'metadata':{
                        'subject_id':subject_id,
                        'sitk':sitk_img
                    }}

        no_ext=img_fn.split('.')
        no_ext=no_ext[0]
        lbl = sitk.GetArrayFromImage(sitk.ReadImage(
            os.path.join(str(no_ext)+'_mask.tif'))).astype(np.int32)
        lbl=lbl[:,:,0]
        lbl=lbl[np.newaxis,...]

        
        # Augment if used in training mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            img, lbl = _augment(img, lbl)
        
        if params['extract_examples']:
            n_examples = params['n_examples']
            example_size = params['example_size']

            images, lbl = extract_class_balanced_example_array(
                image=images,
                label=lbl,
                example_size=example_size,
                n_examples=n_examples,
                classes=2)

            for e in range(n_examples):
                yield {'features': {'x': images[e].astype(np.float32)},
                       'labels': {'y': lbl[e].astype(np.int32)},
                       'subject_id': subject_id}
        else:
            
            yield{ 'features':{'x':img},
                'labels':{'y':lbl},
                'metadata':{
                    'subject_id':subject_id,
                    'sitk':sitk_img
                }

        }
    return
