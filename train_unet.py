from tf_unet import unet, util, image_util
import numpy as np 

#preparing data loading
data_provider = image_util.ImageDataProvider("/Users/thanosvlo/Desktop/FYP/Images/Wire/Train/*.tif")
data=image_util.ImageDataProvider("/Users/thanosvlo/Desktop/FYP/Images/Wire/Test/*.tif")
output_path="/Users/thanosvlo/Desktop/FYP/Unet_checkpoints/"
#setup & training
net = unet.Unet(layers=3, features_root=64, channels=1, n_class=2)
trainer = unet.Trainer(net)
path = trainer.train(data_provider, output_path, training_iters=30, epochs=100)

#verification


prediction = net.predict(path, data)

unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))

img = util.combine_img_prediction(data, label, prediction)
util.save_image(img, "prediction.jpg")