import pickle
import numpy as np
from PIL import Image


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from tensorflow.keras.applications.vgg16 import  VGG16 ,preprocess_input
from tensorflow.keras.models import Model

def open_image(image_file):
	img = Image.open(image_file)
	return img

def vgg_feature_extractor(image):
    vgg_extract = VGG16()
    vgg_extract = Model(inputs=vgg_extract.inputs , outputs=vgg_extract.layers[-2].output)
    feature_array = vgg_extract.predict(image , verbose=0)
    return feature_array

def preprocess_image(img_path):
    image = load_img(img_path , target_size=(224,224)) ##resizing the input image
    image = img_to_array(image)
    image = image.reshape((1,224,224,3))
    ##process the image for vgg model
    image = preprocess_input(image)
    feature_array=vgg_feature_extractor(image)
    return feature_array

def idx_to_word(integer , tokenizer):
  for word , index in tokenizer.word_index.items():
    if index == integer:
      return word
  return None

def predict_caption(model,image,tokenizer,max_length=35):
    ##add start tag for generation process
    in_text = '<start>'
    # iterate over the max length if sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        ##pad the sequence
        sequence = pad_sequences([sequence] , max_length)
        #predict next word
        yhat = model.predict([image,sequence] , verbose=0)
        #get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat , tokenizer)
        ##stop if word not found
        if word is None:
            break
        #append word as input for generating next word
        in_text += " " + word
        #stop if we reach end tag
        if word == 'end':
            break

    in_text =in_text.replace('end' , '<end>')
    
    return in_text 