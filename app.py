
import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('bulldog-or-mini-schnauzer.pkl')

labels = learn.dls.vocab

def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Bulldog and Mini Schnauzer Image Classifier"
description = "<b>You can click on the example images below for an easy way to test or you can upload images from your machine</b>"
article="<p style='text-align: center'><a href='http://nowhere.com' target='_blank'>Blog post</a></p>"
examples = ['bulldog.jpeg', 'mini-schnauzer.jpeg', 'snaggle-tooth.jpeg', 'little-puppy.jpeg']
interpretation='default'
enable_queue=True

app = gr.Interface(fn=predict,inputs=gr.Image(shape=(512, 512)),outputs=gr.Label(num_top_classes=3),title=title,description=description,article=article,examples=examples,interpretation=interpretation)
app.launch(enable_queue=enable_queue)