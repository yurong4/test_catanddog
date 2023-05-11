import gradio as gr
from fastai.vision.all import *

def is_cat(x): return x[0].isupper() 


learn = load_learner('model.pkl')

categories =('dog','cat')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories,map(float,probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['dog.jpg','cat.jpg','dunno.jpg']

demo = gr.Interface(fn=classify_image, inputs=image, outputs=label , examples=examples)

demo.launch(inline=False)