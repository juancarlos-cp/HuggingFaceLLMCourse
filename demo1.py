import gradio as gr
from transformers import pipeline

model = pipeline("text-generation")


def predict(prompt):
    completion = model(prompt)[0]["generated_text"]
    return completion

def greet(name):
    return "Hello " + name

# We instantiate the Textbox class
# textbox = gr.Textbox(label="Type your name here:", placeholder="John Doe", lines=2)

# gr.Interface(fn=greet, inputs=textbox, outputs="text").launch()
# all I really do is change the function
gr.Interface(fn=predict, inputs="text", outputs="text").launch()

