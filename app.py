from flask_ngrok import run_with_ngrok
from flask import Flask, request, render_template
from stable_diffusion_setup import *
import io
import base64


app = Flask(__name__)


# set up model in advance of app running
print("DOWNLOADING PRE-TRAINED")
concept_str = <CONCEPT STRING HERE>
access_token, pretrained_model_name_or_path, learned_embeds_path, device = model_setup(concept_str)
# get pretrained tokenizer and embeddings
print("LOAD MODELS")
tokenizer, text_encoder = load_pretrained(pretrained_model_name_or_path, access_token)
load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer)
# create pipe for the HuggingFace Models
print("CREATING MODEL PIPELINE")
pipe = load_pipe(pretrained_model_name_or_path, text_encoder, tokenizer, access_token, device)
pipe = pipe.to(device)


@app.route("/", methods=["GET"])
def home():
  return render_template("index.html")

@app.route("/", methods=["POST"])
def show_image():

  prompt_str = request.form['prompt']
  
  print('Calling Generation')
  image = create_image_from_prompt(prompt_str, pipe, 1)

  # convert image to buffer to post to frontend to render
  buffered = io.BytesIO()
  image.save(buffered, format="JPEG")
  img_str = base64.b64encode(buffered.getvalue())

  return render_template('index.html', image_data=img_str.decode('utf-8'))

# Run the HuggingFace Stable Diffusion Pipe with the users prompt
def create_image_from_prompt(prompt, pipe, num_samples):
  image = pipe(prompt, num_images_per_prompt=num_samples, num_inference_steps=50, guidance_scale=7.5).images[0]
  image.save(f"generated.png")
  return image


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8000)
  run_with_ngrok(app)