#Import required libraries
import os
import torch

import PIL
from PIL import Image

from diffusers import StableDiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from huggingface_hub import hf_hub_download

# Download the model to the local folders
def hf_hub_download_to_local(downloaded_embedding_folder, embeds_url, repo_id_embeds, access_token, placeholder_token_string):
  if not os.path.exists(downloaded_embedding_folder):
    os.mkdir(downloaded_embedding_folder)
  if(not embeds_url):
    embeds_path = hf_hub_download(repo_id=repo_id_embeds, filename="learned_embeds.bin", use_auth_token=access_token)
    token_path = hf_hub_download(repo_id=repo_id_embeds, filename="token_identifier.txt", use_auth_token=access_token)
    os.system("cp " + embeds_path + " " + downloaded_embedding_folder)
    os.system("cp " + token_path + " " + downloaded_embedding_folder)
    with open(f'{downloaded_embedding_folder}/token_identifier.txt', 'r') as file:
      placeholder_token_string = file.read()
  else:
    os.system("wget -q -O " + downloaded_embedding_folder + "/learned_embeds.bin $" + embeds_url)


# Load the newly learned embeddings into CLIP
def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
  loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

  # separate token and the embeds
  trained_token = list(loaded_learned_embeds.keys())[0]
  embeds = loaded_learned_embeds[trained_token]

  # cast to dtype of text_encoder
  dtype = text_encoder.get_input_embeddings().weight.dtype
  embeds.to(dtype)

  # add the token in tokenizer
  token = token if token is not None else trained_token
  num_added_tokens = tokenizer.add_tokens(token)
  if num_added_tokens == 0:
    raise ValueError(f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")

  # resize the token embeddings
  text_encoder.resize_token_embeddings(len(tokenizer))

  # get the id for the token and assign the embeds
  token_id = tokenizer.convert_tokens_to_ids(token)
  text_encoder.get_input_embeddings().weight.data[token_id] = embeds


# Load the pretrained model - tokenizer and encoder
def load_pretrained(pretrained_model_name_or_path, access_token):
  tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
    use_auth_token=access_token
  )
  text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder",
    use_auth_token=access_token
  )

  return tokenizer, text_encoder

# Load the pipeline for the model with the pretrained components
def load_pipe(pretrained_model_name_or_path, text_encoder, tokenizer, access_token, device):
  
  if device == 'mps':

    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        revision="fp16",
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        use_auth_token=access_token,
    )

  else:
    pipe = StableDiffusionPipeline.from_pretrained(
      pretrained_model_name_or_path,
      revision="fp16",
      torch_dtype=torch.float16,
      text_encoder=text_encoder,
      tokenizer=tokenizer,
      use_auth_token=access_token
    )

  # helps in a low availability GPU environment - took M2 mac from 15 minutes to 6.05s/it i.e. 5min 56sec
  pipe.enable_attention_slicing()
  pipe = pipe.to(device)
  
  return pipe

# Base set up to be called first to set parameters
def model_setup(concept_str):
  # set access token for HuggingFace Hub
  access_token = <ACCESS TOKEN HERE>
  # set the string for the pretrained model
  pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4" 

  # can swap this out to run on non apple silicon
  device = "mps" #"cuda"

  # Load model concept 
  repo_id_embeds = "sd-concepts-library/" + concept_str


  embeds_url = "" #Add the URL or path to a learned_embeds.bin file in case you have one
  placeholder_token_string = "" #Add what is the token string in case you are uploading your own embed
  downloaded_embedding_folder = "./downloaded_embedding"
  learned_embeds_path = f"{downloaded_embedding_folder}/learned_embeds.bin"
  hf_hub_download_to_local(downloaded_embedding_folder, embeds_url, repo_id_embeds, access_token, placeholder_token_string)

  return access_token, pretrained_model_name_or_path, learned_embeds_path, device