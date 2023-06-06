# requment.txt:
# flask
# transformers
# accelerate
# bitsandbytes
from flask import Flask, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

checkpoint = "bigscience/bloomz"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

@app.route('/api/prompt', methods=['POST'])
def generate_response():
    data = request.get_json()
    prompt = data['prompt']

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs)

    response = tokenizer.decode(outputs[0])
    return {'response': response}

@app.route('/')
def home():
    return {'response': "ok"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
