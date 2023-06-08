# requment.txt:
# flask
# transformers
# accelerate
# bitsandbytes

from flask import Flask, request

from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

checkpoint = "bigscience/bloomz-7b1-mt"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", load_in_8bit=True)

# Use CUDA if available




@app.route('/api/prompt', methods=['POST'])
def generate_response():
    data = request.get_json()
    prompt = data['prompt']

    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs)

    response = tokenizer.decode(outputs[0])
    return {'response': response}

@app.route('/')
def home():
    return {'response': "ok"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
