from flask import Flask, request, jsonify
from torchvision import models, transforms
from PIL import Image
import torch

app = Flask(__name__)

# Set up transformations for images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load class names
with open("imagenet_classes.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

@app.route('/')
def home():
    return app.send_static_file('main.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    image = request.files['image']
    img = Image.open(image).convert('RGB')

    # Ensure the image is loaded and converted properly
    if img is None:
        return jsonify({'error': 'Failed to load or convert the image'})

    print("Image shape:", img.size)  # Print the shape of the image
    img = transform(img).unsqueeze(0)

    model_name = request.form['model']  

    if model_name == 'model1':
        model_path = "model_resnet.pt"
    elif model_name == 'model2':
        model_path = "my_model.pt"
    elif model_name == 'model3':
        model_path = "ozel_model.pt"
    else:
        return jsonify({'error': 'Invalid model selection'})

    # Load the selected model
    model = torch.jit.load(model_path, map_location=torch.device('cpu'))
    model.eval()

    with torch.no_grad():
        output = model(img)
        _, prediction = torch.max(output, 1)

    # Convert the prediction to class name
    class_name = class_names[prediction.item()]

    return jsonify({'prediction': class_name})

if __name__ == '__main__':
    app.run(debug=True)
