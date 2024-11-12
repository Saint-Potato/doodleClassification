import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torch
import torch.nn as nn  

in_channels = 1
num_classes = 6

class CNN(nn.Module):
    def __init__(self, in_channels = in_channels, num_classes = num_classes ):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 14x14x5
            nn.Conv2d(5, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 7x7x8
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU() # 7x7x16
        )
        
        self.linear = nn.Sequential(
            nn.Linear(7*7*16, 100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(100, num_classes)
        )
                
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)  # Ensure input shape is correct for Conv2d
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out

model = CNN() 
model.load_state_dict(torch.load("doodle_model.pth"))
model.eval()  # Set model to evaluation mode

class DoodleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Doodle Canvas")
        
        # Create a canvas widget for drawing
        self.canvas = tk.Canvas(root, bg="white", width=280, height=280)
        self.canvas.pack()

        # Set up a PIL image to draw on
        self.image = Image.new("L", (280, 280), "white")  # L mode for grayscale
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.paint)

        # Button to clear the canvas
        self.clear_button = tk.Button(root, text="Clear", command=self.clear)
        self.clear_button.pack(side=tk.LEFT)
        
        # Button to predict the doodle
        self.predict_button = tk.Button(root, text="Predict", command=self.predict_doodle)
        self.predict_button.pack(side=tk.RIGHT)

        # Mapping of labels to class names
        self.class_names = ["airplane", "apple", "banana", "bird", "bicycle", "clock"]

    def paint(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill="white")

    def preprocess_image(self):
        # Step 1: Resize to 28x28 and invert colors
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        
        # Step 2: Convert to numpy array, scale to [0, 1]
        img = np.array(img, dtype=np.float32) / 255.0
        img = img.reshape(1, 1, 28, 28)  # Reshape for model input

        # Convert to tensor (no additional normalization yet)
        img_tensor = torch.tensor(img, dtype=torch.float32)
        print("Processed image tensor:", img_tensor.shape)  # Debugging
        return img_tensor

    def predict_doodle(self):
        img_tensor = self.preprocess_image()
        
        with torch.no_grad():  # Disable gradients for inference
            output = model(img_tensor)
            print("Model output logits:", output)  # Debugging logits for variability
            prediction = output.argmax(dim=1).item()
            predicted_label = self.class_names[prediction]
        
        print(f"Predicted label: {predicted_label}")
# Run the app
root = tk.Tk()
app = DoodleApp(root)
root.mainloop()
