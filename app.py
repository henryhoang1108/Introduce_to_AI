import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from model import initialize_model

class WeatherClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phân loại ảnh thời tiết")
        self.root.geometry("600x500")

        self.class_names = ["Sương mù", "Bình thường", "Mưa", "Tuyết"]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = initialize_model(num_classes=4, device=self.device, id=15)
        self.model.load_state_dict(torch.load("/home/henryhoang/Project/WeatherClassification/TransferLearning/best_model_resnet18_1.pth", map_location=self.device, weights_only=True))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.label = Label(root, text="Choose picture", font=("Arial", 14))
        self.label.pack(pady=10)

        self.image_label = Label(root)
        self.image_label.pack(pady=10)

        self.result_label = Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

        self.browse_button = Button(root, text="Browse", command=self.browse_image, font=("Arial", 14))
        self.browse_button.pack(pady=10)

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            img = Image.open(file_path)
            img = img.resize((300, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=photo)
            self.image_label.image = photo

            label, confidence = self.predict_image(file_path)
            self.result_label.configure(text=f"Dự đoán: {label} - {confidence:.2f}%")

    def predict_image(self, file_path):

        img = Image.open(file_path).convert("RGB")
        img = self.transform(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            img = img.to(self.device)
            outputs = self.model(img)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_label = self.class_names[predicted.item()]
            confidence = confidence.item() * 100

        return predicted_label, confidence

if __name__ == "__main__":
    root = tk.Tk()
    app = WeatherClassificationApp(root)
    root.mainloop()