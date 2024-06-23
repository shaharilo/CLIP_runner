# # import autogluon.core as ag
# from autogluon.vision import ImagePredictor
# # from autogluon.tabular import TabularPredictor
# # import pandas as pd
# # import numpy as np
# import torch
# import clip
# from PIL import Image
#
#
# def load_model(device='cpu'):
#     # Load the CLIP model
#     device = torch.device(device)
#     model, preprocess = clip.load("ViT-B/32", device=device)
#     model.eval()
#     return model, preprocess,device
#
#
# def get_features(model, preprocess, device, images):
#     # Preprocess images and obtain features
#     image_features = []
#     for image in images:
#         image_input = preprocess(image).unsqueeze(0).to(device)
#         with torch.no_grad():
#             image_features.append(model.encode_image(image_input))
#     return torch.stack(image_features)
#
#
# def classify_photos(images):
#     # Load CLIP model
#     model, preprocess,device = load_model()
#
#     # Get image features
#     image_features = get_features(model, preprocess, device, images)
#
#     # Load the image classification model trained using AutoGluon
#     predictor = ImagePredictor.load("autogluon_models/ImageNetV2-FineTune_CLIP-ViT-B_32")
#
#     # Predict image classes
#     predictions = predictor.predict_proba(image_features.cpu().numpy())
#
#     return predictions
#
#
# if __name__ == "__main__":
#     # Example images for classification
#     images = [Image.open("dog.jpg")]
#
#     # Classify photos
#     predictions = classify_photos(images)
#
#     # Print results
#     print(predictions)

#--------------------------------------try it only on one photo - WORKED--------------------------------------------------
#
# import torch
# import clip
# from PIL import Image
#
# def load_clip_model(device='cpu'):
#     # Load the CLIP model
#     device = torch.device(device)
#     model, preprocess = clip.load("ViT-B/32", device=device)
#     model.eval()
#     return model, preprocess, device
#
# def classify_image(image_path, labels):
#     # Load the model and preprocess
#     model, preprocess, device = load_clip_model()
#
#     # Preprocess the image
#     image = Image.open(image_path)
#     image_input = preprocess(image).unsqueeze(0).to(device)
#
#     # Preprocess the labels (text descriptions)
#     text_inputs = torch.cat([clip.tokenize(label) for label in labels]).to(device)
#
#     # Get image and text features
#     with torch.no_grad():
#         image_features = model.encode_image(image_input)
#         text_features = model.encode_text(text_inputs)
#
#     # Calculate cosine similarity between image and text features
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)
#     similarities = (image_features @ text_features.T).squeeze(0)
#
#     # Get the best matching label
#     best_match_index = similarities.argmax().item()
#     best_match_label = labels[best_match_index]
#
#     return best_match_label, similarities
#
# if __name__ == "__main__":
#     # Define the image path and the labels for classification
#     image_path = "dog.jpg"
#     labels = ["a photo of a dog", "a photo of a cat", "a photo of a bird"]
#
#     # Classify the image
#     best_match_label, similarities = classify_image(image_path, labels)
#
#     # Print the results
#     print(f"Best match label: {best_match_label}")
#     print("Similarities with each label:")
#     for label, similarity in zip(labels, similarities):
#         print(f"{label}: {similarity:.4f}")

#---------------------try on 3 different photos------------------------------------
import torch
import clip
from PIL import Image

def load_clip_model(device='cpu'):
    # Load the CLIP model
    device = torch.device(device)
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device

def classify_images(image_paths, labels):
    # Load the model and preprocess
    model, preprocess, device = load_clip_model()

    results = []

    for image_path in image_paths:
        # Preprocess the image
        image = Image.open(image_path)
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Preprocess the labels (text descriptions)
        text_inputs = torch.cat([clip.tokenize(label) for label in labels]).to(device)

        # Get image and text features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # Calculate cosine similarity between image and text features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T).squeeze(0)

        # Get the best matching label
        best_match_index = similarities.argmax().item()
        best_match_label = labels[best_match_index]

        results.append((image_path, best_match_label, similarities.cpu().numpy()))

    return results

if __name__ == "__main__":
    # Define the list of image paths and the labels for classification
    image_paths = ["dog.jpg", "tit-8620213_1280.jpg", "pexels-pixabay-45201.jpg", "Cute_d.jpg"]
    labels = ["a photo of a dog", "a photo of a cat", "a photo of a bird"]

    # Classify the images
    results = classify_images(image_paths, labels)

    # Print the results
    for image_path, best_match_label, similarities in results:
        print(f"\nImage: {image_path}")
        print(f"Best match label: {best_match_label}")
        print("Similarities with each label:")
        for label, similarity in zip(labels, similarities):
            print(f"{label}: {similarity:.4f}")

