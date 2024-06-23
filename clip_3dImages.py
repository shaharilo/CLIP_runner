# import os
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import trimesh
# import pyrender
# import numpy as np
# import imageio
# import torch
# import clip
# from PIL import Image
# import matplotlib.pyplot as plt
#
#
# def load_clip_model(device='cpu'):
#     device = torch.device(device)
#     model, preprocess = clip.load("ViT-B/32", device=device)
#     model.eval()
#     return model, preprocess, device
#
#
# def render_3d_model(obj_path):
#     # Load the 3D model
#     mesh = trimesh.load(obj_path)
#
#     # Create a scene
#     scene = pyrender.Scene()
#
#     # Define camera poses for front, back, left, and right views
#     camera_poses = {
#         'front': np.array([[1, 0, 0, 0],
#                            [0, 1, 0, 0],
#                            [0, 0, 1, 2.5],  # Adjust Z translation to position closer to the front
#                            [0, 0, 0, 1]]),
#         'back': np.array([[1, 0, 0, 0],
#                           [0, 1, 0, 0],
#                           [0, 0, -1, -2],
#                           [0, 0, 0, 1]]),
#         'left': np.array([[0, 0, -1, -2],
#                           [0, 1, 0, 0],
#                           [1, 0, 0, 0],
#                           [0, 0, 0, 1]]),
#         'right': np.array([[0, 0, 1, 2],
#                            [0, 1, 0, 0],
#                            [-1, 0, 0, 0],
#                            [0, 0, 0, 1]])
#     }
#
#     # Rendering images from different angles
#     images = {}
#     for view, pose in camera_poses.items():
#         # Create a new scene for each view to reset nodes
#         scene = pyrender.Scene()
#
#         # Add mesh node to the scene
#         mesh_node = pyrender.Mesh.from_trimesh(mesh)
#         scene.add(mesh_node)
#
#         # Add camera node to the scene
#         camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
#         camera_node = pyrender.Node(camera=camera, matrix=pose)
#         scene.add_node(camera_node)
#
#         # Add directional light node to the scene
#         light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
#         light_pose = np.eye(4)
#         light_pose[:3, 3] = np.array([0, 2, 2])
#         light_node = pyrender.Node(light=light, matrix=light_pose)
#         scene.add_node(light_node)
#
#         # Render scene
#         renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
#         color, _ = renderer.render(scene)
#
#         images[view] = color
#
#         # Save the image for debugging purposes
#         imageio.imwrite(f"{view}.png", color)
#
#     return images
#
#
# def classify_images_with_clip(images, labels):
#     model, preprocess, device = load_clip_model()
#
#     results = []
#     for view, image_array in images.items():
#         image = Image.fromarray(image_array)
#         image_input = preprocess(image).unsqueeze(0).to(device)
#
#         text_inputs = torch.cat([clip.tokenize(label) for label in labels]).to(device)
#
#         with torch.no_grad():
#             image_features = model.encode_image(image_input)
#             text_features = model.encode_text(text_inputs)
#
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
#         similarities = (image_features @ text_features.T).squeeze(0)
#
#         best_match_index = similarities.argmax().item()
#         best_match_label = labels[best_match_index]
#
#         results.append((view, best_match_label, similarities.cpu().numpy()))
#
#     return results
#
#
# if __name__ == "__main__":
#     obj_path = "T89.obj"
#     labels = ["a photo of a dog", "a photo of a cat", "a photo of a woman"]
#
#     # Render 3D model to get images
#     images = render_3d_model(obj_path)
#
#     # Display the rendered images
#     fig, axes = plt.subplots(1, 4, figsize=(15, 5))
#     for i, (view, image_array) in enumerate(images.items()):
#         axes[i].imshow(image_array)
#         axes[i].set_title(view)
#         axes[i].axis('off')
#     plt.tight_layout()
#     plt.show()
#
#     # Classify images using CLIP
#     results = classify_images_with_clip(images, labels)
#
#     # Display classification results
#     for view, best_match_label, similarities in results:
#         print(f"\nView: {view}")
#         print(f"Best match label: {best_match_label}")
#         print("Similarities with each label:")
#         for label, similarity in zip(labels, similarities):
#             print(f"{label}: {similarity:.4f}")

#---------------------another try - the first one worked for 4 angles - now 20 -----------------
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import trimesh
import pyrender
import numpy as np
import imageio
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def load_clip_model(device='cpu'):
    device = torch.device(device)
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device


def render_3d_model(obj_path):
    # Load the 3D model using trimesh
    mesh = trimesh.load(obj_path)

    # Calculate object center
    object_center = mesh.bounds.mean(axis=0)

    # Create a pyrender scene
    scene = pyrender.Scene()

    # Define a fixed camera pose
    fixed_camera_pose = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 3.0],  # Camera positioned in front
                                  [0, 0, 0, 1]])

    # Define rotations around x and y axes for different views
    num_views = 20
    rotations = []
    for i in range(num_views):
        angle_x = (i // 4) * np.pi / 2  # Rotate around x-axis every 4 views
        angle_y = (i % 4) * np.pi / 2  # Rotate around y-axis for each group of 4 views
        rotation_x = Rotation.from_euler('x', angle_x).as_matrix()
        rotation_y = Rotation.from_euler('y', angle_y).as_matrix()
        rotations.append(rotation_x @ rotation_y)

    # Rendering images from different angles
    images = {}
    for i, rotation in enumerate(rotations):
        # Apply rotation to the mesh
        rotated_mesh = mesh.copy()
        rotated_mesh.apply_transform(np.eye(4))  # Reset to identity matrix
        rotated_mesh.apply_transform(np.linalg.inv(trimesh.transformations.translation_matrix(object_center)))
        rotated_mesh.apply_transform(np.eye(4))  # Reset to identity matrix
        rotated_mesh.apply_transform(np.vstack((np.hstack((rotation, [[0], [0], [0]])), [0, 0, 0, 1])))
        rotated_mesh.apply_transform(trimesh.transformations.translation_matrix(object_center))

        # Create a new scene for each view to reset nodes
        scene = pyrender.Scene()

        # Add rotated mesh node to the scene
        mesh_node = pyrender.Mesh.from_trimesh(rotated_mesh)
        scene.add(mesh_node)

        # Add camera node to the scene
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=fixed_camera_pose)

        # Add directional light node to the scene
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, 2, 2])
        scene.add(light, pose=light_pose)

        # Render scene
        renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
        color, _ = renderer.render(scene)

        view = f'view_{i + 1}'
        images[view] = color

        # Save the image for debugging purposes
        imageio.imwrite(f"{view}.png", color)

    return images


def classify_images_with_clip(images, labels):
    model, preprocess, device = load_clip_model()

    results = []
    for view, image_array in images.items():
        image = Image.fromarray(image_array)
        image_input = preprocess(image).unsqueeze(0).to(device)

        text_inputs = torch.cat([clip.tokenize(label) for label in labels]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T).squeeze(0)

        best_match_index = similarities.argmax().item()
        best_match_label = labels[best_match_index]

        results.append((view, best_match_label, similarities.cpu().numpy()))

    return results


if __name__ == "__main__":
    obj_path = "T89.obj"
    labels = ["a photo of a dog", "a photo of a cat", "a photo of a woman"]

    # Render 3D model to get images
    images = render_3d_model(obj_path)

    # Display the rendered images
    num_views = len(images)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    for i, (view, image_array) in enumerate(images.items()):
        ax = axes[i // 5, i % 5]
        ax.imshow(image_array)
        ax.set_title(view)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Classify images using CLIP
    results = classify_images_with_clip(images, labels)

    # Display classification results
    for view, best_match_label, similarities in results:
        print(f"\nView: {view}")
        print(f"Best match label: {best_match_label}")
        print("Similarities with each label:")
        for label, similarity in zip(labels, similarities):
            print(f"{label}: {similarity:.4f}")

