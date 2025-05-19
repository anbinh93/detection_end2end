import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from pathlib import Path

DEFAULT_DETR_MODEL_NAME = "facebook/detr-resnet-101"

def load_detr_from_local_pth(model_path: Path, device: torch.device):
    """
    Loads a DETR model from a local .pth file and its processor.

    Args:
        model_path (Path): Path to the .pth model file.
        device (torch.device): The device to load the model onto (e.g., 'cpu', 'cuda').

    Returns:
        tuple: (model, processor, class_names) or (None, None, None) if loading fails.
    """
    if not model_path.exists():
        print(f"DETR model .pth file not found at {model_path}")
        return None, None, None

    try:
        print(f"Attempting to load DETR model from local path: {model_path}...")

        # Load the processor using the default model name
        # The processor is not typically part of the .pth file
        processor = DetrImageProcessor.from_pretrained(DEFAULT_DETR_MODEL_NAME)

        # Initialize the model architecture from the default pre-trained model name
        # This ensures the architecture matches the weights we are about to load.
        # We load the state_dict into this architecture.
        model = DetrForObjectDetection.from_pretrained(DEFAULT_DETR_MODEL_NAME, ignore_mismatched_sizes=True)
        
        # Load the state dictionary from the .pth file
        state_dict = torch.load(model_path, map_location=device)
        
        # If the .pth file contains the full model, it might be under a 'model' key
        # or it might be the state_dict directly. Common for HuggingFace or Timm.
        # Check if 'model' key exists (common in checkpoints saved during training)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        # Some checkpoints might have keys prefixed with 'model.'
        # We need to strip this prefix if the model architecture doesn't expect it.
        # Example: if keys are 'model.conv1.weight' but model expects 'conv1.weight'
        # This is a common adjustment.
        
        # Create a new state_dict without the 'model.' prefix if present
        new_state_dict = {}
        has_model_prefix = any(key.startswith("model.") for key in state_dict.keys())

        if has_model_prefix: # Heuristic: if any key starts with "model.", assume all relevant keys do.
            for key, value in state_dict.items():
                if key.startswith("model."):
                    new_state_dict[key[len("model."):]] = value
                else: # Keep other keys if any (e.g. EMA weights, optimizer state - though we only need model)
                    new_state_dict[key] = value # This might not be needed if it's just model weights
        else:
            new_state_dict = state_dict

        # It's also possible the .pth file is from a different DETR variant or was saved differently.
        # Sometimes, the issue is with the final classification head if number of classes differ.
        # `load_state_dict` has a `strict` parameter. If False, it allows partial loads.
        # For DETR, the classfier layer is `class_labels_classifier`
        
        # Before loading, let's check if the number of classes in the checkpoint matches the model config.
        # The classifier weights are typically:
        # 'class_labels_classifier.weight' and 'class_labels_classifier.bias'
        num_classes_in_config = model.config.num_labels
        
        # Check for classifier weights in the state_dict
        classifier_weight_key = None
        for key in new_state_dict.keys():
            if 'class_labels_classifier.weight' in key: # could be nested e.g. model.class_labels_classifier.weight
                 classifier_weight_key = key
                 break
        
        if classifier_weight_key and new_state_dict[classifier_weight_key].shape[0] != num_classes_in_config:
            print(f"Warning: Number of classes in checkpoint ({new_state_dict[classifier_weight_key].shape[0]}) "
                  f"differs from model config ({num_classes_in_config}).")
            print("Attempting to load weights, excluding the incompatible classifier head.")
            # Remove problematic classifier layers from state_dict
            keys_to_remove = [k for k in new_state_dict if 'class_labels_classifier' in k]
            for k in keys_to_remove:
                del new_state_dict[k]
            
            model.load_state_dict(new_state_dict, strict=False)
            print("Loaded DETR weights from .pth with mismatched classifier (classifier excluded). "
                  "The model will use the classifier head from the pre-trained config.")
        else:
            model.load_state_dict(new_state_dict, strict=True)
            print("Successfully loaded DETR state_dict from .pth file.")

        model.to(device)
        model.eval()

        class_names = list(model.config.id2label.values())

        print(f"DETR model from {model_path.name} loaded on {device} with {len(class_names)} classes.")
        return model, processor, class_names

    except Exception as e:
        print(f"Error loading DETR model from local .pth file {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

