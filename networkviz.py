# Combined PyTorch loading and Manim Visualization Script
# UPDATED FOR THE PROVIDED Network CLASS
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from manim import *

# --- Configuration ---

# Environment and Model Parameters (MUST MATCH YOUR SETUP)
MODEL_PATH = 'NetworkStep3M.pth'
ENV_SIZE = 8  # Grid size (e.g., 8x8)
# Define obstacles, start, and goal exactly as used for the input state you want to visualize
TEST_OBSTACLES = [(0,3),(1,3),(3,1),(2,5),(3,6),(4,6),(4,2),(4,4),(5,3),(6,2)]
TEST_AGENT_START = (0, 0)
TEST_GOAL = (ENV_SIZE - 1, ENV_SIZE - 1)

# --- Network Structure Configuration (Derived from your network.py) ---
# These parameters match the defaults in your Network class.
# If your saved model used different hidden sizes, change them here.
INPUT_SIZE = ENV_SIZE * ENV_SIZE # 64
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32
OUTPUT_SIZE = 4 # Number of actions

# !! IMPORTANT: Set this to True if your saved model was trained with Leaky ReLU !!
USE_LEAKY_RELU_FOR_LOADING = False

# --- Manim Visualization Parameters ---
# Layer structure based on your Network's forward pass:
# Input -> fc1 -> activation1 -> fc2 -> activation2 -> fc3 (logits) -> softmax
# We will capture activations AFTER fc1, fc2, fc3(logits), and final softmax output
LAYER_SIZES = {
    'input': INPUT_SIZE,    # 64
    'fc1': HIDDEN_SIZE_1,   # 64 (Output neurons of fc1, before activation)
    'fc2': HIDDEN_SIZE_2,   # 32 (Output neurons of fc2, before activation)
    'fc3': OUTPUT_SIZE,     # 4  (Output neurons of fc3, these are logits)
    'softmax': OUTPUT_SIZE  # 4  (Final output after softmax)
}
# Order for visualization layout - we show the outputs of the linear layers and the final softmax
LAYER_NAMES_ORDERED = ['input', 'fc1', 'fc2', 'fc3', 'softmax']

# Optional: Limit max nodes shown per layer for clarity if layers are huge
MAX_NODES_TO_SHOW = 30 # Set to None to show all

# Manim visual styles (remain the same)
NODE_RADIUS = 0.15
LAYER_SPACING = 2.0
NODE_SPACING = 0.3
WEIGHT_COLOR_NEGATIVE = BLUE
WEIGHT_COLOR_ZERO = GRAY
WEIGHT_COLOR_POSITIVE = RED
WEIGHT_STROKE_WIDTH_MAX = 3.0
WEIGHT_STROKE_WIDTH_MIN = 0.5
ACTIVATION_COLOR_OFF = BLACK
ACTIVATION_COLOR_ON = YELLOW
ACTIVATION_OPACITY_MAX = 1.0
ACTIVATION_OPACITY_MIN = 0.1

# --- Helper Functions ---

def load_model_and_env(model_path, env_size, obstacles, start, goal,
                       input_size, hidden_size1, hidden_size2, output_size, use_leaky_relu):
    """Loads the environment and the PyTorch model based on provided Network class."""
    try:
        import environment as env
        # Import the specific Network class provided by the user
        from network import Network # Assumes network.py contains the class
    except ImportError as e:
        print(f"Error importing environment or network module: {e}")
        print("Please ensure environment.py and network.py are in the same directory or Python path.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during import: {e}")
        sys.exit(1)

    print("Setting up environment...")
    environment = env.Environment(size=env_size, obstacles=obstacles, agent_start=start, goal=goal)
    initial_state = environment.reset(obstacles=obstacles, start=start, goal=goal)
    state_tensor = torch.tensor(initial_state, dtype=torch.float32).unsqueeze(0).view(1, -1)
    print(f"Input state tensor shape: {state_tensor.shape}")

    print(f"Loading policy network from {model_path}...")
    # Instantiate the network using the provided class definition and parameters
    try:
        # Use the parameters defined above, including the Leaky ReLU flag
        policy_network = Network(
            input_size=input_size,
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            output_size=output_size,
            use_leaky_relu=use_leaky_relu
        )
        print(f"Instantiated Network with: input={input_size}, h1={hidden_size1}, h2={hidden_size2}, output={output_size}, leaky={use_leaky_relu}")
    except Exception as e:
        print(f"Error instantiating network.Network(): {e}")
        print("Check if network.py is correct and matches the provided class definition.")
        sys.exit(1)

    # Load trained weights
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        policy_network.load_state_dict(state_dict)
        policy_network.eval()
        print("Model loaded successfully.")
        # print("\nModel Structure:")
        # print(policy_network) # Optional: print structure to confirm
        # print("-" * 30)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model state dictionary: {e}")
        print("Ensure the parameters (hidden sizes, leaky relu) used for loading match the saved model.")
        sys.exit(1)

    return policy_network, state_tensor, environment

def get_activations_and_weights(model, input_tensor):
    """Performs forward pass, captures activations via hooks, and extracts weights."""
    activation_outputs = {}
    hooks = []

    def get_activation(name):
        def hook(model_layer, input_to_layer, output_from_layer):
            # Capture the output tensor of the hooked layer
            activation_outputs[name] = output_from_layer.detach()
            # print(f"Hook '{name}' captured output shape: {output_from_layer.shape}") # Debug print
        return hook

    print("Registering forward hooks...")
    # Hook the outputs of the Linear layers (fc1, fc2, fc3)
    # The keys 'fc1', 'fc2', 'fc3' MUST match the keys in LAYER_SIZES and layer attributes in Network class
    layer_map = {
        'fc1': model.fc1 if hasattr(model, 'fc1') else None,
        'fc2': model.fc2 if hasattr(model, 'fc2') else None,
        'fc3': model.fc3 if hasattr(model, 'fc3') else None, # This will capture logits
    }

    for name, layer_module in layer_map.items():
        if layer_module is not None and isinstance(layer_module, nn.Linear): # Ensure it's a Linear layer
            print(f" - Hooking output of: {name} ({type(layer_module).__name__})")
            hooks.append(layer_module.register_forward_hook(get_activation(name)))
        elif layer_module is None:
             print(f" - Warning: Layer for '{name}' not found in model. Skipping hook.")
        else:
             print(f" - Warning: Module '{name}' is not Linear. Skipping hook.")


    # Perform forward pass
    print("Performing forward pass for the given input state...")
    with torch.no_grad():
        # The forward pass inherently applies activations and softmax
        output_probabilities = model(input_tensor)
    print("Forward pass complete.")
    # print(f"Final output probabilities shape: {output_probabilities.shape}") # Debug print

    # Store input tensor and the final softmax output
    activation_outputs['input'] = input_tensor.detach()
    activation_outputs['softmax'] = output_probabilities.detach() # Capture the final result

    # Remove hooks
    for handle in hooks:
        handle.remove()
    print("Hooks removed.")

    # Verify captured keys align with expected layers
    print("\nKeys captured by hooks/manual addition:", list(activation_outputs.keys()))

    # Extract weights (this part remains the same)
    print("Extracting model weights...")
    weights = {}
    biases = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Extract layer name like 'fc1' from 'fc1.weight'
            layer_name = name.split('.')[0]
            if layer_name in ['fc1', 'fc2', 'fc3']: # Check if it's one of the expected layers
                weights[layer_name] = param.data.numpy()
                print(f" - Extracted weights for {layer_name}: shape {weights[layer_name].shape}")
        elif 'bias' in name:
            layer_name = name.split('.')[0]
            if layer_name in ['fc1', 'fc2', 'fc3']:
                biases[layer_name] = param.data.numpy()
                print(f" - Extracted biases for {layer_name}: shape {biases[layer_name].shape}")

    # Convert captured activations to numpy arrays
    numpy_activations = {}
    print("\nCaptured Activations (Shapes after flatten):")
    for name in LAYER_NAMES_ORDERED: # Iterate in the desired order
        if name in activation_outputs:
            act_tensor = activation_outputs[name]
            numpy_activations[name] = act_tensor.numpy().flatten()
            print(f" - {name}: {numpy_activations[name].shape}")
        else:
            print(f" - Warning: Activation for layer '{name}' not found in captured outputs!")
            # Handle missing activations if necessary (e.g., assign zeros or skip)
            # For visualization, it's better if all expected layers are present
            # numpy_activations[name] = np.zeros(LAYER_SIZES.get(name, 1)) # Placeholder

    return numpy_activations, weights, biases

def normalize_for_visual(data, min_val=None, max_val=None):
    """Normalizes data (weights or activations) to [0, 1] range for visualization."""
    # This function remains the same as before
    if isinstance(data, dict): # Handle weights dict
        all_vals = np.concatenate([arr.flatten() for arr in data.values()])
    else: # Handle activation array
        all_vals = data.flatten()

    if len(all_vals) == 0: # Handle case with no data
         return data, 0, 1 if isinstance(data, dict) else (np.array([]), 0, 1)


    if min_val is None:
        min_val = np.min(all_vals)
    if max_val is None:
        max_val = np.max(all_vals)

    range_val = max_val - min_val
    # Prevent division by zero if all values are the same
    if range_val == 0:
        # Return 0.5 for constant values (neutral gray/opacity)
        norm_val = 0.5
        if isinstance(data, dict):
            return {k: np.full_like(v, norm_val) for k, v in data.items()}, min_val, max_val
        else:
            return np.full_like(data, norm_val), min_val, max_val

    if isinstance(data, dict):
        normalized_data = {}
        for key, arr in data.items():
            normalized_data[key] = (arr - min_val) / range_val
        return normalized_data, min_val, max_val
    else:
        normalized_data = (data - min_val) / range_val
        return normalized_data, min_val, max_val


# --- Manim Scene ---

class NeuralNetworkViz(Scene):
    def construct(self):
        # --- 1. Load Data ---
        model, input_tensor, environment = load_model_and_env(
            MODEL_PATH, ENV_SIZE, TEST_OBSTACLES, TEST_AGENT_START, TEST_GOAL,
            INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE, USE_LEAKY_RELU_FOR_LOADING
        )
        activations_np, weights_np, biases_np = get_activations_and_weights(model, input_tensor)

        # --- 2. Prepare Data for Visualization ---
        # Normalize weights (using absolute max for symmetric color mapping)
        all_weights_flat = np.concatenate([w.flatten() for w in weights_np.values()]) if weights_np else np.array([])
        max_abs_weight = np.max(np.abs(all_weights_flat)) if len(all_weights_flat) > 0 else 1.0
        if max_abs_weight == 0: max_abs_weight = 1.0 # Avoid division by zero

        # Normalize activations (globally for simplicity)
        all_activations_flat = np.concatenate([a.flatten() for a in activations_np.values()]) if activations_np else np.array([])
        min_act, max_act = (np.min(all_activations_flat), np.max(all_activations_flat)) if len(all_activations_flat) > 0 else (0, 1)
        if max_act == min_act: max_act += 1e-6 # Avoid division by zero

        # --- 3. Create Network Topology in Manim ---
        manim_layers = self.create_manim_network_topology(LAYER_SIZES, LAYER_NAMES_ORDERED, MAX_NODES_TO_SHOW)
        self.add(manim_layers)
        self.wait(0.5)

        # --- 4. Create Connections (Weights) ---
        connections = VGroup()
        # Keys of weights_np should be 'fc1', 'fc2', 'fc3'
        weight_keys_available = list(weights_np.keys())

        # Iterate through the gaps between visualized layers
        for i in range(len(manim_layers) - 1):
            prev_manim_layer_nodes = manim_layers[i]
            curr_manim_layer_nodes = manim_layers[i+1]

            prev_layer_name = LAYER_NAMES_ORDERED[i]     # e.g., 'input', 'fc1', 'fc2'
            curr_layer_name = LAYER_NAMES_ORDERED[i+1]   # e.g., 'fc1', 'fc2', 'fc3', 'softmax'

            # Determine which weight matrix connects these layers
            # Weights connect the output of the previous layer to the input of the current *linear* layer.
            # The weight key corresponds to the name of the *destination* linear layer being visualized.
            current_weight_key = None
            if curr_layer_name in weight_keys_available: # e.g., if curr_layer_name is 'fc1', 'fc2', or 'fc3'
                current_weight_key = curr_layer_name

            if current_weight_key:
                weight_matrix = weights_np[current_weight_key] # Shape (out_features, in_features)

                num_prev_nodes_actual = LAYER_SIZES[prev_layer_name] # Get actual size from config
                num_curr_nodes_actual = LAYER_SIZES[curr_layer_name]
                num_prev_nodes_shown = len(prev_manim_layer_nodes)
                num_curr_nodes_shown = len(curr_manim_layer_nodes)

                # Sanity check shape: weight matrix shape[1] should match prev_layer_size
                # weight matrix shape[0] should match curr_layer_size (the linear layer itself)
                if weight_matrix.shape[1] != num_prev_nodes_actual or weight_matrix.shape[0] != num_curr_nodes_actual:
                     print(f"!! WARNING: Weight shape mismatch for '{current_weight_key}' !!")
                     print(f"   Weight shape: {weight_matrix.shape}")
                     print(f"   Expected based on LAYER_SIZES: ({num_curr_nodes_actual}, {num_prev_nodes_actual})")
                     print(f"   Skipping connections for this layer.")
                     continue # Skip drawing connections for this layer if shapes mismatch

                print(f"Connecting layer {i} ({prev_layer_name}, {num_prev_nodes_shown} shown) to layer {i+1} ({curr_layer_name}, {num_curr_nodes_shown} shown) using weights '{current_weight_key}' ({weight_matrix.shape})")

                # --- (Connection drawing loop - remains the same as before) ---
                for j in range(num_prev_nodes_shown):
                    for k in range(num_curr_nodes_shown):
                        prev_node_idx_orig = j * (num_prev_nodes_actual // num_prev_nodes_shown) if num_prev_nodes_shown < num_prev_nodes_actual and num_prev_nodes_shown > 0 else j
                        curr_node_idx_orig = k * (num_curr_nodes_actual // num_curr_nodes_shown) if num_curr_nodes_shown < num_curr_nodes_actual and num_curr_nodes_shown > 0 else k

                        if prev_node_idx_orig < weight_matrix.shape[1] and curr_node_idx_orig < weight_matrix.shape[0]:
                            weight_val = weight_matrix[curr_node_idx_orig, prev_node_idx_orig]
                            norm_weight_color = np.clip((weight_val / max_abs_weight + 1) / 2, 0, 1)
                            color = interpolate_color(WEIGHT_COLOR_NEGATIVE, WEIGHT_COLOR_POSITIVE, norm_weight_color)
                            if abs(weight_val) < max_abs_weight * 0.05: color = WEIGHT_COLOR_ZERO
                            norm_weight_width = np.clip(abs(weight_val) / max_abs_weight, 0, 1)
                            stroke_width = interpolate(WEIGHT_STROKE_WIDTH_MIN, WEIGHT_STROKE_WIDTH_MAX, norm_weight_width)
                            opacity = 0.1 + 0.7 * norm_weight_width

                            line = Line(
                                prev_manim_layer_nodes[j].get_center(),
                                curr_manim_layer_nodes[k].get_center(),
                                stroke_color=color,
                                stroke_width=stroke_width,
                                stroke_opacity=opacity,
                                z_index=-1 # Send lines behind nodes
                            )
                            connections.add(line)
                        # else: (Removed warning for brevity, truncation should handle this)
                           # print(f"Warning: Index out of bounds skipped...")
                # --- (End connection drawing loop) ---
            else:
                print(f" - Skipping connections between Manim layer {i} ({prev_layer_name}) and {i+1} ({curr_layer_name}) (No weights connect these, e.g., fc3 to softmax)")


        # Add connections behind nodes
        manim_layers.move_to(ORIGIN) # Center network
        connections.move_to(ORIGIN)
        self.play(FadeIn(connections), run_time=1.5)
        # self.bring_to_front(manim_layers) # Ensure nodes are on top (can also use z_index)
        self.wait(0.5)


        # --- 5. Animate Activations ---
        activation_anims = []
        print("\nApplying activations to nodes:")
        for i, layer_name in enumerate(LAYER_NAMES_ORDERED):
            if layer_name in activations_np and i < len(manim_layers): # Check if layer exists in topology
                manim_layer_nodes = manim_layers[i]
                node_activations = activations_np[layer_name]

                num_nodes_actual = len(node_activations)
                num_nodes_shown = len(manim_layer_nodes)
                print(f" - Layer '{layer_name}': {num_nodes_actual} activations, {num_nodes_shown} nodes shown.")

                layer_anims = []
                for j, node in enumerate(manim_layer_nodes):
                    node_idx_orig = j * (num_nodes_actual // num_nodes_shown) if num_nodes_shown < num_nodes_actual and num_nodes_shown > 0 else j

                    if node_idx_orig < num_nodes_actual:
                        activation_val = node_activations[node_idx_orig]
                        norm_act = np.clip((activation_val - min_act) / (max_act - min_act), 0, 1) # Normalize and clip
                        fill_color = interpolate_color(ACTIVATION_COLOR_OFF, ACTIVATION_COLOR_ON, norm_act)
                        fill_opacity = interpolate(ACTIVATION_OPACITY_MIN, ACTIVATION_OPACITY_MAX, norm_act)
                        layer_anims.append(node.animate.set_style(fill_color=fill_color, fill_opacity=fill_opacity))
                    # else: (Removed warning)
                        # print(f"Warning: Activation index out of bounds...")

                if layer_anims:
                    # Group animations for the layer and add to the sequence
                    activation_anims.append(AnimationGroup(*layer_anims, lag_ratio=0.0)) # Animate nodes in layer simultaneously

            elif layer_name not in activations_np:
                 print(f" - Skipping animation for layer '{layer_name}' (activations not found).")


        # Play activation animations sequentially layer by layer
        if activation_anims:
            self.play(LaggedStart(*activation_anims, lag_ratio=0.5), run_time=max(1.0, 0.7 * len(activation_anims)))

        self.wait(5)


    def create_manim_network_topology(self, layer_sizes_dict, layer_names_ordered, max_nodes_to_show=None):
        """Creates the VGroups representing network layers in Manim."""
        # This function remains the same as before
        layers_vgroup = VGroup()
        max_layer_height = 0

        print("Creating Manim network topology...")
        node_lists_for_layers = [] # Store lists of nodes for adding ellipsis later

        for name in layer_names_ordered:
            size = layer_sizes_dict.get(name, 0)
            if size == 0:
                print(f"Warning: Size for layer '{name}' not found in LAYER_SIZES dict. Skipping.")
                continue

            num_nodes_to_draw = size
            is_truncated = False
            if max_nodes_to_show is not None and size > max_nodes_to_show:
                num_nodes_to_draw = max_nodes_to_show
                is_truncated = True
                print(f" - Layer '{name}': Truncating {size} nodes to {num_nodes_to_draw} for display.")
            else:
                 print(f" - Layer '{name}': {size} nodes.")

            nodes_list = [
                Circle(radius=NODE_RADIUS, stroke_color=WHITE, stroke_width=1.5, fill_opacity=ACTIVATION_OPACITY_MIN, fill_color=ACTIVATION_COLOR_OFF)
                for _ in range(num_nodes_to_draw)
            ]
            node_lists_for_layers.append(nodes_list) # Store the list

            layer_vgroup = VGroup(*nodes_list) # Create VGroup for this layer
            layer_vgroup.arrange(DOWN, buff=NODE_SPACING)
            layers_vgroup.add(layer_vgroup) # Add the VGroup to the main layers VGroup
            max_layer_height = max(max_layer_height, layer_vgroup.height)


        # Arrange layers horizontally first
        layers_vgroup.arrange(RIGHT, buff=LAYER_SPACING)

        # Add ellipsis and labels AFTER arranging layers
        for i, name in enumerate(layer_names_ordered):
             if i < len(layers_vgroup): # Check if layer exists
                 layer_vgroup = layers_vgroup[i]
                 nodes_list = node_lists_for_layers[i]
                 size = layer_sizes_dict.get(name, 0)
                 is_truncated = max_nodes_to_show is not None and size > max_nodes_to_show

                 # Add ellipsis dots if truncated
                 if is_truncated and len(nodes_list)>0:
                     ellipsis_top = Dot(nodes_list[0].get_center() + UP * NODE_SPACING * 0.6, radius=DEFAULT_DOT_RADIUS*0.4, color=GRAY)
                     ellipsis_bottom = Dot(nodes_list[-1].get_center() + DOWN * NODE_SPACING * 0.6, radius=DEFAULT_DOT_RADIUS*0.4, color=GRAY)
                     # Add dots relative to the layer group's final position
                     layer_vgroup.add(ellipsis_top, ellipsis_bottom)

                 # Add layer labels
                 label = Text(name, font_size=18).next_to(layer_vgroup, UP, buff=0.35)
                 layer_vgroup.add(label) # Add label to the layer VGroup

        print("Topology created.")
        return layers_vgroup


# --- Main Execution ---
if __name__ == "__main__":
    print("="*50)
    print("Manim Script Execution Info (Updated for specific Network)")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Environment Size: {ENV_SIZE}x{ENV_SIZE}")
    print(f"Network Params Used for Loading: H1={HIDDEN_SIZE_1}, H2={HIDDEN_SIZE_2}, Leaky={USE_LEAKY_RELU_FOR_LOADING}")
    print("-> If Leaky ReLU setting is wrong, model loading might fail or give unexpected results.")
    print("\nTo render the visualization, run from your terminal:")
    print(f"manim -pql {os.path.basename(__file__)} NeuralNetworkViz")
    print("\nEnsure environment.py and network.py (with the provided Network class) are present.")
    print("="*50)
    pass