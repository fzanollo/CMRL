import json
import os
from collections import deque

import onnx
import torch

from DQNAgent import DQNAgent


def load_model(model_path, env):
    agent_params = {}
    try:
        # get the agent params from the saved model path if they exist
        params_file = model_path.replace('.pth', '_params.json')
        with open(params_file, 'r') as f:
            agent_params = json.load(f)
    except FileNotFoundError:
        print(f"Warning: No parameters file found for the model at {model_path}. Using default parameters.")

    agent = DQNAgent(env, **agent_params)

    obs, _ = env.reset()
    sequence_length = agent_params.get('sequence_length', 10)
    agent.observation_history = deque([obs] * sequence_length, maxlen=sequence_length)

    # Load the model state dictionary
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        agent.policy_net.load_state_dict(state_dict)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        raise

    # Sync target network
    agent.target_net.load_state_dict(agent.policy_net.state_dict())

    return agent


def add_metadata_to_model(onnx_model, name, value):
    meta = onnx_model.metadata_props.add()
    meta.key = name
    meta.value = value


def save_agent(agent, model_path, agent_params, instance_name, episodes, env):
    # Save the trained agent
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(agent.policy_net.state_dict(), f"{model_path}.pth")

    # Save the agent parameters as a JSON file
    agent_params_path = f"{model_path}_params.json"
    with open(agent_params_path, 'w') as f:
        json.dump(agent_params, f, indent=4)

    # Save as onnx model
    model_device = next(agent.policy_net.parameters()).device
    obs, _ = env.reset()
    dummy_input = torch.as_tensor(obs, dtype=torch.float32, device=model_device).unsqueeze(0)
    onnx_path = f"{model_path}.onnx"
    was_training = agent.policy_net.training
    agent.policy_net.eval()
    try:
        torch.onnx.export(
            agent.policy_net,
            (dummy_input,),
            onnx_path,
            opset_version=11,
            input_names=["input"],  # Ensure this matches the expected input name
            output_names=["output"]  # Specify output names
        )
    finally:
        if was_training:
            agent.policy_net.train()

    # add instance name to the metadata
    onnx_model = onnx.load(onnx_path)
    add_metadata_to_model(onnx_model, 'instance_name', instance_name)
    add_metadata_to_model(onnx_model, 'episodes', str(episodes))
    # Save the ONNX model with metadata
    onnx.save(onnx_model, onnx_path)
