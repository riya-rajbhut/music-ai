import os
import importlib.util


def load_main_module():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.join(current_dir, "main.py")
    spec = importlib.util.spec_from_file_location("main", main_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_network():
    module = load_main_module()
    if hasattr(module, "NeuralNetwork"):
        return module.NeuralNetwork()
    if hasattr(module, "build_model"):
        return module.build_model()
    if hasattr(module, "build_network"):
        return module.build_network()
    raise AttributeError(
        "main.py must define a NeuralNetwork class or a build_model/build_network function"
    )


def main():
    network = create_network()
    print(f"Loaded neural network from main.py: {type(network).__name__}")
    if hasattr(network, "summary"):
        network.summary()
    if hasattr(network, "predict"):
        sample_input = [0.0] * getattr(network, "input_size", 1)
        try:
            prediction = network.predict(sample_input)
            print("Sample prediction:", prediction)
        except Exception:
            pass


if __name__ == "__main__":
    main()
