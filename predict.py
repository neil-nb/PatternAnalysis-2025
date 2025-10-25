import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print("Hello, World!")