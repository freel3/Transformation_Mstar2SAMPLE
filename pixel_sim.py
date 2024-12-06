'''
    Evaluate the similarity between two images by calculating their simple differential encoding
'''
import torch

def image_encode(image, threshold=0.1):
    encoded_image = torch.zeros_like(image)

    diff = image[:, 1:] - image[:, :-1]
    encoded_image[:, 1:] = (diff > threshold).float()

    return encoded_image


def similar(fake, real):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fake = torch.as_tensor(fake, dtype=torch.float32, device=device)
    real = torch.as_tensor(real, dtype=torch.float32, device=device)

    fake1 = torch.flip(fake, dims=[1])
    encoded_image1 = image_encode(fake1)
    encoded_image2 = image_encode(real)

    N = torch.sum(encoded_image1 * encoded_image2) + torch.sum((1 - encoded_image1) * (1 - encoded_image2))
    riic = N / (fake.shape[0] * fake.shape[1])

    return riic.item()
