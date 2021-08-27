import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from midi_replay_dataset import MidiReplayDataset
from conv_net import ConvNet

BATCH_SIZE = 128
EPOCHS = 2
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = r"C:\Users\Christian\Documents\cbhower\cbhower\urban-sound-dataset-classifier-torch\UrbanSound8K\metadata\UrbanSound8K.csv"
AUDIO_DIR = r"C:\Users\Christian\Documents\cbhower\cbhower\urban-sound-dataset-classifier-torch\UrbanSound8K\audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")

def test(model, data_loader, loss_fn):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for input_data, target in data_loader:
            input_data, target = input_data.to(device), target.to(device)
            prediction = model(input_data)
            test_loss += loss_fn(prediction, target).item()
            correct += (prediction.argmax(1) == target).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    mrd = MidiReplayDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    train_dataloader = create_data_loader(mrd, BATCH_SIZE)
    test_dataloader = create_data_loader(mrd, BATCH_SIZE)
    # construct model and assign it to device
    cnn = ConvNet().to(device)
    print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)
    test(cnn, train_dataloader, loss_fn)

    # save model
    torch.save(cnn.state_dict(), "feedforwardnet.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")