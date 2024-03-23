# Tutorial video link here: https://www.youtube.com/watch?v=4p0G6tgNLis&list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm&index=2

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio

from pytorch_tut_UrbanSound import UrbanSoundDataset
from pytorch_tut_nn import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

def create_data_loader(train_data, batch_size):
	train_data_loader = DataLoader(train_data, batch_size=batch_size)
	return train_data_loader

def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
	for inputs, targets in data_loader:
		inputs, targers = inputs.to(device), targets.to(device)
		# calculate loss
		predictions = model(inputs)
		loss = loss_fn(predictions, targets)
		# backpropagate loss and update weights
		optimiser.zero_grad()
		loss.backward()
		optimiser.step()
	print(f"Loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimiser, device, epochs):
	for i in range(epochs):
		print(f"Epoch {i+1}")
		train_one_epoch(model, data_loader, loss_fn, optimiser, device)
		print("-------------------------")
	print("Training is done!")

if __name__ == '__main__':

	# instatiating dataset
	ANNOTATIONS_FILE = 'C:\\Users\\MIKAL\\Music\\UrbanSound8K\\metadata\\UrbanSound8K.csv'
	AUDIO_DIR = 'C:\\Users\\MIKAL\\Music\\UrbanSound8K\\audio'
	SAMPLE_RATE = 22050
	NUM_SAMPLES = 22050
	OUT_FILE = "feedforwardnet.pth"

	if torch.cuda.is_available():
		device = 'cuda'
	else:
		device = 'cpu'
	print(f"Using device: {device}")

	mel_spectrogram = torchaudio.transforms.MelSpectrogram(
		sample_rate=SAMPLE_RATE,
		n_fft=1024,
		hop_length=512,
		n_mels=64
	)

	usd = UrbanSoundDataset(ANNOTATIONS_FILE,
							AUDIO_DIR,
							mel_spectrogram,
							SAMPLE_RATE,
							NUM_SAMPLES,
							device)
	print(f"There are {len(usd)} samples in the dataset.")
	print("END")


	# create a data loader for train set
	train_data_loader = create_data_loader(usd, BATCH_SIZE)

	# # build model
	cnn = CNNNetwork().to(device)
	print(cnn)

	# instantiate loss function & optimiser
	loss_fn = nn.CrossEntropyLoss()
	optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

	# train model
	train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)

	# store model
	torch.save(cnn.state_dict(), OUT_FILE)
	print(f"Model trained and stored at {OUT_FILE}")
