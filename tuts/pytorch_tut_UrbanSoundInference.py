import torch
import torchaudio
import os

from pytorch_tut_UrbanSound import UrbanSoundDataset
from pytorch_tut_nn import CNNNetwork
from pytorch_tut_MNIST import FeedForwardNet, download_mnist_datasets


ANNOTATIONS_FILE = os.environ.get("ANNOTATIONS_FILE")
AUDIO_DIR = os.environ.get("AUDIO_DIR")
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

class_mapping = [
	"air_conditioner",
	"car horn",
	"children_playing",
	"dog_bark",
	"drilling",
	"engine_idling",
	"gun_shot",
	"jackhammer",
	"siren",
	"street_music"
]


def predict(model, input, target, class_mapping):
	model.eval()
	with torch.no_grad():
		predictions = model(input)
		# Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
		predicted_index = predictions[0].argmax(0)
		predicted = class_mapping[predicted_index]
		expected = class_mapping[target]
	return predicted, expected


if __name__ == '__main__':

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
	# # load back the model
	# feed_forward_net = FeedForwardNet()
	# state_dict = torch.load("feedforwardnet.pth")
	# feed_forward_net.load_state_dict(state_dict)

	# get a sample from the urbansound validation dataset for inference
	val_input, target = usd[0][0], usd[0][1] # [batch_size, num_channels, fr, time]
	input_unsqueeze_(0)

	# make an inference
	predicted, expected = predict(cnn, val_input, target, class_mapping)
	print(f"Predicted: '{predicted}', expected: '{expected}'")
