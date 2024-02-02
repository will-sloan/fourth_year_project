# Runs the transformer from simple_transformer


# Make sure we can import on the path we are on
import sys
sys.path.append('/workspace/fourth_year_project/MusicGen')

from MyAudioDataset import MyAudioDataset
from AudioCodesDataset import AudioCodesDataset
from simple_transformer import AudioTransformer
from audiocraft.models import CompressionModel
from audiocraft.models.encodec import InterleaveStereoCompressionModel



# Compression model, shortens 10secs to 8,500
model = CompressionModel.get_pretrained('facebook/encodec_32khz')
comp_model = InterleaveStereoCompressionModel(model).cuda()

#mydataset = MyAudioDataset('/workspace/small_model_data3', 'recording_01_')
audio_codes_dataset = AudioCodesDataset(comp_model)
audio_codes_dataset.load_data('90_degree_compress_tensors_10sec_augmented.pkl')
#audio_codes_dataset.set_audio_dataset(mydataset)

assert len(audio_codes_dataset) == 5130, "Dataset is not the right size"

# Create transformer
myTransformer = AudioTransformer(comp_model=comp_model, d_model=500, nhead=4, num_layers=3, dim_feedforward=500).cuda()
myTransformer.train()

# Runs our training function
myTransformer.train_loop(dataset=audio_codes_dataset, batch_size=4, epochs=151, lr=0.001)