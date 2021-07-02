import torch
import torchvision
import coremltools as ct
from models import ResnetGenerator
import io
# Convert to Core ML using the Unified Conversion API

# Load a pre-trained version of MobileNetV2
f = './models/photo2cartoon_weights.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# params = torch.load(f, map_location=device)
#
# net = ResnetGenerator(ngf=32, img_size=256, light=True).to(device)
# net.load_state_dict(params['genA2B'])
# net.eval()


example_input = torch.rand(1, 3, 256, 256) # after test, will get 'size mismatch' error message with size 256x256
traced_model = torch.jit.load(f, map_location="cpu")
# traced_model = torch.jit.trace(net, example_input)


# with open('./models/photo2cartoon_weights.pt', 'rb') as f:
#     buffer = io.BytesIO(f.read())
# buffer.seek(0)
# loaded = torch.jit.load(buffer,map_location='cpu')
print(type(traced_model))



model = ct.convert(
    traced_model, source='pytorch',
    inputs = [ct.ImageType(name="input_1", shape=example_input.shape)],  # name "input_1" is used in 'quickstart'

)

model.save("MobileNetV2.mlmodel")
