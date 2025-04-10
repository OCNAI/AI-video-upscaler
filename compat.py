import sys
import types
from torchvision.transforms.functional import rgb_to_grayscale

# Create a fake module called 'torchvision.transforms.functional_tensor'
functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
functional_tensor.rgb_to_grayscale = rgb_to_grayscale

# Inject it into sys.modules so imports elsewhere don't crash
sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor
