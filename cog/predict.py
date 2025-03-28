from cog import BasePredictor, Input, Path, BaseModel
from starvector.model.starvector_arch import StarVectorForCausalLM
from starvector.data.util import process_and_rasterize_svg
import torch

class Prediction(BaseModel):
    svg: str
    raster_svg: Path

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = StarVectorForCausalLM.from_pretrained("starvector/starvector-8b-im2svg")

    def predict(self, image: Path = Input(description="Image to vectorize")) -> Prediction:
        """Run a single prediction on the model"""
        image = self.model.process_images([image])[0].cuda()
        batch = {"image": image}
        raw_svg = self.model.generate_im2svg(batch, max_length=4096)[0]
        svg, raster_image = process_and_rasterize_svg(raw_svg)
        return Prediction(svg=svg, raster_svg=Path(raster_image))
