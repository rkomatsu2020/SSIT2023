from test_Guided_Trans import Test

from SSIT.Model.net import Generator
from SSIT.config import model_setting

class TestSSIT(Test):
    def __init__(self, gpu_no, dataset_name='animal2animal', **kargs):
        super().__init__(gpu_no, dataset_name=dataset_name, model_config=model_setting)
        # Set Generator
        img_size_config = 'default' if dataset_name not in model_setting['img_size'] else dataset_name
        img_size = model_setting['img_size'][img_size_config]
        self.netG = Generator(img_size=img_size, input_ch=3).to(self.device)
