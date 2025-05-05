from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class BrainLMConfig(PretrainedConfig):

    model_type = "brainlm_mae"

    def __init__(
        self,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        qkv_bias=True,
        decoder_num_attention_heads=4,
        decoder_hidden_size=256,
        decoder_num_hidden_layers=2,
        decoder_intermediate_size=512,
        num_brain_voxels=1,
        num_timepoints_per_voxel=250,
        mask_ratio=0.75,
        timepoint_patching_size=20,
        use_tanh_decoder=False,
        loss_fn="mse",
        segment_means_seq_len=64,
        num_landmarks=64,
        conv_kernel_size=65,
        inv_coeff_init_option=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.num_brain_voxels = num_brain_voxels
        self.num_timepoints_per_voxel = num_timepoints_per_voxel
        self.mask_ratio = mask_ratio
        self.timepoint_patching_size = timepoint_patching_size
        self.use_tanh_decoder = use_tanh_decoder
        self.norm_pix_loss = norm_pix_loss
        self.loss_fn = loss_fn
        self.segment_means_seq_len = segment_means_seq_len
        self.num_landmarks = num_landmarks
        self.conv_kernel_size = conv_kernel_size
        self.inv_coeff_init_option = inv_coeff_init_option
