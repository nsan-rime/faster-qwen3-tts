import torch
import soundfile as sf
from torch import nn
from transformers import AutoConfig, AutoModel, AutoProcessor
from transformers.modeling_utils import PreTrainedModel

from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSTalkerForConditionalGeneration,
    Qwen3TTSTalkerModel,
)
from qwen_tts.core.models.configuration_qwen3_tts import (
    Qwen3TTSConfig,
    Qwen3TTSTalkerConfig,
)
from qwen_tts.core.models import Qwen3TTSProcessor
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

class ExtendedVocabQwen3TTSTalkerModel(Qwen3TTSTalkerModel):
    """
    Identical to Qwen3TTSTalkerModel except codec_embedding uses
    config.extended_vocab_size (dim-0) instead of config.vocab_size.
    """

    def __init__(self, config):
        super().__init__(config)
        extended_vocab_size = getattr(config, "extended_vocab_size", config.vocab_size)
        self.codec_embedding = nn.Embedding(extended_vocab_size, config.hidden_size)

class ExtendedVocabQwen3TTSTalkerForConditionalGeneration(Qwen3TTSTalkerForConditionalGeneration):
    """
    Identical to parent except self.model is an ExtendedVocabQwen3TTSTalkerModel.
    """

    def __init__(self, config: Qwen3TTSTalkerConfig):
        super().__init__(config)
        self.model = ExtendedVocabQwen3TTSTalkerModel(config)

class ExtendedVocabQwen3TTSForConditionalGeneration(Qwen3TTSForConditionalGeneration):
    """
    Identical to parent except self.talker is an ExtendedVocabQwen3TTSTalkerForConditionalGeneration.
    """

    def __init__(self, config):
        PreTrainedModel.__init__(self, config)
        self.config = config

        self.talker = ExtendedVocabQwen3TTSTalkerForConditionalGeneration(
            config.talker_config
        )

        # Everything else below is identical to parent's __init__
        if config.tts_model_type == "base":
            from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder
            self.speaker_encoder = Qwen3TTSSpeakerEncoder(config.speaker_encoder_config)
        else:
            self.speaker_encoder = None

        self.speech_tokenizer = None
        self.generate_config = None
        self.supported_speakers = config.talker_config.spk_id.keys()
        self.supported_languages = ["auto"]
        for lang_id in config.talker_config.codec_language_id.keys():
            if "dialect" not in lang_id:
                self.supported_languages.append(lang_id)
        self.speaker_encoder_sample_rate = config.speaker_encoder_config.sample_rate
        self.tokenizer_type = config.tokenizer_type
        self.tts_model_size = config.tts_model_size
        self.tts_model_type = config.tts_model_type
        self.post_init()

class ExtendedVocabQwen3TTSModel(Qwen3TTSModel):
    """
    Inherits the entire Qwen3TTSModel API:
        generate_custom_voice, generate_voice_clone, generate_voice_design,
        create_voice_clone_prompt, audio loading, tokenization, etc.

    The only override is from_pretrained to register extended model classes.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, ExtendedVocabQwen3TTSForConditionalGeneration)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path, fix_mistral_regex=True
        )
        generate_defaults = model.generate_config
        return cls(model=model, processor=processor, generate_defaults=generate_defaults)
