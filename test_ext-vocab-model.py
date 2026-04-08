import soundfile as sf
import torch

from faster_qwen3_tts import FasterQwen3TTS

if __name__ == "__main__":

    model = FasterQwen3TTS.from_pretrained(
        "tmp/vocab-4096-checkpoint",
        device="cuda",
        dtype=torch.bfloat16,
    )

    wavs, sr = model.generate_custom_voice(
        text="i am a speech generation model that can sound like a real person who can read numbers like one one seven, one five three, two two three one.",
        speaker="rime-gold/am",
        language="english"
    )

    sf.write("tmp/faster-test-converted.wav", wavs[0], 24_000)
