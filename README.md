# neural-voice-models

ONNX builds of open neural text-to-speech models, for on-device synthesis.

Everything here is converted from published checkpoints and redistributed under the
original licences. Nothing was trained here.

## German — `kokoro-de-thorsten-ep5`

| | |
|---|---|
| Architecture | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0) |
| Checkpoint | [Thorsten-Voice/Kokoro](https://huggingface.co/Thorsten-Voice/Kokoro), epoch 5 (Apache-2.0) |
| Voice data | [Thorsten-Voice dataset](https://www.thorsten-voice.de) — **CC0-1.0** |
| Sample rate | 24 000 Hz |
| Speakers | 1 (`thorsten`, male) |

The speaker recorded and released his own voice under CC0, so commercial use needs no
further permission.

### Why this build exists

Thorsten-Voice publishes PyTorch weights only. Getting a *clean* ONNX out of them took
four attempts; the traps are recorded so nobody repeats them:

1. **Ringing.** kokoro's `TorchSTFT` uses `torch.istft` (complex tensors), which the
   ONNX exporter cannot convert. Rewriting the STFT/iSTFT by hand *does* export — but
   the result rings audibly during speech at 4800/9600 Hz (the iSTFT frame rate and
   its harmonic): +11.8/+16.8 dB measured on a phone recording, versus +1.9/+0.4 dB in
   PyTorch. That was the first release of this file (`kokoro-de-thorsten-ep5`, now
   superseded). The fix is not to write an STFT at all: the `kokoro` package ships its
   own ONNX-safe one — `KModel(disable_complex=True)` selects `CustomSTFT`. With it the
   export measures +1.6/+3.4 dB, identical to PyTorch.
2. **Silently half-loaded weights.** kokoro's loader misses this checkpoint's
   `module.` prefix (DataParallel) and modern `parametrizations.weight.original0/1`
   weight-norm names, and `load_state_dict(strict=False)` swallows both — 235 of the
   decoder's 491 tensors load, the rest stay random, and the model emits white noise.
   The export script translates the keys (548 tensors).
3. **Training noise leaking into inference.** The vocoder's `torch.randn_like` /
   `torch.rand` calls become `RandomNormalLike` nodes in ONNX: audible hiss, and the
   same sentence renders differently on every run. Disabled; output is deterministic.

The export script (`export_thorsten_kokoro_onnx.py` in the consuming project) verifies
all three before it will say "OK": ≥540 tensors loaded, deterministic output,
9600 Hz peak < 8 dB, max deviation from PyTorch < 0.1.

### Releases

| Asset | Status |
|---|---|
| `kokoro-de-thorsten-ep5-v2.tar.bz2` | **current** — CustomSTFT export, no ringing |
| `kokoro-de-thorsten-ep5.tar.bz2` | superseded — hand-rolled STFT, rings at 9600 Hz; kept for reproducibility |

### Contents

```
kokoro-de-thorsten-ep5-v2/
  model.onnx     325 MB   inputs: tokens [1,N] int64, style [1,256] float32, speed [1] float32
  voices.bin     522 KB   510 × 256 float32, little-endian (one speaker)
  tokens.txt     687 B    the standard Kokoro 178-token IPA vocabulary
```

Phonemes must follow espeak-ng's IPA conventions — that is what the checkpoint was
trained on. `ʏ` is absent from the vocabulary and should be substituted with `y`.

## Licences

Model weights Apache-2.0; the German voice recordings CC0-1.0. Attribution to
Thorsten Müller (Thorsten-Voice) and the Kokoro authors is kept in this README.
