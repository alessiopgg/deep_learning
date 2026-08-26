"""
Preprocessing audio per MAE-AST.

Pipeline:

    file audio
        ↓
    caricamento waveform
        ↓
    conversione mono
        ↓
    resampling a 16 kHz
        ↓
    rimozione componente DC
        ↓
    log-Mel filterbank
        ↓
    padding / troncamento
        ↓
    [solo train downstream] SpecAugment
        ↓
    normalizzazione
        ↓
    [solo train downstream] noise + time roll

L'output utilizza sempre la convenzione:

    (T, F)

Dove:
    T = frame temporali
    F = bin Mel
"""

from pathlib import Path
from typing import Literal

import soundfile as sf
import torch
import torchaudio
from omegaconf import DictConfig


class AudioPreprocessor:
    """
    Preprocessore audio condiviso tra pretraining e fine-tuning.

    Le augmentation downstream sono disattivate per default e vengono
    applicate solo quando ``augment=True`` e ``mode='finetune'``.
    """

    def __init__(
            self,
            cfg: DictConfig,
            norm_mean: float | None = None,
            norm_std: float | None = None,
    ):
        self.cfg = cfg

        self.norm_mean = norm_mean
        self.norm_std = norm_std

        # Cache dei resampler per evitare di ricrearli continuamente.
        self._resamplers: dict[
            tuple[int, int],
            torchaudio.transforms.Resample,
        ] = {}

    def process(
            self,
            wav_path: str | Path,
            mode: Literal["pretrain", "finetune"] = "pretrain",
            augment: bool = False,
    ) -> torch.Tensor:
        """
        Esegue l'intera pipeline di preprocessing.

        Nel fine-tuning con ``augment=True`` replica la recipe SSAST:
        SpecAugment prima della normalizzazione, poi noise e piccolo
        time-roll dopo la normalizzazione.

        Returns:
            Spettrogramma di shape (T, n_mels).
        """

        fbank = self.process_unnormalized(
            wav_path=wav_path,
            mode=mode,
        )

        if augment and mode == "finetune":
            fbank = self._apply_specaugment(fbank)

        fbank = self._normalize(fbank)

        if augment and mode == "finetune":
            fbank = self._apply_finetune_noise(fbank)

        return fbank.float()

    def process_fbank_raw(
            self,
            wav_path: str | Path,
    ) -> torch.Tensor:
        """
        Calcola il log-Mel/fbank prima di padding, troncamento e normalizzazione.

        Questo metodo è utile per l'EDA, perché permette di distinguere i frame
        realmente derivati dall'audio dai frame artificiali aggiunti in seguito.
        """

        waveform, sample_rate = self._load(wav_path)

        waveform = self._to_mono(waveform)

        waveform, sample_rate = self._resample_if_needed(
            waveform,
            sample_rate,
        )

        # Rimozione della componente continua del segnale.
        waveform = waveform - waveform.mean()

        return self._compute_fbank(
            waveform,
            sample_rate,
        ).float()

    def process_unnormalized(
            self,
            wav_path: str | Path,
            mode: Literal["pretrain", "finetune"] = "pretrain",
    ) -> torch.Tensor:
        """
        Esegue il preprocessing senza applicare la normalizzazione finale.

        Utile anche per il calcolo delle statistiche del dataset.
        """

        fbank = self.process_fbank_raw(
            wav_path=wav_path,
        )

        target_frames = self.target_frames(mode)

        fbank = self._pad_or_truncate(
            fbank,
            target_frames,
        )

        return fbank.float()

    def target_frames(
            self,
            mode: str,
    ) -> int:
        """Restituisce la lunghezza temporale richiesta."""

        if mode == "pretrain":
            return int(
                self.cfg.pretrain_target_frames
            )

        if mode == "finetune":
            return int(
                self.cfg.finetune_target_frames
            )

        raise ValueError(
            f"Modalità non supportata: {mode}"
        )

    def expected_shape(
            self,
            mode: str,
    ) -> tuple[int, int]:
        """Restituisce la shape attesa dello spettrogramma."""

        return (
            self.target_frames(mode),
            int(self.cfg.n_mels),
        )

    def _load(
            self,
            wav_path: str | Path,
    ) -> tuple[torch.Tensor, int]:
        """Carica il file audio."""

        wav_path = Path(wav_path)

        if not wav_path.exists():
            raise FileNotFoundError(
                f"File audio non trovato: {wav_path}"
            )

        # SoundFile mantiene l'I/O dei WAV semplice e portabile.
        data, sample_rate = sf.read(
            str(wav_path),
            dtype="float32",
            always_2d=True,
        )

        # SoundFile restituisce (N, C); il resto della pipeline usa (C, N).
        waveform = torch.from_numpy(data.T.copy())

        return waveform, int(sample_rate)

    @staticmethod
    def _to_mono(
            waveform: torch.Tensor,
    ) -> torch.Tensor:
        """Converte un eventuale audio multicanale in mono."""

        if waveform.ndim != 2:
            raise ValueError(
                "Waveform attesa con shape (C, N), "
                f"trovata {tuple(waveform.shape)}"
            )

        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True,
            )

        return waveform

    def _resample_if_needed(
            self,
            waveform: torch.Tensor,
            sample_rate: int,
    ) -> tuple[torch.Tensor, int]:
        """Effettua il resampling al sample rate configurato."""

        target_sample_rate = int(
            self.cfg.sample_rate
        )

        if (
                not self.cfg.resample
                or sample_rate == target_sample_rate
        ):
            return waveform, sample_rate

        key = (
            sample_rate,
            target_sample_rate,
        )

        if key not in self._resamplers:
            self._resamplers[key] = (
                torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=target_sample_rate,
                )
            )

        waveform = self._resamplers[key](waveform)

        return waveform, target_sample_rate

    def _compute_fbank(
            self,
            waveform: torch.Tensor,
            sample_rate: int,
    ) -> torch.Tensor:
        """
        Calcola le feature log-Mel tramite Kaldi fbank.
        """

        return torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=int(self.cfg.n_mels),
            dither=0.0,
            frame_shift=float(
                self.cfg.frame_shift_ms
            ),
            frame_length=float(
                self.cfg.frame_length_ms
            ),
            low_freq=float(
                self.cfg.f_min
            ),
            high_freq=float(
                self.cfg.f_max
            ),
        )

    @staticmethod
    def _pad_or_truncate(
            fbank: torch.Tensor,
            target_frames: int,
    ) -> torch.Tensor:
        """
        Porta tutti gli spettrogrammi alla stessa lunghezza temporale.
        """

        if fbank.ndim != 2:
            raise ValueError(
                "Fbank atteso con shape (T, F), "
                f"trovato {tuple(fbank.shape)}"
            )

        time_frames, n_mels = fbank.shape

        if time_frames == target_frames:
            return fbank

        if time_frames > target_frames:
            return fbank[:target_frames]

        output = torch.zeros(
            target_frames,
            n_mels,
            dtype=fbank.dtype,
        )

        output[:time_frames] = fbank

        return output

    def _apply_specaugment(
            self,
            fbank: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applica FrequencyMasking e TimeMasking come nella recipe SSAST.

        ``freqm`` e ``timem`` indicano la massima ampiezza della maschera.
        Con valori 0 l'operazione è disattivata.
        """

        freq_mask = int(
            self.cfg.get(
                "finetune_freq_mask",
                0,
            )
        )

        time_mask = int(
            self.cfg.get(
                "finetune_time_mask",
                0,
            )
        )

        if freq_mask <= 0 and time_mask <= 0:
            return fbank

        # torchaudio si aspetta (..., freq, time).
        augmented = fbank.transpose(0, 1).unsqueeze(0)

        if freq_mask > 0:
            augmented = torchaudio.transforms.FrequencyMasking(
                freq_mask_param=freq_mask,
            )(augmented)

        if time_mask > 0:
            augmented = torchaudio.transforms.TimeMasking(
                time_mask_param=time_mask,
            )(augmented)

        return augmented.squeeze(0).transpose(0, 1)

    def _apply_finetune_noise(
            self,
            fbank: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replica la semplice noise augmentation di SSAST.

        Quando abilitata:
        - aggiunge rumore uniforme positivo con ampiezza casuale <= 0.1;
        - esegue un piccolo roll temporale casuale (default ±10 frame).

        Usiamo il RNG di PyTorch per mantenere il comportamento coerente
        con i seed dei worker del DataLoader.
        """

        enabled = bool(
            self.cfg.get(
                "finetune_noise",
                False,
            )
        )

        if not enabled:
            return fbank

        max_scale = float(
            self.cfg.get(
                "finetune_noise_max_scale",
                0.1,
            )
        )

        if max_scale > 0.0:
            scale = torch.rand(
                (),
                device=fbank.device,
            ) * max_scale

            fbank = (
                    fbank
                    + torch.rand_like(fbank)
                    * scale
            )

        max_roll = int(
            self.cfg.get(
                "finetune_time_roll_max",
                10,
            )
        )

        if max_roll > 0:
            shift = int(
                torch.randint(
                    low=-max_roll,
                    high=max_roll,
                    size=(1,),
                    device=fbank.device,
                ).item()
            )

            fbank = torch.roll(
                fbank,
                shifts=shift,
                dims=0,
            )

        return fbank

    def _normalize(
            self,
            fbank: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalizza lo spettrogramma.

        Se sono disponibili statistiche globali del dataset,
        vengono utilizzate quelle.

        In caso contrario viene utilizzata una normalizzazione
        per singolo esempio.
        """

        if (
                self.norm_mean is None
                or self.norm_std is None
        ):
            mean = float(
                fbank.mean()
            )

            std = float(
                fbank.std(unbiased=False)
            )

        else:
            mean = self.norm_mean
            std = self.norm_std

        if self.cfg.normalize_to_half_std:
            denominator = std * 2.0
        else:
            denominator = std

        denominator = max(
            denominator,
            float(self.cfg.eps),
        )

        return (
                fbank - mean
        ) / denominator
