# Sample codes for audio adversarial example

1. Download the target model from http://download.tensorflow.org/models/speech_commands_v0.02.zip

2. Extract `conv_actions_frozen.pb` and `conv_actions_labels.txt` into `model/`

3. Download the VB100 dataset from http://zenodo.org/record/60375/files/vb100_audio.zip

4. Convert each audio into `data/vb100/` as `ffmpeg -i Acorn_Woodpecker_00001.mp3 -acodec pcm_f32le -ac 1 -ar 16000 data/vb100/Acorn_Woodpecker_00001.wav`

5. Run `runner/yes.sh` or `runner/no.sh`