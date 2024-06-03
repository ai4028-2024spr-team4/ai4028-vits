import os
import numpy as np
import torch
from torch import no_grad, LongTensor
import argparse
import commons
from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn
import gradio as gr
import librosa
import soundfile
import webbrowser

from flask import Flask, request, jsonify, send_file
import pickle
import numpy as np
from urllib import parse
app = Flask(__name__)




from text import text_to_sequence, _clean_text
device = "cuda:0" if torch.cuda.is_available() else "cpu"
import logging
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "한국어": "[KO]",
    "Mix": "",
}
lang = ['日本語', '简体中文', 'English', 'Mix','한국어']
def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(text, speaker=0, language="한국어", speed=1):
        if language is not None:
            text = language_marks[language] + text + language_marks[language]
        speaker_id = 0#speaker_ids[speaker]
        stn_tst = get_text(text, hps, False)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return "Success", (hps.data.sampling_rate, audio)

    return tts_fn

def create_vc_fn(model, hps, speaker_ids):
    def vc_fn(original_speaker, target_speaker, record_audio, upload_audio):
        input_audio = record_audio if record_audio is not None else upload_audio
        if input_audio is None:
            return "You need to record or upload an audio", None
        sampling_rate, audio = input_audio
        original_speaker_id = speaker_ids[original_speaker]
        target_speaker_id = speaker_ids[target_speaker]

        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != hps.data.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=hps.data.sampling_rate)
        with no_grad():
            y = torch.FloatTensor(audio)
            y = y / max(-y.min(), y.max()) / 0.99
            y = y.to(device)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(y, hps.data.filter_length,
                                     hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                     center=False).to(device)
            spec_lengths = LongTensor([spec.size(-1)]).to(device)
            sid_src = LongTensor([original_speaker_id]).to(device)
            sid_tgt = LongTensor([target_speaker_id]).to(device)
            audio = model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
                0, 0].data.cpu().float().numpy()
        del y, spec, spec_lengths, sid_src, sid_tgt
        return "Success", (hps.data.sampling_rate, audio)

    return vc_fn



# 모델 초기화 부분.
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="./OUTPUT_MODEL/G_latest.pth", help="directory to your fine-tuned model")
parser.add_argument("--config_dir", default="./finetune_speaker.json", help="directory to your model config file")
#parser.add_argument("--share", default=False, help="make link public (used in colab)")

args = parser.parse_args()
hps = utils.get_hparams_from_file(args.config_dir)
net_g = SynthesizerTrn(
    len(hps.symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).to(device)
_ = net_g.eval()


_ = utils.load_checkpoint(args.model_dir, net_g, None)
speaker_ids = hps.speakers
speakers = list(hps.speakers.keys())


# tts 함수 생성
tts_fn = create_tts_fn(net_g, hps, speaker_ids)


@app.route('/')
def home():
    return "tts model"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No input text provided'}), 400
    
    text = data['text']
    string = parse.unquote(text, 'utf8')
    
    # 텍스트를 음성으로 변환
    #text_output, audio_output = tts_fn("안녕하세요. 전기전자컴퓨터공학과 김종원 교수입니다.")
    text_output, audio_output = tts_fn(string)
    
    # 고유한 파일 이름 생성
    file_name = f"test.wav"
    
    # 음성 파일 저장
    #tts.save(file_name)
    soundfile.write(file_name, 
                audio_output[1], 
                audio_output[0], 
                format='WAV')

    # 파일을 응답으로 전송
    #return string
    return send_file(file_name, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
    
    # 코드 변환 - 한국어를 그대로 보낼 수 없으므로 아래 방식으로 변환 필요.
    #text = "안녕하세요. 전기전자컴퓨터공학과 김종원 교수입니다."
    #encode  = parse.quote(text)
    
    # 위 방식으로 텍스트 변환 시 결과
    # %EC%95%88%EB%85%95%ED%95%98%EC%84%B8%EC%9A%94.%20%EC%A0%84%EA%B8%B0%EC%A0%84%EC%9E%90%EC%BB%B4%ED%93%A8%ED%84%B0%EA%B3%B5%ED%95%99%EA%B3%BC%20%EA%B9%80%EC%A2%85%EC%9B%90%20%EA%B5%90%EC%88%98%EC%9E%85%EB%8B%88%EB%8B%A4.
    
    # app.run 실행 후 ,실행 예제 코드
    #curl -X POST -H "Content-Type: application/json" -d '{"text": "%EC%95%88%EB%85%95%ED%95%98%EC%84%B8%EC%9A%94.%20%EC%A0%84%EA%B8%B0%EC%A0%84%EC%9E%90%EC%BB%B4%ED%93%A8%ED%84%B0%EA%B3%B5%ED%95%99%EA%B3%BC%20%EA%B9%80%EC%A2%85%EC%9B%90%20%EA%B5%90%EC%88%98%EC%9E%85%EB%8B%88%EB%8B%A4."}' http://127.0.0.1:5000/predict --output output.wav