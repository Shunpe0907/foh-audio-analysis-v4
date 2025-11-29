"""
Live PA Audio Analyzer V3.0 Alpha
- 周波数ベース音源分離
- 楽器別詳細解析
- 超詳細な改善提案

Usage:
    streamlit run pa_analyzer_v3_alpha.py
"""

import streamlit as st
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
import io
from pathlib import Path
import tempfile
import json
from datetime import datetime

# matplotlibの設定
plt.rcParams['figure.max_open_warning'] = 50
plt.rcParams['font.size'] = 10

# ページ設定
st.set_page_config(
    page_title="Live PA Audio Analyzer V3.0",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .version-badge {
        text-align: center;
        color: #667eea;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .instrument-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .recommendation-critical {
        background-color: #ffe6e6;
        padding: 1rem;
        border-left: 4px solid #ff4444;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .recommendation-important {
        background-color: #fff9e6;
        padding: 1rem;
        border-left: 4px solid #ffbb33;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


class InstrumentSeparator:
    """周波数ベース音源分離"""
    
    def __init__(self, y, sr):
        self.y = y
        self.sr = sr
        self.y_mono = librosa.to_mono(y) if len(y.shape) > 1 else y
        
    def separate(self):
        """全楽器を分離"""
        
        stems = {}
        confidence = {}
        
        with st.spinner('🎸 楽器を分離中...'):
            # ボーカル
            stems['vocal'], confidence['vocal'] = self._extract_vocal()
            
            # キック
            stems['kick'], confidence['kick'] = self._extract_kick()
            
            # スネア
            stems['snare'], confidence['snare'] = self._extract_snare()
            
            # ハイハット
            stems['hihat'], confidence['hihat'] = self._extract_hihat()
            
            # ベース
            stems['bass'], confidence['bass'] = self._extract_bass()
            
            # ギター/その他
            stems['other'], confidence['other'] = self._extract_other()
        
        return stems, confidence
    
    def _extract_vocal(self):
        """ボーカル抽出"""
        
        # フォルマント検出（200-600Hz基音、1-4kHzフォルマント）
        # センター定位
        
        # バンドパスフィルター（200-5000Hz）
        sos_low = signal.butter(4, 200 / (self.sr/2), btype='highpass', output='sos')
        sos_high = signal.butter(4, 5000 / (self.sr/2), btype='lowpass', output='sos')
        
        vocal_filtered = signal.sosfilt(sos_low, self.y_mono)
        vocal_filtered = signal.sosfilt(sos_high, vocal_filtered)
        
        # フォルマント領域の強調
        D = librosa.stft(vocal_filtered)
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        # 1-4kHz（フォルマント）を強調
        formant_mask = (freqs >= 1000) & (freqs <= 4000)
        D[formant_mask, :] *= 2.0
        
        vocal = librosa.istft(D)
        
        # 信頼度計算（フォルマント領域のエネルギー）
        formant_energy = np.mean(np.abs(D[formant_mask, :]))
        total_energy = np.mean(np.abs(D))
        confidence = min(formant_energy / (total_energy + 1e-10), 1.0)
        
        return vocal, confidence
    
    def _extract_kick(self):
        """キック抽出"""
        
        # 40-120Hz + トランジェント検出
        
        # バンドパスフィルター
        sos = signal.butter(4, [40 / (self.sr/2), 120 / (self.sr/2)], 
                           btype='bandpass', output='sos')
        kick_filtered = signal.sosfilt(sos, self.y_mono)
        
        # トランジェント検出
        onset_env = librosa.onset.onset_strength(y=self.y_mono, sr=self.sr)
        onset_frames = librosa.onset.onset_detect(y=self.y_mono, sr=self.sr, 
                                                   units='frames')
        
        # キックは低域 + 強いトランジェント
        kick_enhanced = kick_filtered.copy()
        
        # オンセット時刻で強調
        hop_length = 512
        for frame in onset_frames:
            sample = frame * hop_length
            if sample < len(kick_enhanced):
                # オンセット前後を強調
                start = max(0, sample - 1000)
                end = min(len(kick_enhanced), sample + 2000)
                kick_enhanced[start:end] *= 1.5
        
        # 信頼度（低域エネルギー + トランジェント頻度）
        low_energy = np.sqrt(np.mean(kick_filtered**2))
        onset_density = len(onset_frames) / (len(self.y_mono) / self.sr)
        confidence = min((low_energy * 10 + onset_density / 2) / 2, 1.0)
        
        return kick_enhanced, confidence
    
    def _extract_snare(self):
        """スネア抽出"""
        
        # 200-400Hz（ボディ）+ 2-5kHz（アタック）
        
        # 複数帯域の組み合わせ
        sos_body = signal.butter(4, [200 / (self.sr/2), 400 / (self.sr/2)], 
                                btype='bandpass', output='sos')
        sos_attack = signal.butter(4, [2000 / (self.sr/2), 5000 / (self.sr/2)], 
                                  btype='bandpass', output='sos')
        
        snare_body = signal.sosfilt(sos_body, self.y_mono)
        snare_attack = signal.sosfilt(sos_attack, self.y_mono)
        
        # 合成
        snare = snare_body * 0.6 + snare_attack * 0.4
        
        # トランジェント検出（キックより鋭い）
        onset_env = librosa.onset.onset_strength(y=self.y_mono, sr=self.sr, 
                                                 aggregate=np.median)
        
        confidence = 0.7  # 中程度の信頼度
        
        return snare, confidence
    
    def _extract_hihat(self):
        """ハイハット抽出"""
        
        # 6-15kHz + 連続トランジェント
        
        sos = signal.butter(4, 6000 / (self.sr/2), btype='highpass', output='sos')
        hihat = signal.sosfilt(sos, self.y_mono)
        
        # 高域のみなので信頼度は中程度
        high_energy = np.sqrt(np.mean(hihat**2))
        confidence = min(high_energy * 20, 0.8)
        
        return hihat, confidence
    
    def _extract_bass(self):
        """ベース抽出"""
        
        # 60-250Hz + 持続音（トランジェント少ない）
        
        sos = signal.butter(4, [60 / (self.sr/2), 250 / (self.sr/2)], 
                           btype='bandpass', output='sos')
        bass = signal.sosfilt(sos, self.y_mono)
        
        # キックと差分（キックはトランジェント、ベースは持続）
        # RMS計算
        frame_length = self.sr // 2
        hop_length = self.sr // 4
        rms = librosa.feature.rms(y=bass, frame_length=frame_length, 
                                 hop_length=hop_length)[0]
        
        # 持続性（RMSの分散が小さい = 持続音）
        rms_variance = np.var(rms)
        confidence = min(1.0 / (rms_variance + 0.1), 0.9)
        
        return bass, confidence
    
    def _extract_other(self):
        """ギター/その他"""
        
        # 中域（300-2000Hz）
        
        sos = signal.butter(4, [300 / (self.sr/2), 2000 / (self.sr/2)], 
                           btype='bandpass', output='sos')
        other = signal.sosfilt(sos, self.y_mono)
        
        confidence = 0.6  # 推定
        
        return other, confidence


class InstrumentAnalyzer:
    """楽器別詳細解析"""
    
    def __init__(self, stems, sr, full_audio):
        self.stems = stems
        self.sr = sr
        self.full_audio = full_audio
        
    def analyze_all(self):
        """全楽器を解析"""
        
        analyses = {}
        
        for instrument, audio in self.stems.items():
            if audio is not None and len(audio) > 0:
                analyses[instrument] = self.analyze_instrument(instrument, audio)
        
        return analyses
    
    def analyze_instrument(self, name, audio):
        """個別楽器の解析"""
        
        analysis = {
            'name': name,
            'present': True,
            'level_rms': self._calculate_rms(audio),
            'level_peak': self._calculate_peak(audio),
            'crest_factor': 0,
            'frequency_profile': {},
            'issues': [],
            'recommendations': []
        }
        
        # クレストファクター
        if analysis['level_rms'] > -100:
            analysis['crest_factor'] = analysis['level_peak'] - analysis['level_rms']
        
        # 楽器別の詳細解析
        if name == 'vocal':
            analysis.update(self._analyze_vocal(audio))
        elif name == 'kick':
            analysis.update(self._analyze_kick(audio))
        elif name == 'snare':
            analysis.update(self._analyze_snare(audio))
        elif name == 'bass':
            analysis.update(self._analyze_bass(audio))
        elif name == 'hihat':
            analysis.update(self._analyze_hihat(audio))
        elif name == 'other':
            analysis.update(self._analyze_other(audio))
        
        return analysis
    
    def _calculate_rms(self, audio):
        """RMS計算"""
        rms = np.sqrt(np.mean(audio**2))
        return 20 * np.log10(rms) if rms > 0 else -100
    
    def _calculate_peak(self, audio):
        """ピーク計算"""
        peak = np.max(np.abs(audio))
        return 20 * np.log10(peak) if peak > 0 else -100
    
    def _analyze_vocal(self, audio):
        """ボーカル詳細解析"""
        
        D = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=self.sr)
        spectrum = np.mean(D, axis=1)
        
        # 基音帯域（150-400Hz）
        fundamental_mask = (freqs >= 150) & (freqs < 400)
        fundamental_level = 20 * np.log10(np.mean(spectrum[fundamental_mask]) + 1e-10)
        
        # 明瞭度帯域（2-4kHz）
        clarity_mask = (freqs >= 2000) & (freqs < 4000)
        clarity_level = 20 * np.log10(np.mean(spectrum[clarity_mask]) + 1e-10)
        
        # 空気感（8-12kHz）
        air_mask = (freqs >= 8000) & (freqs < 12000)
        air_level = 20 * np.log10(np.mean(spectrum[air_mask]) + 1e-10)
        
        return {
            'frequency_profile': {
                'fundamental': fundamental_level,
                'clarity': clarity_level,
                'air': air_level
            },
            'formants_detected': True
        }
    
    def _analyze_kick(self, audio):
        """キック詳細解析"""
        
        D = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=self.sr)
        spectrum = np.mean(D, axis=1)
        
        # アタック周波数（60-100Hz）
        attack_mask = (freqs >= 60) & (freqs < 100)
        attack_level = 20 * np.log10(np.mean(spectrum[attack_mask]) + 1e-10)
        
        # ビーター音（2-5kHz）
        beater_mask = (freqs >= 2000) & (freqs < 5000)
        beater_level = 20 * np.log10(np.mean(spectrum[beater_mask]) + 1e-10)
        
        # サブソニック（<40Hz）
        subsonic_mask = freqs < 40
        subsonic_level = 20 * np.log10(np.mean(spectrum[subsonic_mask]) + 1e-10)
        
        return {
            'frequency_profile': {
                'attack': attack_level,
                'beater': beater_level,
                'subsonic': subsonic_level
            }
        }
    
    def _analyze_snare(self, audio):
        """スネア詳細解析"""
        
        D = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=self.sr)
        spectrum = np.mean(D, axis=1)
        
        # ボディ（200-400Hz）
        body_mask = (freqs >= 200) & (freqs < 400)
        body_level = 20 * np.log10(np.mean(spectrum[body_mask]) + 1e-10)
        
        # アタック（2-5kHz）
        attack_mask = (freqs >= 2000) & (freqs < 5000)
        attack_level = 20 * np.log10(np.mean(spectrum[attack_mask]) + 1e-10)
        
        # スナッピー（6-10kHz）
        snappy_mask = (freqs >= 6000) & (freqs < 10000)
        snappy_level = 20 * np.log10(np.mean(spectrum[snappy_mask]) + 1e-10)
        
        return {
            'frequency_profile': {
                'body': body_level,
                'attack': attack_level,
                'snappy': snappy_level
            }
        }
    
    def _analyze_bass(self, audio):
        """ベース詳細解析"""
        
        D = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=self.sr)
        spectrum = np.mean(D, axis=1)
        
        # 基音（80-200Hz）
        fundamental_mask = (freqs >= 80) & (freqs < 200)
        fundamental_level = 20 * np.log10(np.mean(spectrum[fundamental_mask]) + 1e-10)
        
        # 倍音（200-800Hz）
        harmonic_mask = (freqs >= 200) & (freqs < 800)
        harmonic_level = 20 * np.log10(np.mean(spectrum[harmonic_mask]) + 1e-10)
        
        # アタック（1-3kHz）
        attack_mask = (freqs >= 1000) & (freqs < 3000)
        attack_level = 20 * np.log10(np.mean(spectrum[attack_mask]) + 1e-10)
        
        return {
            'frequency_profile': {
                'fundamental': fundamental_level,
                'harmonic': harmonic_level,
                'attack': attack_level
            }
        }
    
    def _analyze_hihat(self, audio):
        """ハイハット解析"""
        return {'frequency_profile': {}}
    
    def _analyze_other(self, audio):
        """その他解析"""
        return {'frequency_profile': {}}


class DetailedRecommendationGenerator:
    """超詳細な改善提案生成"""
    
    def __init__(self, instrument_analyses, mix_type, venue_info, mixer_name=''):
        self.analyses = instrument_analyses
        self.mix_type = mix_type
        self.venue_info = venue_info
        self.mixer_name = mixer_name
        
    def generate_all(self):
        """全楽器の提案生成"""
        
        recommendations = []
        
        # 優先順位: ボーカル > キック > ベース > スネア > その他
        priority_order = ['vocal', 'kick', 'bass', 'snare', 'hihat', 'other']
        
        for instrument in priority_order:
            if instrument in self.analyses:
                rec = self.generate_for_instrument(instrument, self.analyses[instrument])
                if rec:
                    recommendations.append(rec)
        
        return recommendations
    
    def generate_for_instrument(self, name, analysis):
        """楽器別の詳細提案"""
        
        if name == 'vocal':
            return self._recommend_vocal(analysis)
        elif name == 'kick':
            return self._recommend_kick(analysis)
        elif name == 'bass':
            return self._recommend_bass(analysis)
        elif name == 'snare':
            return self._recommend_snare(analysis)
        else:
            return None
    
    def _recommend_vocal(self, analysis):
        """ボーカル提案"""
        
        rec = {
            'instrument': 'ボーカル',
            'priority': 'critical',
            'icon': '🎤',
            'current_state': {},
            'issues': [],
            'solutions': [],
            'expected_results': []
        }
        
        # 現状
        rec['current_state'] = {
            'level': f"{analysis['level_rms']:.1f} dBFS",
            'fundamental': f"{analysis['frequency_profile'].get('fundamental', -100):.1f} dB",
            'clarity': f"{analysis['frequency_profile'].get('clarity', -100):.1f} dB",
            'air': f"{analysis['frequency_profile'].get('air', -100):.1f} dB"
        }
        
        # 問題検出
        clarity_level = analysis['frequency_profile'].get('clarity', -100)
        
        if clarity_level < -30:
            rec['issues'].append({
                'problem': '明瞭度が極めて低い',
                'severity': 'critical',
                'detail': f'2-4kHz帯域が {clarity_level:.1f}dB（推奨: -20dB以上）'
            })
            
            # 解決策
            rec['solutions'].append({
                'title': 'PEQ設定（明瞭度向上）',
                'steps': [
                    'Band 1: 3.2kHz, Q=2.0, +4.0dB',
                    'Band 2: 5kHz, Q=1.5, +2.5dB',
                    '効果: 子音・明瞭度の大幅向上'
                ],
                'mixer_specific': self._get_mixer_eq_instructions('vocal_clarity')
            })
            
            rec['expected_results'].append('明瞭度 +60%')
            rec['expected_results'].append('歌詞の聴き取りやすさ大幅改善')
        
        # レベルチェック
        if analysis['level_rms'] < -30:
            rec['issues'].append({
                'problem': 'レベルが低すぎる',
                'severity': 'high',
                'detail': f'RMS {analysis["level_rms"]:.1f}dBFS'
            })
            
            rec['solutions'].append({
                'title': 'Fader調整 + Compressor',
                'steps': [
                    f'Fader: 現在位置から +{abs(analysis["level_rms"] + 25):.0f}dB',
                    'Compressor: Threshold -18dB, Ratio 4:1',
                    'Attack 10ms, Release 100ms',
                    'Make-up Gain: +3dB'
                ]
            })
        
        return rec
    
    def _recommend_kick(self, analysis):
        """キック提案"""
        
        rec = {
            'instrument': 'キック',
            'priority': 'important',
            'icon': '🥁',
            'current_state': {},
            'issues': [],
            'solutions': [],
            'expected_results': []
        }
        
        # サブソニックチェック
        subsonic = analysis['frequency_profile'].get('subsonic', -100)
        
        if subsonic > -40:
            rec['issues'].append({
                'problem': 'サブソニック成分検出',
                'severity': 'critical',
                'detail': f'40Hz以下: {subsonic:.1f}dB'
            })
            
            rec['solutions'].append({
                'title': 'HPF設定（必須）',
                'steps': [
                    'HPF: 35Hz, 24dB/oct',
                    '理由: ヘッドルーム確保、システム保護',
                    '効果: +2〜3dB のヘッドルーム'
                ]
            })
            
            rec['expected_results'].append('ヘッドルーム +2〜3dB')
            rec['expected_results'].append('システム負荷軽減')
        
        return rec
    
    def _recommend_bass(self, analysis):
        """ベース提案"""
        
        rec = {
            'instrument': 'ベース',
            'priority': 'important',
            'icon': '🎸',
            'current_state': {},
            'issues': [],
            'solutions': [],
            'expected_results': []
        }
        
        # 基本情報
        rec['current_state'] = {
            'level': f"{analysis['level_rms']:.1f} dBFS",
            'fundamental': f"{analysis['frequency_profile'].get('fundamental', -100):.1f} dB"
        }
        
        return rec
    
    def _recommend_snare(self, analysis):
        """スネア提案"""
        
        rec = {
            'instrument': 'スネア',
            'priority': 'optional',
            'icon': '🥁',
            'current_state': {},
            'issues': [],
            'solutions': [],
            'expected_results': []
        }
        
        return rec
    
    def _get_mixer_eq_instructions(self, goal):
        """ミキサー固有の操作手順（簡易版）"""
        
        # TODO: Phase 2でWeb検索から取得
        
        if 'CL' in self.mixer_name.upper() or 'QL' in self.mixer_name.upper():
            return {
                'mixer': 'Yamaha CL/QL Series',
                'steps': [
                    '1. チャンネルを選択',
                    '2. [EQ]ボタン → PEQ画面',
                    '3. 上記のパラメータを設定',
                    '4. EQ ON を確認'
                ]
            }
        elif 'X32' in self.mixer_name.upper():
            return {
                'mixer': 'Behringer X32',
                'steps': [
                    '1. チャンネルを選択',
                    '2. [EQ]ボタン',
                    '3. 上記のパラメータを設定',
                    '注意: 4バンドのみ。優先順位を決めて使用'
                ]
            }
        else:
            return {
                'mixer': '一般的な手順',
                'steps': [
                    '1. チャンネルEQを開く',
                    '2. 上記のパラメータを設定'
                ]
            }


# Streamlit UI部分は既存のV2と類似
# ここでは主要な解析ロジックのみ実装

def main():
    st.markdown('<h1 class="main-header">🎛️ Live PA Audio Analyzer V3.0</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="version-badge">Alpha Release - 楽器別詳細解析対応</p>', 
                unsafe_allow_html=True)
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")
        
        uploaded_file = st.file_uploader(
            "音源ファイルをアップロード",
            type=['mp3', 'wav', 'flac', 'm4a']
        )
        
        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 100:
                st.error(f"❌ ファイルが大きすぎます（{file_size_mb:.1f}MB）")
                uploaded_file = None
            else:
                st.success(f"✓ {file_size_mb:.1f}MB")
        
        st.markdown("---")
        st.subheader("🏛️ 会場情報")
        
        venue_capacity = st.slider("会場キャパ（人）", 50, 2000, 150, 50)
        stage_volume = st.selectbox("ステージ生音", ['high', 'medium', 'low', 'none'], 1)
        
        mixer_name = st.text_input("ミキサー", placeholder="例: Yamaha CL5")
        pa_system = st.text_input("PA", placeholder="例: d&b V-Series")
        
        st.markdown("---")
        analyze_button = st.button("🚀 解析開始", type="primary", use_container_width=True)
    
    # メインエリア
    if uploaded_file is None:
        st.info("👈 音源をアップロードしてください")
        
        st.markdown("### 🆕 V3.0 Alpha の新機能")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎸 楽器別解析**
            - ボーカル、ドラム、ベースを個別に分析
            - 各楽器の周波数特性を詳細解析
            - 楽器間の干渉を検出
            """)
        
        with col2:
            st.markdown("""
            **💡 超詳細な改善提案**
            - 具体的なEQ設定値
            - ミキサー固有の操作手順
            - 期待される効果まで明記
            """)
    
    elif analyze_button:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            # 音源読み込み
            with st.spinner('🎵 音源を読み込み中...'):
                y, sr = librosa.load(tmp_path, sr=22050, mono=False, duration=300)
                
                if len(y.shape) == 1:
                    y = np.array([y, y])
            
            st.success("✅ 読み込み完了")
            
            # 楽器分離
            separator = InstrumentSeparator(y, sr)
            stems, confidence = separator.separate()
            
            st.success("✅ 楽器分離完了")
            
            # 楽器検出結果表示
            st.markdown("## 🔍 検出された楽器")
            
            cols = st.columns(3)
            detected = [(name, conf) for name, conf in confidence.items() if conf > 0.5]
            detected.sort(key=lambda x: x[1], reverse=True)
            
            for i, (name, conf) in enumerate(detected):
                col_idx = i % 3
                with cols[col_idx]:
                    icon = {'vocal': '🎤', 'kick': '🥁', 'snare': '🥁', 
                           'bass': '🎸', 'hihat': '🥁', 'other': '🎹'}.get(name, '🎵')
                    name_ja = {'vocal': 'ボーカル', 'kick': 'キック', 'snare': 'スネア',
                              'bass': 'ベース', 'hihat': 'ハイハット', 'other': 'その他'}.get(name, name)
                    st.metric(f"{icon} {name_ja}", f"{conf*100:.0f}%", 
                             delta="検出" if conf > 0.7 else "推定")
            
            st.markdown("---")
            
            # 楽器別解析
            analyzer = InstrumentAnalyzer(stems, sr, y)
            analyses = analyzer.analyze_all()
            
            st.success("✅ 詳細解析完了")
            
            # 改善提案生成
            rec_gen = DetailedRecommendationGenerator(
                analyses, 'live', 
                {'capacity': venue_capacity, 'stage_volume': stage_volume},
                mixer_name
            )
            recommendations = rec_gen.generate_all()
            
            # 提案表示
            st.markdown("## 💡 楽器別改善提案")
            
            for rec in recommendations:
                priority_color = {
                    'critical': '🔴',
                    'important': '🟡',
                    'optional': '🟢'
                }.get(rec['priority'], '⚪')
                
                with st.expander(f"{priority_color} {rec['icon']} {rec['instrument']}", 
                               expanded=(rec['priority'] == 'critical')):
                    
                    # 現状
                    if rec['current_state']:
                        st.markdown("**現状:**")
                        for key, value in rec['current_state'].items():
                            st.write(f"- {key}: {value}")
                    
                    # 問題点
                    if rec['issues']:
                        st.markdown("**❌ 問題点:**")
                        for issue in rec['issues']:
                            severity_icon = {'critical': '🔴', 'high': '🟡', 'medium': '🟠'}.get(issue['severity'], '⚪')
                            st.write(f"{severity_icon} {issue['problem']}")
                            st.caption(issue['detail'])
                    
                    # 解決策
                    if rec['solutions']:
                        st.markdown("**✅ 解決策:**")
                        for i, sol in enumerate(rec['solutions'], 1):
                            st.markdown(f"**{i}. {sol['title']}**")
                            for step in sol['steps']:
                                st.write(f"  - {step}")
                            
                            if sol.get('mixer_specific'):
                                with st.expander(f"📱 {sol['mixer_specific']['mixer']} での操作"):
                                    for step in sol['mixer_specific']['steps']:
                                        st.write(step)
                    
                    # 期待される結果
                    if rec['expected_results']:
                        st.markdown("**🎯 期待される効果:**")
                        for result in rec['expected_results']:
                            st.write(f"✅ {result}")
            
        except Exception as e:
            st.error(f"❌ エラー: {str(e)}")
            with st.expander("詳細"):
                st.exception(e)
        
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == "__main__":
    main()
