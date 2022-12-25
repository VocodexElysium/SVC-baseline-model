from .audio_utils import *

import torchaudio as tau

class AudioMetric:
    """Create a global frame for calculating audio metrics."""
    def __init__(self):
        self.AUDIO = {}
        self.SPECT = {}
        self.OTHER = {}
    
    def register(self, mode):
        if mode == 'audio':
            def r(f):
                self.AUDIO[f.__name__] = f
                return f
            return r
        if mode == 'spect':
            def r(f):
                self.SPECT[f.__name__] = f
                return f
            return r
        if mode == 'other':
            def r(f):
                self.OTHER[f.__name__] = f
                return f
            return r

class ParamMetric:
    """Create a global frame for calculating param metrics."""
    def __init__(self):
        self.WORLD  = {}
    
    def register(self, mode):
        if mode == "world":
            def r(f):
                self.WORLD[f.__name__] = f
                return f
            return r

AUDIOMETRIC = AudioMetric()
PARAMMETRIC = ParamMetric()

@PARAMMETRIC.register(mode='world')
def MSE(matrix1, matrix2):
    return np.mean(np.square(matrix1 - matrix2))

@PARAMMETRIC.register(mode='world')
def MAE(matrix1, matrix2):
    return np.mean(np.absolute(matrix1 - matrix2))

@PARAMMETRIC.register(mode='world')
def LSD(matrix1, matrix2):
    return np.mean(np.sqrt(np.mean(np.square(matrix1 - matrix2), axis=-1)))

@PARAMMETRIC.register(mode='world')
def COV(vector1, vector2):
    return np.mean(vector1 * vector2) - np.mean(vector1) * np.mean(vector2)

@AUDIOMETRIC.register(mode='spect')
def MSE(matrix1, matrix2):
    return ((matrix1 - matrix2) ** 2).mean()

@AUDIOMETRIC.register(mode='spect')
def MAE(matrix1, matrix2):
    return ((matrix1 - matrix2).abs()).mean()

@AUDIOMETRIC.register(mode='spect')
def LSD(matrix1, matrix2):
    return ((matrix1 - matrix2) ** 2).mean(dim=-1).sqrt().mean()

@AUDIOMETRIC.register(mode='other')
def COV(vector1, vector2):
    return (vector1 * vector2).mean() - vector1.mean() * vector2.mean()

@AUDIOMETRIC.register(mode='other')
def PCC(vector1, vector2):
    return COV(vector1, vector2) / vector1.std() / vector2.std()

# Attention: 'sample_rate' should be specified
@AUDIOMETRIC.register(mode='audio')
def AudioMSE(audio1, audio2, **melargs):
    return MSE(AudioToMel(audio1, **melargs), AudioToMel(audio2, **melargs))

# Attention: 'sample_rate' should be specified
@AUDIOMETRIC.register(mode='audio')
def AudioMAE(audio1, audio2, **melargs):
    return MAE(AudioToMel(audio1, **melargs), AudioToMel(audio2, **melargs))

# Attention: 'sample_rate' should be specified
@AUDIOMETRIC.register(mode='audio')
def AudioLSD(audio1, audio2, **melargs):
    return LSD(AudioToMel(audio1, **melargs), AudioToMel(audio2, **melargs))

# Attention: 'sample_rate' should be specified
@AUDIOMETRIC.register(mode='audio')
def AudioMFCCD(audio1, audio2, **mfccargs):
    return LSD(AudioToMFCC(audio1, **_args_to_metric_args(_use_mfccd=True, **mfccargs)), AudioToMFCC(audio2, **_args_to_metric_args(_use_mfccd=True, **mfccargs)))

# Attention: 'sample_rate' should be specified
@AUDIOMETRIC.register(mode='audio')
def AudioPCC(audio1, audio2, **melargs):
    mel1 = AudioToMel(audio1, **melargs)
    mel2 = AudioToMel(audio2, **melargs)
    #The output of AudioToMel is in the shape of (channel, n_mels, time).
    mel1 = mel1.reshape(mel1.shape[0], -1)
    mel2 = mel2.reshape(mel2.shape[0], -1)
    return PCC(mel1, mel2)

# Attention: 'sample_rate' should be specified
@AUDIOMETRIC.register(mode='audio')
def AudioMSSMAE(audio1, audio2, log_margin=1e-10, **specargs):
    loss = 0
    fft_sizes = len(get_config("default_metric_args")['n_ffts'])
    a = get_config("default_metric_args")['alpha']
    for i in range(fft_sizes):
        spec1 = AudioToSpec(audio1, **_args_to_metric_args(_use_mssmae=i, **specargs))
        spec2 = AudioToSpec(audio2, **_args_to_metric_args(_use_mssmae=i, **specargs))
        loss += MAE(spec1, spec2) + a * MAE(torch.log(spec1+log_margin), torch.log(spec2+log_margin))
    return loss / fft_sizes

def _args_to_metric_args(**args):
    """Adjust the args."""
    metric_args = deepcopy(args)
    if '_use_mfccd' in metric_args:
        if metric_args.pop('_use_mfccd'):
            metric_args['n_mfcc'] = get_config("default_metric_args")['n_mfcc']
    
    if '_use_mssmae' in metric_args:
        if 'sample_rate' in metric_args:
            metric_args.pop('sample_rate')
        i = metric_args.pop('_use_mssmae')
        metric_args['n_fft'] = get_config("default_metric_args")['n_ffts'][i]
        metric_args['hop_length'] = get_config("default_metric_args")['n_hops'][i]
    return metric_args

def ComputeAudioMetrics(pd_file, gt_file):
    """Compute the metrics for audio.
    
    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav")
        >>> waveform_hat, sample_rate_hat = torchaudio.load("test_hat.wav")
        >>> new_audio1, new_audio2 = AlignAudioLength(waveform, waveform_hat, mode='pad')
        >>> path1 = "audio1.wav"
        >>> path2 = "audio2.wav"
        >>> torchaudio.save(path1, new_audio1, sample_rate)
        >>> torchaudio.save(path2, new_audio2, sample_rate)
        >>> ComputeAudioMetrics(path1, path2)
            {'count': 1, 
            'AudioMSE': 0.0021917209960520267, 
            'AudioMAE': 0.0052396394312381744, 
            'AudioLSD': 0.029458042234182358, 
            'AudioMFCCD': 0.9125940203666687, 
            'AudioPCC': 0.9995368719100952, 
            'AudioMSSMAE': 0.14594422280788422, 
            'MSE': 0.9229055643081665, 
            'MAE': 0.06717035919427872, 
            'LSD': 0.40060627460479736, 
            'COV': 382.4127197265625, 
            'PCC': 0.9999992251396179}
    """
    profile = {'count': 1}
    pd, sr = tau.load(pd_file)
    gt, sr = tau.load(gt_file)
    pd_spec = AudioToSpec(pd)
    gt_spec = AudioToSpec(gt)
    for metric in AUDIOMETRIC.AUDIO:
        profile[metric] = float(AUDIOMETRIC.AUDIO[metric](pd, gt))
    for metric in AUDIOMETRIC.SPECT:
        profile[metric] = float(AUDIOMETRIC.SPECT[metric](pd_spec, gt_spec))
    for metric in AUDIOMETRIC.OTHER:
        profile[metric] = float(AUDIOMETRIC.OTHER[metric](pd_spec, pd_spec))
    return profile

def EvaluateAudio(folder, tqdm=True):
    """Evaluate the losses between Pred audio files and GroundTruth audio files in the given folder.
        
    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav")
        >>> waveform_hat, sample_rate_hat = torchaudio.load("test_hat.wav")
        >>> new_audio1, new_audio2 = AlignAudioLength(waveform, waveform_hat, mode='pad')
        >>> path1 = "audio1_gt.wav"
        >>> path2 = "audio1_pd.wav"
        >>> torchaudio.save(path1, new_audio1, sample_rate)
        >>> torchaudio.save(path2, new_audio2, sample_rate)
        >>> path = "/home/yichenggu/Sound2Synth-Linux-Reproduction"
        >>> EvaluateAudio(path)
            {'count': 1, 
            'AudioMSE': 0.0021917209960520267, 
            'AudioMAE': 0.0052396394312381744, 
            'AudioLSD': 0.029458042234182358, 
            'AudioMFCCD': 0.9125940203666687, 
            'AudioPCC': 0.99953693151474, 
            'AudioMSSMAE': 0.14594422280788422, 
            'MSE': 0.9229055643081665, 
            'MAE': 0.06717035919427872, 
            'LSD': 0.40060627460479736, 
            'COV': 381.51593017578125, 
            'PCC': 0.9999992251396179}
    """
    profile = {}
    instances = list(set(file.split('_')[0] for file in ListFiles(folder)))
    #TQDM is an iterable pbar defined in pyheaven.
    for instance in (TQDM(instances) if tqdm else instances):
        pd_file = pjoin(folder, instance + "_pd.wav")
        gt_file = pjoin(folder, instance + "_gt.wav")
        if ExistFile(pd_file) and ExistFile(gt_file):
            instance_profile = ComputeAudioMetrics(pd_file, gt_file)
            for key in instance_profile:
                profile[key] = profile[key] + instance_profile[key] if key in profile else instance_profile[key]
    return profile

def ComputeParamMetrics(pd_file, gt_file):
    """Compute the metrics for vocoder parameters.
    
    Example
        >>> path1 = 'datas/WORLDdata/sample1.WORLD_sp.npy'
        >>> path2 = 'datas/WORLDdata/sample1.WORLD_sp_hat.npy'
        >>> ComputeParamMetrics(path1, path2)
            {'count': 1, 
            'MSE': 0.0001421546357648842, 
            'MAE': 0.0007366000717481728, 
            'LSD': 0.002377360879054555, 
            'COV': 0.003887017222740962}
    """
    profile = {'count': 1}
    pd = np.load(pd_file)
    gt = np.load(gt_file)
    for metric in PARAMMETRIC.WORLD:
        profile[metric] = float(PARAMMETRIC.WORLD[metric](pd, gt))
    return profile

def EvaluateParam(folder, mode, tqdm=True):
    """Compute the metrics for vocoder parameters.
    
    Example
        >>> path = "datas/WORLDdata"
        >>> EvaluateParam(path, 'world')
            {'count': 1, 
            'MSE': 0.0001421546357648842, 
            'MAE': 0.0007366000717481728, 
            'LSD': 0.002377360879054555, 
            'COV': 0.003887017222740962}
    """
    profile = {}
    if mode == 'world':
        instances = list(set(file.split('.')[0] for file in ListFiles(folder)))
        for instance in (TQDM(instances) if tqdm else instances):
            pd_file = pjoin(folder, instance + ".WORLD_sp_hat.npy")
            gt_file = pjoin(folder, instance + ".WORLD_sp.npy")
            if ExistFile(pd_file) and ExistFile(gt_file):
                instance_profile = ComputeParamMetrics(pd_file, gt_file)
                for key in instance_profile:
                    profile[key] = profile[key] + instance_profile[key] if key in profile else instance_profile[key]
    return profile