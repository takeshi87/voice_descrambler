# written by: takeshi87
# sorry for the quality, it was a quick experiment :)

import os
import wave
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import butter, lfilter, freqz, filtfilt
from scipy import signal
from scipy.fft import fftshift
import re

def c_cw(samplerate, freq, time):
	return np.exp(np.array(range(int(round(time*samplerate))))*2j*np.pi*(freq/samplerate))
	
	
def deprecated_cw(samplerate, freq, time):
	return np.cos(np.array(range(int(round(time*samplerate))))*2*np.pi*(freq/samplerate))
	
def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def silence(srate, time):
    return np.zeros((int(round(srate*time))))

def deprecated_fsk(srate, f_arr, baudrate, val):
    res = np.array([])
    time_symbol = 1 / baudrate
    for v in val:
        f_idx = (v=='1')
        f = f_arr[f_idx]
        res = np.concatenate((res, c_cw(srate, f, time_symbol)))
    return res

def cph_fsk(srate, f_arr, baudrate, val, override_dur = None):
    res = []
    time_symbol = 1 / baudrate
    samp_symbol = time_symbol*srate
    f0rot = np.exp(2j*np.pi*(f_arr[0]/srate))
    f1rot = np.exp(2j*np.pi*(f_arr[1]/srate))
    curVal = 1
    normalRange = int(round(len(val)*srate/baudrate))
    usedRange = normalRange if override_dur is None else int(round(override_dur*srate))
    for i in range(usedRange):
        idx = np.minimum(int(i//samp_symbol), len(val)-1)
        res.append(curVal)
        curVal = curVal * (f0rot if val[idx]=='0' else f1rot)
    return np.array(res)
    

SilenceShift = 0.001

def gen_preamble(srate=8000, f0 = 1200, f1 = 2400, fx = 2031, add_silence = 0.020-SilenceShift):
    return np.concatenate((c_cw(srate, fx, 0.264),
                           silence(srate, 0.064),
                           cph_fsk(srate, [f0, f1], 200, '11010100111011001001011000011100001111011100010100010000111110010000100010001000100000000000011011010100100010111001001011101000'),
                           silence(srate, add_silence)))
                           

    
def gen_postamble(srate=8000, f0 = 1200, f1 = 2400):
    return np.concatenate((silence(srate, SilenceShift), cph_fsk(srate, [f0, f1], 22.8, '1111000100110100', override_dur = 0.68)))
    


def rotation_vec(samplerate, freq, n):
    return np.exp(np.array(range(n))*2j*np.pi*(freq/samplerate))

def real_to_iq(srate, signal):
    shiftHalfFact = rotation_vec(srate, -2000, len(signal))
    shifted = signal * shiftHalfFact
    
    order = 22
    cutoff = 1900

    y = butter_lowpass_filter(shifted, cutoff, srate, order)
    assert(len(signal)==len(y))
    
    shiftHalfFactRev = rotation_vec(srate, 2000, len(signal))
    return y * shiftHalfFactRev
    
    

def shift_freq(srate, signal, shift):
    return signal*rotation_vec(srate, shift, len(signal))

def c_normalize(cxs):
    return cxs / (np.abs(cxs)+0.0000001)

def fm_demod(data):
    return c_normalize(np.conjugate(data[:-1]) * data[1:])
    #return np.angle(np.conjugate(data[:-1]) * data[1:])
    
    
    
def save_iq_real_part_to_wave(srate, fname, data):
    pwave = wave.open(fname, "wb")
    pwave.setnchannels(1)
    pwave.setsampwidth(2)
    pwave.setframerate(srate)
    factor = 30000/np.max(np.abs(data.real))
    pwave.writeframes(np.array(data.real*factor, dtype='short').tobytes())
    pwave.close()
    return
    
def match_transmission_start_stop(srate, pre, post):
    min_trans = 0.75
    min_samp_dist = min_trans * srate
    last_post = -2*min_samp_dist
    last_beg = None
    min_beg_post_samp_dist = 1.1 * srate
    postIdx=0
    res = []
    for beg in pre:
        if beg < last_post + min_samp_dist:
            print(f"too early peak... will update old entry: {beg/srate}")
            lastEnt = res[-1]
            print(f"Peak {lastEnt[0]/srate} has no matching postamble found - skipping transmission!!!" )
            res = res[:-1]
            postIdx = np.maximum(0, postIdx-1)

        while postIdx < len(post) and post[postIdx] < beg + min_beg_post_samp_dist:
            if beg > post[postIdx]:
                print(f"skipped old postamble: {post[postIdx]/srate}")
            postIdx += 1
        if postIdx == len(post):
            break
        res.append((beg, post[postIdx]))
        last_post = post[postIdx]
        postIdx += 1
    return res
        
def split_voice(audio, samp_in_part):
    res = []
    len_audio = len(audio)
    numSegments = int(round(len_audio/samp_in_part))
    for i in range(numSegments):
        if (i+1)*samp_in_part < len_audio:
            res.append(audio[i*samp_in_part:(i+1)*samp_in_part])
        else:
            segm = audio[i*samp_in_part:]
            res.append(np.concatenate((segm, np.zeros((samp_in_part-len(segm))))))
    return np.array(res)
    
        
def get_segm_voice_trans(srate, audio, pream_postam_pairs, pream_len, postam_len):
    res = []
    res2 = []
    timedesc = []
    inter = []
    singleSegmentSamples = int(round(0.044 * srate))
    for (a,b) in pream_postam_pairs:
        vbeg = a+pream_len
        vend = b
        nsamp = vend-vbeg
        numSegments = int(round(nsamp/singleSegmentSamples))
        inacc = nsamp - numSegments*singleSegmentSamples
        print(f"Inaccurate by: {inacc*1000/srate} ms")
        if abs(inacc)/srate > 0.0025:
            print("Inaccurate by more than 2.5ms! Skipping for now...")
            continue
        if numSegments%15 != 0:
            print(f"numSegments%15 != 0 (=={numSegments%15})! Skipping for now...")
            continue
        res.append(split_voice(audio[vbeg:vend], singleSegmentSamples))
        res2.append(np.array(audio[a:b+postam_len+1]))
        timedesc.append(f"_T+{int(round(a/srate))}s")
        inter.append((vbeg, vbeg+numSegments*singleSegmentSamples))
        print(f"sizeIs {np.sum([len(x) for x in res[-1]])} vs {numSegments*singleSegmentSamples}")
    return res, res2, {'timedesc' : timedesc, 'interval' : inter}
       
        

def show_spectrogram(srate, data):
    f, t, Sxx = signal.spectrogram(data, srate, return_onesided=False)
    plt.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show() 
    return
    #powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(aiq[:30*srate], Fs=srate)
    #plt.show()

def simp_qnorm_j(a):
    return a
    totpow = np.sum(a*a)
    targetTotPow = len(a)/20*1024
    return a * (targetTotPow / totpow)

# COST FUNCTION - probably good place to optimize to get better results
def judge_a_plus_b(srate, a, b):
    M=128
    window = signal.windows.kaiser(M, beta=4, sym=False)
    f1 = np.abs(np.fft.fft(a[-M:]*window))
    f2 = np.abs(np.fft.fft(b[0:M]*window))
    nf1 = simp_qnorm_j(f1)
    nf2 = simp_qnorm_j(f2)
    dd = nf1-nf2
    r = dd*dd
    return np.mean(r)


def permut_rec_window_and_concat(rec, P):
        if len(rec)==0:
            return np.array([])
        window = 1# signal.windows.kaiser(len(rec[0]), beta=1, sym=True)
        recP = [rec[P[i]]*window for i in range(len(rec))]
        return np.concatenate(recP)

def extract_transmissions_from_wav(fname, show=False):
    basename = os.path.splitext(os.path.split(fname)[1])[0]
    w = wave.open(fname)
    srate = w.getframerate()
    print(f"sampling rate: {srate}")
    
    preamble = gen_preamble(srate)
    postamble = gen_postamble(srate)

    data=w.readframes(w.getnframes())
    w.close()
    audio=np.frombuffer(data, dtype='short') #unhack, assumes 2 bytes per sample
    print("Data read")
    aiq = real_to_iq(srate, audio) #filtering embedded... is there some resulting time shift? #TODO: unhack!
    print("converted to I/Q")

    pat_dem_pre = np.flip(fm_demod(shift_freq(srate, preamble, -1800)))
    pat_dem_post = np.flip(fm_demod(shift_freq(srate, postamble, -1800)))
    
    shifted_ver = shift_freq(srate, aiq, -1800)
#    shifted_ver = butter_bandpass_filter(aiq, 450, 750, srate, order=5)
    dem = fm_demod(shifted_ver)
    dem_conj = np.conjugate(dem)
    #plt.plot(dem.imag)
    #plt.plot(dem.real)
    #plt.show()
    #show_spectrogram(srate, aiqf)
    #plt.plot(dem)
    #plt.show()

    conv_dem_pre = np.abs(signal.fftconvolve(dem_conj, pat_dem_pre, 'valid'))
    conv_dem_post = np.abs(signal.fftconvolve(dem_conj, pat_dem_post, 'valid'))

    conv_dem_pre-=np.mean(conv_dem_pre)
    conv_dem_post-=np.mean(conv_dem_post)

    THRESHOLD_PREAMBLE_PEAK_HEIGHT = 0.3
    THRESHOLD_PREAMBLE_PROMINENCE = 0.2
    THRESHOLD_POSTAMBLE_PEAK_HEIGHT = 0.3
    THRESHOLD_POSTAMBLE_PROMINENCE = 0.18

    pre_peaks, peak_prop = signal.find_peaks(conv_dem_pre, height = THRESHOLD_PREAMBLE_PEAK_HEIGHT * np.max(conv_dem_pre), distance = int(1.5*srate), prominence=THRESHOLD_PREAMBLE_PROMINENCE * np.max(conv_dem_pre), wlen=srate/10)
    print(np.array(pre_peaks)/srate)
#    print(peak_prop)
    po_peaks, po_peak_prop = signal.find_peaks(conv_dem_post, height = THRESHOLD_POSTAMBLE_PEAK_HEIGHT * np.max(conv_dem_post), distance = int(1.5*srate), prominence= THRESHOLD_POSTAMBLE_PROMINENCE * np.max(conv_dem_post), wlen=srate/8)
    print(np.array(po_peaks)/srate)
#    print(po_peak_prop)


    signal_intervals = match_transmission_start_stop(srate, pre_peaks, po_peaks)
    print("Scrambled fragments found:")
    print(np.array(signal_intervals)/srate)


    if show:
        plt.plot(np.array(range(len(conv_dem_pre)))/srate, conv_dem_pre)
        plt.plot(np.array(range(len(conv_dem_post)))/srate, conv_dem_post)
        plt.show()

    print(f"Num identified transmissions: {len(signal_intervals)}")
    print("Parts durations:")
    for (a,b) in signal_intervals:
        voicePartDur = (b-a-len(preamble))/srate
        print(f"{voicePartDur}")

    segm_voice_transmissions, allRecsCut, metaDesc = get_segm_voice_trans(srate, audio, signal_intervals, len(preamble), len(postamble))
    return srate, segm_voice_transmissions, allRecsCut, {'timedesc':[basename+"_"+i for i in metaDesc['timedesc']], 'interval' : metaDesc['interval']}

def replace_decoded_in_wav(fname, all_segm_transmissions, meta, scrambler_permut):
    dirname = os.path.split(fname)[0]
    basename = os.path.splitext(os.path.split(fname)[1])[0]
    w = wave.open(fname)
    srate = w.getframerate()

    data=w.readframes(w.getnframes())
    w.close()
    audio=np.array(np.frombuffer(data, dtype='short'))

    for i, rec in enumerate(all_segm_transmissions):
        fix = permut_rec_window_and_concat(rec, scrambler_permut)
        fix = butter_bandpass_filter(fix, 100, 3500, srate, order=5)
        audio[meta['interval'][i][0]:meta['interval'][i][1]] = fix[:]

    save_iq_real_part_to_wave(srate, f"cut\\dscr_{basename}.wav", audio)


def read_permutation_from_file(fname):
    perm = []
    f = open(fname, 'r')
    c = f.readlines()
    f.close()
    
    for line in c:
        for w in re.split(r'[,\s]+', line):
            if w!='':
                perm.append(int(w))
    print("Permutation:")
    print(perm)
    return perm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', dest='weights_fname', type=str, help='Generate weights file with provided name')
    parser.add_argument('-p', dest='perm_fname', type=str, help='Use provided file with list of integers as a scrambler permutation', default='permutation.txt')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('filenames', metavar='filename', type=str, nargs='+', help='wave file names to descramble or use for weights calculation')
    args = parser.parse_args()

    scrambler_permut = read_permutation_from_file(args.perm_fname)

    all_segm_transmissions = []
    all_recs_cut = []
    all_meta = {'timedesc':[], 'interval':[]}
    perFile = {}

    for fname in args.filenames:
        srate, segm_voice_transmissions, recs_cut, meta = extract_transmissions_from_wav(fname, show=args.verbose>0)
        all_segm_transmissions.extend(segm_voice_transmissions)
        all_recs_cut.extend(recs_cut)
        perFile[fname] = (segm_voice_transmissions, meta)
        all_meta['interval'].extend(meta['interval'])
        all_meta['timedesc'].extend(meta['timedesc'])
        
    print(f"num of transmissions collected: {len(all_segm_transmissions)}")

    trans_len = [len(trans) for trans in all_segm_transmissions]

    longest_trans = np.max(trans_len+[0])
    if longest_trans==0:
        return
    print(f"Longest transmission has {longest_trans} segments!")
    print(f"all trans len: {trans_len}")
    print(f"all trans len%15: {np.array(trans_len)%15}")

    scrambler_permut = np.concatenate((scrambler_permut, np.array(list(range(len(scrambler_permut), longest_trans)), dtype=int)))
   
    for fname in args.filenames:
        replace_decoded_in_wav(fname, perFile[fname][0], perFile[fname][1], scrambler_permut)


    for i, rec in enumerate(all_segm_transmissions):
        fix = permut_rec_window_and_concat(rec, scrambler_permut)
        fix = butter_bandpass_filter(fix, 100, 3500, srate, order=5)
        save_iq_real_part_to_wave(srate, f"cut\\descrambled_{all_meta['timedesc'][i]}.wav", fix)

    print("Possible descrambled files saved.")
    print((np.array(scrambler_permut)%15).reshape(int(len(scrambler_permut)//15), 15))
    
    if args.verbose>1:
        lastP = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        print("Op list:")
        for i in range(int(len(scrambler_permut)//15)):
            op=[0]*15
            for x in range(15):
                lastwas = lastP[x]%15
                nowis = scrambler_permut[15*i + x]%15
                op[lastwas]=nowis
            lastP = np.array(scrambler_permut[15*i : 15*(i+1)]) % 15
            print(np.array(op))

    for i, rec in enumerate(all_segm_transmissions):
        save_iq_real_part_to_wave(srate, f"cut\\part_{all_meta['timedesc'][i]}.wav", np.concatenate(rec))
        
        
    for i, rec in enumerate(all_recs_cut):
        save_iq_real_part_to_wave(srate, f"cut\\part_{all_meta['timedesc'][i]}_withBegEnd.wav", rec)

        
    if args.weights_fname is not None:
        print("Calculating weights...")

        max_straddle = 29 ##UNHACK

        prob_segm_a_before_b = {}
        all_probs = []
        for i in range(longest_trans):
            prob_segm_a_before_b[i] = {}
            for j in range(np.maximum(0, i-max_straddle), np.minimum(longest_trans-1, i+max_straddle+1)):
                if i==j:
                    continue
                prob_segm_a_before_b[i][j] = 0
                
                partial_metrics = []
                for rec in [rec for rec in all_segm_transmissions if len(rec) > np.maximum(i, j)]:
                    partial_metrics.append(judge_a_plus_b(srate, rec[i], rec[j]))
                prob_segm_a_before_b[i][j] = np.sqrt(np.mean(partial_metrics))
                all_probs.append((i, j, prob_segm_a_before_b[i][j]))

        weiF = open(args.weights_fname, "w")

        weiF.write(f"{longest_trans} {len(all_probs)}\n")
        for (a, b, c) in all_probs:
            weiF.write(f"{a} {b} {c}\n")
        weiF.close()

        print("Probs:")
        print(prob_segm_a_before_b[20])



if __name__ == "__main__":
    main()
