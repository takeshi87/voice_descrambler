# written by: takeshi87
# sorry for the quality, it was a quick experiment :)

import os
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from scipy import signal
from scipy.fft import fftshift


def c_cw(samplerate, freq, time):
	return np.exp(np.array(range(int(round(time*samplerate))))*2j*np.pi*(freq/samplerate))
	
	
def deprecated_cw(samplerate, freq, time):
	return np.cos(np.array(range(int(round(time*samplerate))))*2*np.pi*(freq/samplerate))
	
def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
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
    
    order = 42
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

    
def find_best_permutation(prob_segm_a_before_b):
    keys = prob_segm_a_before_b.keys()
    
    for a in keys:
        mprob = []
        for b in prob_segm_a_before_b[a]:
            if a//15 == b//15:
                mprob.append((prob_segm_a_before_b[a][b], b))
        mprob = list(sorted(mprob))
        print(f"For a={a}, smallestW possib: {mprob[:4]}")
    
    return

def permutRecAndConcat(rec, P):
        recP = [rec[P[i]] for i in range(len(rec))]
        return np.concatenate(recP)

def extract_transmissions_from_wav(fname, show=False):
    basename = os.path.splitext(os.path.split(fname)[1])[0]
    w = wave.open(fname)
    srate = w.getframerate()
    print(f"sampling rate: {srate}")
    
    preamble = gen_preamble(srate)
    postamble = gen_postamble(srate)
#    save_iq_real_part_to_wave(srate, "artif_post.wav", postamble)
#    save_iq_real_part_to_wave(srate, "artif_pream.wav", preamble)

    data=w.readframes(w.getnframes())
    w.close()
    audio=np.frombuffer(data, dtype='short')
    print("Data read")
    aiq = real_to_iq(srate, audio) #filtering embedded... is there some resulting time shift? #TODO: unhack!
    print("converted to I/Q")

    #pat_dem_pre = np.array(list(reversed(fm_demod(shift_freq(srate, preamble, -1800)))))
    pat_dem_pre = np.flip(fm_demod(shift_freq(srate, preamble, -1800)))
    
    #pat_dem_post = np.array(list(reversed(fm_demod(shift_freq(srate, postamble, -1800)))))
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

def replace_decoded_in_wav(fname, all_segm_transmissions, meta, testPermut):
    dirname = os.path.split(fname)[0]
    basename = os.path.splitext(os.path.split(fname)[1])[0]
    w = wave.open(fname)
    srate = w.getframerate()

    data=w.readframes(w.getnframes())
    w.close()
    audio=np.array(np.frombuffer(data, dtype='short'))

    for i, rec in enumerate(all_segm_transmissions):
        fix = permutRecAndConcat(rec, testPermut)
        fix = butter_bandpass_filter(fix, 100, 3500, srate, order=5)
        audio[meta['interval'][i][0]:meta['interval'][i][1]] = fix[:]

    save_iq_real_part_to_wave(srate, f"cut\\dscr_{basename}.wav", audio)


def main():
    testPermut = [14, 13, 11, 7, 8, 4, 3, 2, 1, 0, 5, 6, 9, 10, 12, 15, 29, 25, 17, 20, 26, 27, 23, 19, 21, 22, 18, 16, 28, 24, 36, 32, 44, 40, 38, 34, 42, 30, 35, 31, 43, 39, 37, 33, 41, 55, 53, 47, 51, 59, 57, 45, 49, 54, 52, 50, 48, 58, 56, 46, 62, 70, 74, 72, 60, 66, 64, 68, 63, 69, 73, 71, 61, 67, 65, 87, 85, 83, 79, 77, 75, 89, 81, 86, 84, 82, 80, 78, 76, 88, 96, 92, 104, 100, 98, 94, 102, 90, 95, 91, 103, 99, 97, 93, 101, 105, 119, 115, 111, 109, 117, 107, 113, 106, 118, 114, 112, 110, 116, 108, 132, 130, 128, 124, 122, 120, 134, 126, 131, 129, 127, 125, 123, 121, 133, 141, 137, 149, 145, 143, 139, 147, 135, 140, 136, 148, 144, 142, 138, 146, 158, 150, 162, 156, 164, 160, 154, 152, 157, 151, 161, 155, 163, 159, 153, 175, 173, 167, 171, 179, 177, 165, 169, 174, 172, 170, 168, 178, 176, 166, 180, 194, 190, 186, 184, 192, 182, 188, 181, 193, 189, 187, 185, 191, 183, 195, 209, 205, 201, 199, 207, 197, 203, 196, 208, 204, 202, 200, 206, 198, 218, 210, 222, 216, 224, 220, 214, 212, 217, 211, 221, 215, 223, 219, 213, 237, 235, 233, 229, 227, 225, 239, 231, 236, 234, 232, 230, 228, 226, 238, 242, 246, 244, 240, 252, 250, 254, 248, 243, 247, 245, 241, 251, 249, 253, 263, 261, 259, 269, 267, 257, 265, 255, 262, 260, 258, 268, 266, 256, 264, 274, 272, 280, 278, 276, 270, 284, 282, 275, 273, 281, 279, 277, 271, 283, 291, 287, 299, 295, 293, 289, 297, 285, 290, 286, 298, 294, 292, 288, 296, 314, 300, 308, 306, 304, 302, 312, 310, 313, 301, 309, 307, 305, 303, 311, 323, 315, 321, 328, 327, 326, 325, 329, 319, 317, 318, 324, 316, 322, 320, 332, 330, 344, 342, 340, 338, 334, 336, 341, 339, 337, 335, 333, 331, 343, 355, 353, 347, 351, 359, 357, 345, 349, 354, 352, 350, 348, 358, 356, 346, 370, 373, 372, 371, 374, 364, 362, 368, 366, 360, 365, 363, 369, 361, 367, 389, 375, 383, 381, 379, 377, 387, 385, 388, 376, 384, 380, 378, 386, 382, 394, 404, 402, 392, 400, 390, 397, 395, 398, 396, 393, 403, 401, 391, 399, 413, 405, 417, 419, 415, 409, 406, 416, 410, 418, 414, 411, 407, 412, 408, 420, 432, 430, 428, 424, 422, 434, 426, 431, 427, 425, 429, 423, 421, 433, 449, 445, 441, 447, 443, 439, 435, 437, 440, 446, 442, 438, 436, 448, 444, 456, 462, 458, 454, 455, 452, 459, 461, 457, 453, 451, 463, 460, 450, 464, 465, 471, 477, 479, 473, 467, 469, 475, 474, 478, 468, 476, 466, 472, 470, 482, 490, 494, 492, 484, 488, 483, 489, 493, 481, 487, 485, 480, 486, 491, 503, 495, 507, 501, 509, 505, 499, 497, 502, 496, 506, 500, 508, 504, 498, 515, 512, 514, 521, 518, 513, 520, 511, 517, 524, 522, 510, 516, 519, 523, 537, 538, 539, 530, 528, 536, 534, 532, 526, 527, 525, 533, 535, 531, 529, 544, 548, 543, 549, 542, 540, 550, 553, 551, 541, 547, 545, 554, 552, 546, 560, 566, 556, 568, 558, 561, 567, 563, 559, 555, 557, 569, 565, 562, 564, 574, 572, 571, 578, 581, 576, 584, 579, 583, 577, 570, 573, 575, 582, 580, 593, 587, 599, 591, 585, 594, 588, 592, 596, 597, 595, 586, 598, 589, 590, 607, 604, 612, 609, 611, 608, 602, 601, 603, 605, 600, 606, 614, 613, 610, 628, 621, 623, 624, 627, 619, 629, 615, 625, 617, 620, 626, 616, 622, 618, 631, 635, 638, 632, 644, 630, 633, 636, 634, 641, 637, 640, 639, 642, 643, 645, 647, 646, 654, 652, 655, 659, 649, 651, 658, 657, 648, 650, 656, 653, 664, 666, 661, 663, 660, 673, 665, 662, 668, 671, 670, 667, 669, 672, 674, 679, 687, 684, 686, 688, 678, 681, 680, 675, 677, 689, 685, 683, 682, 676, 704, 694, 698, 700, 701, 702, 703, 696, 695, 691, 693, 690, 692, 697, 699]
    #testPermutX = [0, 6, 3, 2, 12, 1, 13, 7, 10, 8, 11, 9, 4, 5, 14, 23, 21, 27, 19, 15, 29, 25, 17, 20, 26, 22, 18, 16, 28, 24, 42, 40, 38, 34, 32, 37, 35, 30, 44, 36, 41, 39, 33, 31, 43, 45, 59, 55, 51, 49, 57, 47, 53, 46, 58, 54, 52, 50, 56, 48, 66, 72, 68, 64, 60, 74, 70, 62, 65, 71, 67, 63, 61, 73, 69, 83, 81, 79, 89, 87, 77, 85, 75, 82, 80, 78, 88, 86, 76, 84, 98, 102, 92, 100, 90, 97, 95, 93, 103, 101, 91, 99, 96, 94, 104, 116, 106, 112, 110, 108, 118, 115, 113, 107, 111, 119, 117, 105, 109, 114, 120, 127, 125, 121, 129, 128, 126, 124, 134, 132, 122, 130, 123, 133, 131, 144, 138, 142, 146, 148, 141, 140, 135, 136, 147, 145, 139, 143, 137, 149, 158, 150, 162, 160, 154, 152, 156, 164, 157, 151, 161, 155, 163, 159, 153, 171, 167, 179, 175, 173, 169, 177, 170, 165, 178, 166, 172, 168, 176, 174, 186, 193, 192, 191, 185, 184, 182, 187, 188, 180, 183, 189, 181, 190, 194, 195, 198, 199, 208, 196, 204, 202, 206, 207, 205, 203, 197, 201, 209, 200, 220, 219, 215, 216, 214, 212, 218, 210, 211, 224, 223, 222, 221, 217, 213, 235, 227, 228, 226, 232, 233, 225, 237, 229, 230, 236, 238, 234, 239, 231, 244, 253, 241, 254, 240, 252, 251, 249, 243, 247, 248, 242, 246, 250, 245, 269, 257, 258, 263, 264, 255, 268, 265, 259, 266, 260, 262, 261, 267, 256]
    #testPermutY = [14, 4, 13, 0, 7, 5, 3, 9, 1, 2, 8, 6, 10, 12, 11, 27, 23, 17, 21, 19, 15, 16, 20, 22, 18, 28, 24, 26, 29, 25, 42, 40, 38, 34, 33, 31, 39, 37, 35, 32, 30, 44, 36, 41, 43, 45, 59, 55, 49, 57, 47, 50, 56, 48, 58, 54, 52, 51, 53, 46, 66, 62, 73, 71, 67, 60, 63, 61, 65, 70, 74, 68, 64, 72, 69, 75, 82, 87, 77, 85, 78, 81, 79, 89, 80, 83, 84, 88, 86, 76, 99, 90, 103, 98, 96, 94, 104, 102, 92, 100, 93, 97, 91, 101, 95, 115, 107, 105, 106, 108, 112, 111, 109, 117, 118, 116, 114, 110, 113, 119]

    files_to_use = [
    #"kiwi.24x7.hk_2022-03-06T13_04_36Z_5463.61_usn.wav", # low Q / low volume
    #"websdr_recording_2022-03-09T09_44_53Z_7175.0kHz.wav", # debug beg finding
    "websdr_recording_2022-03-09T10 19 56Z_7175.0kHz.wav", #n2
    "websdr_recording_2022-03-09T10 23 16Z_7175.0kHz.wav",
    "websdr_recording_2022-03-09T10 56 29Z_7175.0kHz.wav", #n3
    "websdr_recording_2022-03-09T10 57 44Z_7175.0kHz.wav", #n4
    "websdr_recording_2022-03-09T11 17 05Z_7175.0kHz.wav",
    "websdr_recording_2022-03-09T11 20 40Z_7175.0kHz.wav",
    "important\\websdr_recording_2022-03-09T11 37 18Z_7175.0kHz.wav",
    "websdr_recording_2022-03-09T11 39 23Z_7175.0kHz.wav",
    "disabling DSP noise red\\websdr_recording_2022-03-09T11 49 36Z_7175.0kHz.wav",
    "websdr_recording_2022-03-09T12 05 30Z_7175.0kHz.wav",
    "websdr_recording_2022-03-09T13 11 27Z_7175.0kHz.wav",
    #"websdr_recording_2022-03-09T15 39 17Z_7175.0kHz.wav" # tricky one, why? Didn't work, the only fragment found is incorrect
    "resample2022-03-11-around1900__4781_00_usb.wav"
    ]
     

    all_segm_transmissions = []
    all_recs_cut = []
    all_meta = {'timedesc':[], 'interval':[]}
    perFile = {}

    for fname in files_to_use:
        srate, segm_voice_transmissions, recs_cut, meta = extract_transmissions_from_wav(fname, show=True)
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

    testPermut = np.concatenate((testPermut, np.array(list(range(len(testPermut), longest_trans)), dtype=int)))
   
    for fname in files_to_use:
        replace_decoded_in_wav(fname, perFile[fname][0], perFile[fname][1], testPermut)


    for i, rec in enumerate(all_segm_transmissions):
        fix = permutRecAndConcat(rec, testPermut)
        fix = butter_bandpass_filter(fix, 100, 3500, srate, order=5)
        save_iq_real_part_to_wave(srate, f"cut\\descrambled_{all_meta['timedesc'][i]}.wav", fix)

    print("Possible descrambled files saved.")
    print((np.array(testPermut)%15).reshape(int(len(testPermut)//15), 15))
    lastP = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    print("Op list:")
    for i in range(int(len(testPermut)//15)):
        op=[0]*15
        for x in range(15):
            lastwas = lastP[x]%15
            nowis = testPermut[15*i + x]%15
            op[lastwas]=nowis
        lastP = np.array(testPermut[15*i : 15*(i+1)]) % 15
        print(np.array(op))
        
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


    weiF = open('weights.txt', "w")

    weiF.write(f"{longest_trans} {len(all_probs)}\n")
    for (a, b, c) in all_probs:
        weiF.write(f"{a} {b} {c}\n")
    weiF.close()

    print("Probs:")
    print(prob_segm_a_before_b[20])

    #find_best_permutation(prob_segm_a_before_b)
            
    for i, rec in enumerate(all_segm_transmissions):
        save_iq_real_part_to_wave(srate, f"cut\\part_{all_meta['timedesc'][i]}.wav", np.concatenate(rec))
        
        
    for i, rec in enumerate(all_recs_cut):
        save_iq_real_part_to_wave(srate, f"cut\\part_{all_meta['timedesc'][i]}_withBegEnd.wav", rec)
        


if __name__ == "__main__":
    main()
