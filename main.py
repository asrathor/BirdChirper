import matplotlib.pyplot as plt
import librosa
import numpy as np
from pydub import AudioSegment
import wave
import sys
import librosa.display
import pywt
import math

def file_convert():

    #sound = AudioSegment.from_mp3("Acorn_Woodpecker_00001.mp3")
    #sound.export("Acorn_Woodpecker1",format="wav")
    #sound = AudioSegment.from_mp3("Allen_Hummingbird_00001.mp3")
    #sound.export("Allen_Hummingbird1", format="wav")
    sound = AudioSegment.from_mp3("Red-tailed_Hawk_00001.mp3")
    sound.export("RedTailed_HawkMono1", format="wav", parameters=["-ac","1"])

    return 0

def plot_datapoints(filename_wav):

    #source = wave.open(filename_wav,'r')
    #signal = source.readframes(-1)
    #signal = np.fromstring(signal, 'Int16')
    #fs = source.getframerate()
    arr = np.array((0,1))
    signal, sr = librosa.load(filename_wav)
    ft = np.fft.fft(signal)
    a = int(len(ft)/2)
    ft = ft[0:a]
    #spec_c=spectral_centroid(ft)
    #spec_b=signal_bandwidth(spec_c,ft)
    #spec_d=spectral_flatness(ft)
    spec_d = framesplit(signal)
    Time=np.linspace(0,len(signal)/sr,num=len(signal))
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(Time,signal)
    plt.subplot(2,1,2)
    #plt.plot(Time,lines)
    #plt.subplot(3,1,3)
    plt.plot(abs(ft))
    #plt.show()

    return 0

#Roll-off frequency

def rolloff_frequency(filename_wav):

    y, sr = librosa.load(filename_wav)
    rolloff=librosa.feature.spectral_rolloff(y=y,sr=sr)
    print(rolloff)
    print("Size:",rolloff.size)
    return 0

def spectogram(filename_wav):

    y, sr = librosa.load(filename_wav)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_S = librosa.logamplitude(S, ref_power=np.max)
    print(log_S.shape)
    plt.figure(figsize=(12,4))
    print(log_S[:,0].size)
    new_arr= np.zeros(log_S.shape)
    print(new_arr.shape)
    for i in range(log_S[0,:].size):
        arr = log_S[:,i]
        arr_max = np.amax(arr)
        index= arr.argmax(axis=0)
        #print("index:",index)
        thres = 1.2*arr_max
        #print("Threshold:",thres)
        #j=arr_max
        for j in range (index,log_S[:,i].size,1):
            #print("j",j)
            if(thres<=log_S[j,i]):
                #new_arr[j,i]=log_S[j,i]
                new_arr[j,i]=1
            else:
                break
        for k in range (index,0,-1):
            #print("k",k)
            if(thres<=log_S[k,i]):
                #new_arr[j,i]=log_S[j,i]
                new_arr[k,i]=1
            else:
                break

    prom_arr= new_arr*log_S

    for i in range(new_arr[0,:].size):
        for j in range(new_arr[:,0].size):
            if(new_arr[j,i]==0):
                prom_arr[j,i]=-80

    #print(log_S)
    #print(new_arr)
    plt.subplot(3,1,1)
    librosa.display.specshow(log_S,sr=sr,x_axis='time',y_axis='mel')
    plt.colorbar(format='%+02.0f dB')
    plt.subplot(3,1,2)
    librosa.display.specshow(new_arr, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+02.0f dB')
    plt.subplot(3,1,3)
    librosa.display.specshow(prom_arr, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+02.0f dB')

    plt.tight_layout()

    plt.show()
    return 0

def mfcc(filename_wav):

    y, sr = librosa.load(filename_wav)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=130)
    log_S = librosa.logamplitude(S, ref_power=np.max)
    mfc = librosa.feature.mfcc(S=log_S, n_mfcc=13)

    delta_mfcc = librosa.feature.delta(mfc)
    delta2_mfcc = librosa.feature.delta(mfc, order=2)

    plt.figure(figsize=(12,6))

    plt.subplot(3,1,1)
    librosa.display.specshow(mfc)
    plt.ylabel('MFCC')
    plt.colorbar()

    plt.subplot(3, 1, 2)
    librosa.display.specshow(delta_mfcc)
    plt.ylabel('MFCC-$\Delta$')
    plt.colorbar()

    plt.subplot(3, 1, 3)
    librosa.display.specshow(delta2_mfcc, sr=sr, x_axis='time')
    plt.ylabel('MFCC-$\Delta^2$')
    plt.colorbar()

    plt.tight_layout()

    plt.show()
    return 0


def wavelet(filename):

    cA, cD = pywt.dwt(filename,'haar')
    plt.plot(cA,cD)
    plt.show()

def spectral_centroid(ft):
    sumone=0
    sumtwo=0
    for i in range(len(ft)):
        sumone = sumone + abs(ft[i])**2

    for j in range(len(ft)):
        sumtwo = sumtwo + (abs(ft[j])**2)*j
    sc = sumtwo/sumone
    #print("sc",sc)
    return sc

def signal_bandwidth(sc,ft):
    sumone=0
    sumtwo=0
    for i in range(len(ft)):
        sumone = sumone + abs(ft[i])**2
    for j in range(len(ft)):
        sumtwo = sumtwo + (((j-sc)**2)*abs(ft[j]))
    bw = (sumtwo/sumone)**(1/2)
    #print("bw",bw)
    return bw

def rolloff_freq(ft):
    pow = 0
    sumone=0
    rf=0
    #print("max",max(abs(ft))**2)
    for i in range(len(ft)):
        pow = abs(ft[i])**2
        rf = max(pow,sumone)
        if(sumone<pow):
            sumone=pow
    #print("rf_pow",rf)
    thres = 0.95*rf

    for j in range(len(ft)):
        if(thres==abs(ft[j])):
            rf=abs(ft[j])
            break
    #print("rf",rf)
    return rf

def delta_spectrum(ft):
    dsm=0
    for i in range(1,len(ft)):
        dsm = dsm + abs(ft[i-1])-abs(ft[i])

    #print("dsm",dsm)
    return dsm

def spectral_flatness(ft):

    gm = abs(ft[0])
    am = 0
    sf = 0
    for i in range(1, len(ft)):

        gm = gm * (abs(ft[i])**(1/len(ft)))

    for j in range(len(ft)):
        am = am + abs(ft[i])

    amm = am/len(ft)
    res = gm/amm
    sf = 10 * math.log10((res))
    #print("sf",sf)
    return sf

def framesplit(signal):
    frames=0
    samplesize = len(signal)
    framesize = samplesize / 100
    x=0
    avg=0
    while(frames<100):

        sig = signal[int(x):int(framesize+x)]
        ft = np.fft.fft(sig)
        ftt = ft[0:int(len(ft/2))]
        sc = spectral_centroid(ftt)
        avg = avg+sc
        print("sc",sc)
        x = framesize+x
        frames = frames+1

    avgg=avg/100
    print("avg",avgg)
arr= np.array([])
for k in range(1,2):
    filepath = "Bird Calls/Blue Jays/Blue Jay "+ str(k)+".wav"
    run = plot_datapoints(filepath)
    #arr = np.append(arr,run)
print(arr.shape)



sys.exit(0)