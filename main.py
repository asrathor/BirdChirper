import matplotlib.pyplot as plt
import librosa
import numpy as np
from pydub import AudioSegment
import wave
import sys
import librosa.display

def file_convert():

    #sound = AudioSegment.from_mp3("Acorn_Woodpecker_00001.mp3")
    #sound.export("Acorn_Woodpecker1",format="wav")
    #sound = AudioSegment.from_mp3("Allen_Hummingbird_00001.mp3")
    #sound.export("Allen_Hummingbird1", format="wav")
    sound = AudioSegment.from_mp3("Grey_Hawk_00001.mp3")
    sound.export("Grey_Hawk1", format="wav")

    return 0

def plot_datapoints(filename_wav):

    source = wave.open(filename_wav,'r')

    signal = source.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    fs = source.getframerate()

    #if source.getnchannels()==2:
    #    print ("Just mono files")
    #    sys.exit(0)

    Time=np.linspace(0,len(signal)/fs,num=len(signal))
    print("Time:",source.getnframes()/float(fs))
    plt.figure(1)
    plt.plot(Time,signal)
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
        print("index:",index)
        thres = 1.2*arr_max
        print("Threshold:",thres)
        #j=arr_max
        for j in range (index,log_S[:,i].size,1):
            print("j",j)
            if(thres<=log_S[j,i]):
                #new_arr[j,i]=log_S[j,i]
                new_arr[j,i]=1
            else:
                break
        for k in range (index,0,-1):
            print("k",k)
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

    print(log_S)
    print(new_arr)
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


number = file_convert()
#run1 = plot_datapoints("Acorn_Woodpecker1")
#run2 = plot_datapoints("Allen_Hummingbird1")
#run3 = rolloff_frequency("Acorn_Woodpecker1")
#run4 = rolloff_frequency("Allen_Hummingbird1")
#run5 = spectogram("Allen_Hummingbird1")
run6 = spectogram("Acorn_Woodpecker1")
#run7 = mfcc("Acorn_Woodpecker1")
#run8 = spectogram("Grey_Hawk1")
sys.exit(0)