import matplotlib.pyplot as plt
import librosa
import numpy as np
from pydub import AudioSegment
import wave
import sys
import librosa.display
from scipy.fftpack import fft
from scipy.fftpack import ifft
import math

def file_convert(file):
    #sound = AudioSegment.from_mp3("Surfbird_00002.mp3")
    #sound.export("Surfbird2", format="wav", parameters=["-ac","1"])
    sound = AudioSegment.from_wav(file)
    sound.export("Mono/"+file, format="wav", parameters=["-ac","1"])

def frameFeature(file):
    y, sr = librosa.load(file)
    signaltime = librosa.get_duration(y=y, sr=sr)
    framelength = 0.004 #0.010 s = 10 ms
    numframes = signaltime/framelength#100
    framesize = int(len(y)/numframes)

    print("signal time: ", signaltime, "s")
#    print("signal time*sampling rate: ", signaltime*sr)
#    print("num data points: ", len(y))
#    print("number of frames: ", numframes)
#    print("framesize: ", framesize)

    fty = []
    for i in range(0,len(y),framesize):
        fty.append(fft(y[i:i+framesize]))

#    print("num. fty frames", len(fty))

##### CALCULATE SPECTRAL CENTROID #####
    sc = []
    for i in range(0,len(fty),1):
        sci = spectralCentroid(fty[i])
        sc.append(sci)

    scsum = 0
    for i in range(0, len(sc)):
        scsum = scsum + sc[i]
    scavg = scsum/len(sc)
    print("sc sum ",scsum) # Sum of Spectral Centroid Frames




    bw = []
    for i in range(0,len(fty),1):
        bwi = bandwidth(fty[i], sc[i])
        bw.append(bwi)

    bwsum = 0
    for i in range(0,len(bw)):
        bwsum = bwsum + bw[i]
    bwavg = bwsum/len(bw)
#    print("bw avg: ",bwavg) # this value seems useless
    print("bw sum: ",bwsum)
    exit()
#    print("num. bw frames", len(bw))

##### CALCULATE SPECTRAL ROLLOFF #####
    srf = []
    th = 0.92
    for i in range (0, len(fty), 1):
        l = int(len(fty[i])/2)
        h = int(l/4)
        srfi = h

        fsumt = 0
        for n in range(0, l+1, 1):
            fsumt = fsumt + fty[i][n]
        fsumh = 0
        fsumth = fsumt*th
        for n in range(0, l+1, 1):
            if(fsumh < fsumth):
                fsumh = fsumh + fty[i][n]
            else:
                if(n > srfi):
                    srfi = n
                break
        srf.append(srfi)

    srfsum = 0
    for i in range(0,len(srf)):
        srfsum = srfsum + srf[i]
    srfavg = srfsum/len(srf)
    print("srf avg: ",srfavg)
#    print("srf sum: ",srfsum) # this value seems useless

#    print("num. srf frames", len(srf))

##### CALCULATE BAND ENERGY RATIO #####
    ber = []
    for i in range (0, len(fty), 1):
        l = int(len(fty[i])/2)
        h = srfavg #int(l/4) #sr[i]
        fsumt = 0
        fsumh = 0
        for n in range (0, l+1, 1):
            fsumt = fsumt + fty[i][n]
            if(n < h):
                fsumh = fsumh + fty[i][n]
        beri = abs(fsumh/fsumt) # should this be abs(fsumh/fsumt) ?
        ber.append(beri)

    bersum = 0
    for i in range(0,len(ber)):
        bersum = bersum + ber[i]
    beravg = bersum/len(ber)
#    print("ber avg: ",beravg)
    print("ber sum: ",bersum)

#    print("num. ber frames", len(ber))

##### CALCULATE DELTA SPECTRUM MAGNITUDE #####
    dsm = []
    for i in range(0, len(fty)-1, 1):
        l = int(len(fty[i])/2)
        dsmi = 0
        for n in range (0, l+1, 1):
            if(i+1 < len(fty)-1):
                dsmi = dsmi + abs(abs(fty[i][n]) - abs(fty[i+1][n]))
            else:
                dsmi = dsmi + abs(fty[i][n])
        dsm.append(dsmi)

    dsmsum = 0
    for i in range(0,len(dsm)):
        dsmsum = dsmsum + dsm[i]
    dsmavg = dsmsum/len(dsm)
    print("dsm avg: ",dsmavg)
#    print("dsm sum: ",dsmsum)

#    print("num. dsm frames", len(dsm))

##### CALCULATE SPECTRAL FLATNESS #####
    sf = []
    for i in range (0, len(fty), 1):
        l = int(len(fty[i]/2))
        gm = 1
        am = 0
        for n in range (0, l, 1):
            gm = gm * abs(fty[i][n])**(1/l)
            am = am + abs(fty[i][n])
        am = am/l
        sfi = 10*math.log10(gm/am)
        sf.append(sfi)

    sfsum = 0
    for i in range(0,len(sf)):
        sfsum = sfsum + sf[i]
    sfavg = sfsum/len(sf)
    print("sf avg: ",sfavg)
#    print("sf sum: ",sfsum)

#    print("num. sf frames", len(sf))

##### CALCULATE SHORT TIME SIGNAL ENERGY #####
#    stse = []
#    for i in range (0, len(fty), 1):
#        l = int(len(fty[i]/2)
#        for n in range (0, l, 1):

# not finished here

##### CALCULATE ZERO CROSSING RATE #####
    zc = 0
    for i in range(0, len(y)-1, 1):
        if(y[i] > 0 and y[i+1] < 0):
            zc = zc + 1
        elif(y[i] < 0 and y[i+1] > 0):
            zc = zc + 1
    zcr = zc/len(y)
    print("zcr: ", zcr)

##### CALCULATE MIN/MAX FREQUENCIES #####

def freqBounds(file):
    y, sr = librosa.load(file)
    fty = fft(y)
    fty = fty[:int(len(fty)/2)]
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_s = librosa.logamplitude(spec, ref_power=np.max)
    faxis = np.linspace(0, 12000, num=len(fty), retstep=True)

    print("sr: ", sr)

    maxf = 0
    minf = 128

    for i in range(0, len(log_s), 1):
        for j in range(0, len(log_s[i]), 1):
            if(log_s[i][j] == 0): print(i)
            if(log_s[i][j] > -20 and i > maxf):
                maxf = i
            if(log_s[i][j] > -20 and i < minf):
                minf = i
    print("maxf: ", maxf)
    print("minf: ", minf)

#    dfty = np.diff(abs(fty))
#    print("minf: ",minf)
    print(len(y))
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(y)
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(faxis[0], abs(fty))
    plt.figure()
    librosa.display.specshow(log_s, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+02.0f dB')
    plt.grid()
    plt.show()

def spectralCentroid(fty):
    l = int(len(fty)/2)
    num = 0
    den = 0
    for z in range(0, l+1, 1):
        num = num + z*((abs(fty[z]))**2)
        den = den + abs(fty[z])**2
    sc = num/den
    return sc

def bandwidth(fty, sc):
    l = int(len(fty)/2)
    num=0
    den=0
    for n in range(0, l+1, 1):
        num = num + ((n-sc)**2) * abs(fty[n])
        den = den + abs(fty[n])**2
    bw = (num/den)**(1/2)
    print("num ",num)
    print("den ",den)

    return bw

def printFFT(file):
    y, sr = librosa.load(file)
    fty = fft(y)
    for n in range(0, len(fty)):
        print(abs(fty[n]));

def main():
    for i in range(1,2):
        filepath = "American Kestrel/AK"+str(i)+".wav" #92
        #filepath = "Mono/American Yellow Warbler/AYW"+str(i)+".wav" #101
        #filepath = "Mono/Blue Jays/Blue Jay "+str(i)+".wav" #151
        #filepath = "Mono/Canada Goose/Canada Goose "+str(i)+".wav" #156
        #filepath = "Mono/Northern Cardinal/Northern Cardinal "+str(i)+".wav" #58
        print(filepath)
        #file_convert(filepath)
        freqBounds(filepath)
        #frameFeature(filepath)
        #printFFT(filepath)
    sys.exit(0)

main()
