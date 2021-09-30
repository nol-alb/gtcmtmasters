import os
#glob is for adding extensions and searching
import glob
#import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np


def block_audio(x,blockSize,hopSize,fs):
    l = 0
    rows = blockSize
    columns = int(np.floor(((np.size(x))/hopSize)))
    xb = xb = np.zeros((columns,rows))
    timeInSec = []
    for i in range (int(np.ceil(((np.size(x))/hopSize)))):
        k = blockSize
        x1 = x[(i+l):(i+l+k)]
        if len(x1)<blockSize:
            x1 = np.pad(x1,pad_width=(0,blockSize-len(x1)),mode='constant')
            xb[i] = x1
            timeInSec.append((i+l)/fs)
            break
        xb[i] = x1
        timeInSec.append((i+l)/fs)
        l = l+hopSize-1
    timeInSec = np.asarray(timeInSec)
    return xb, timeInSec

def comp_acf(inputVector,blsNormalized):
    y = np.zeros(np.size(inputVector))

    for i in range(np.size(inputVector)):
        #Creating the lag vector based on the matrix
        x_lag= np.pad(inputVector,pad_width=(i,0),mode='constant')
        x_lag = x_lag[0:np.size(inputVector)]
        #if(blsNormalized==True):
          #  y[i] = np.dot(inputVector,x_lag)
           # lam_ba = 1/(np.dot(inputVector,y))
           # y[i] = lam_ba*y[i]
           # continue

        y[i] = np.dot(inputVector,x_lag)
    if(blsNormalized==True):
        y = y/y[0]
    return y

def  get_f0_from_acf (r, fs):
    rf = np.diff(r)
    #Zero crossing of diff is peak of r and ignores first peak as it does not cross zero
    smples = np.where(np.diff(np.sign(rf)))[0]+1
    rl = [r[i] for i in smples]
    fsample = np.where(r==max(rl))[0]
    #print(fsample)
    freq = fs/fsample
    return freq[0]

def track_pitch_acf(x,blockSize,hopSize,fs):
    xb,timeInSec = block_audio(x,blockSize,hopSize,fs)
    f0 = np.zeros(xb.shape[0])
    for i in range(xb.shape[0]):
        r = comp_acf(xb[i],False)
        f0[i]=get_f0_from_acf (r, fs)
    return f0,timeInSec


#Generating a 2 second Long Sinusoid
def testSinusoid():
    fs = 44100
    f1= 441
    f2 = 882
    duration = [1,1]
    x = np.zeros(2*fs)
    x[0:fs] = np.arange(0,fs*duration[0])*(f1/fs)
    x[fs:]= np.arange(0,fs*duration[1])*(f2/fs)
    SineWave = np.sin(2*np.pi*x)
    f0,timeInSec =track_pitch_acf(SineWave,1024,512,44100)
    #ferr = np.zeros(np.size(f0))
    ferr=[]
    ferr = [np.abs(f0[i]-f1) if (i<(np.size(f0)/2)) else np.abs(f0[i]-f2) for i in range(np.size(f0))]
    plt.figure()
    plt.plot(timeInSec,f0,)
    plt.title("F0 Detection")
    plt.ylabel("Frequency In Hertz")
    plt.xlabel("Time In Seconds")
    plt.savefig("F0Detection.png")
    plt.figure()
    plt.plot(timeInSec,ferr)
    plt.title("Error Graph")
    plt.xlabel("Time In Seconds")
    plt.ylabel("Error In Hertz")
    plt.savefig("Error.png")
    return 0

def convert_freq2midi(freqInHz):
    f_A4 = 440
    def mid_freq(f):
        f_A4=440
        Midi = 69 + 12*np.log2(f/f_A4)
        return (Midi)

    MID = np.vectorize(mid_freq)
    pitchInMidi = MID(freqInHz)
    return pitchInMidi

def eval_pitchtrack(estimateInHz,groundtruthInHz):
    def err_check(estimateInHz,groundtruthInHz):

        midi = convert_freq2midi(estimateInHz)
        midi2 = convert_freq2midi(groundtruthInHz)
        delc = 100*(midi-midi2)
        return delc
    delc = np.vectorize(err_check)
    errCentRms = delc(estimateInHz,groundtruthInHz)
    errCentRms = np.sqrt(np.mean(np.square(errCentRms)))
    return errCentRms

def run_evaluation(path_to_data_folder):
    ftxt =[]
    faud=[]
    fileDir = path_to_data_folder
    fileExt = ".txt"
    [ftxt.append(os.path.join(fileDir, _)) for _ in os.listdir(fileDir) if _.endswith(fileExt)]
    fileExt = ".wav"
    [faud.append(os.path.join(fileDir, _)) for _ in os.listdir(fileDir) if _.endswith(fileExt)]
    fs_=[]
    y_ =[]
    for i in(faud):
        samplerate, data = wavfile.read(i)
        fs_.append(samplerate)
        y_.append(data/max(data))
    k=0
    funda = []
    fchk=[]
    feval = []
    err_rms=[]
    ##Without Pandas
    for aud in range(len(y_)):
        readfile = open(ftxt[aud])
        for line in readfile:
            l = line.split()
            fchk.append(float(l[2]))



    for aud in range(len(y_)):
        #df = pd.read_fwf(ftxt[aud],header=None)
        #[fchk.append(i) for i in(df[2])]
        f0,timeInSec =track_pitch_acf(y_[aud],1024,512,44100)
        [feval.append(i) for i in f0]
    feval = [0 if(fchk[i]==min(fchk)) else feval[i] for i in range(len(feval))]
    fchk1 = [i for i in fchk if i != min(fchk)]
    feval1 = [i for i in feval if i != min(feval)]
    err = eval_pitchtrack(feval1,fchk1[0:len(feval1)])

    return err




