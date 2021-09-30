import os,sys
#glob is for adding extensions and searching
import glob
import scipy
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import os 
import pathlib


def block_audio(x,blockSize,hopSize,fs):
    l = 0
    rows = blockSize
    columns = int(np.ceil(((np.size(x)-blockSize)/hopSize))+1)
    xb = xb = np.zeros((columns,rows))
    timeInSec = []
    for i in range (int(np.ceil(((np.size(x)-blockSize)/hopSize)+1))):
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
        #Creating the lag vector based on the matrix you drew
        x_lag= np.pad(inputVector,pad_width=(i,0),mode='constant')
        x_lag = x_lag[0:np.size(inputVector)]
        if(blsNormalized==True):
            y[i] = np.dot(inputVector,x_lag)
            lam_ba = 1/(np.dot(inputVector,y))
            #y[i] = np.dot(inputVector,x_lag)
            #print(lam_ba)
            y[i] = lam_ba*y[i]
            continue

        y[i] = np.dot(inputVector,x_lag)


        #print(x_lag)
    return y

def  get_f0_from_acf (r, fs):
    #r1 = comp_acf(r,False)
    #stp =0
    #Find inflection(Zero Crossing)
    #strt = np.where(r < 0)
    #strt = strt[0][0]
    #maxpeak2 = np.max(r[strt:])
    #fsample = np.where(r == maxpeak2)
    #freq = fs/fsample[0]
 #   stp = np.max([stp,strt])
 #   rdef = np.diff(r)
 #   strt = np.argmax(rdef > 0)
  #  stp = np.max([stp,strt])
  #  fsample = np.argmax(r[stp+1:])+1
  #  fsample = fsample+stp+1
  #  freq = fs/fsample
    rf = np.diff(r)
    #Zero crossing of diff is peak of r and ignores first peak as it does not cross zero, starts at zero
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
ferr = [np.abs(f0[i]-f1) if (timeInSec[i]<=1) else np.abs(f0[i]-f2) for i in range(np.size(f0))]
#for i in range(np.size(f0)):
#    if (timeInSec[i]<=1):
#        ferr[i] = np.abs(f0[i]-f1)
#    else:
#        ferr[i] = np.abs(f0[i]-f2)
plt.plot(timeInSec,f0)
plt.plot(timeInSec,(f0+ferr))
plt.show()



def convert_freq2midi(freqInHz):
    f_A4 = 440
    def mid_freq(f):
        f_A4=440
        Midi = 69 + 12*np.log2(f/f_A4)
        return (Midi)

    MID = np.vectorize(mid_freq)
    OUT_MID = MID(freqInHz)
 #   try:
 #      MID = [69 + int(12*np.log2(i/f_A4)) for i in(freqInHz)]
 #       MID = np.ceil(MID)
  #  except TypeError:
  #      MID = 69 + int(12*np.log2(int(freqInHz)/440))
  #      MID = np.ceil(MID)
    return OUT_MID
    
def eval_pitchtrack(estimateInHz,groundtruthInHz):
    def err_check(estimateInHz,groundtruthInHz):
        midi = convert_freq2midi(estimateInHz)
        midi2 = convert_freq2midi(groundtruthInHz)
        delc = 100*(midi-midi2)
        #delc = 1200*np.log2(estimateInHz/groundtruthInHz)
        return delc
    delc = np.vectorize(err_check)
    f = delc(estimateInHz,groundtruthInHz)
    f = np.sqrt(np.mean(np.square(f)))
    return f

def run_evaluation(path_to_data_folder):
    ftxt =[]
    faud=[]
    #[f.append(i) for i in os.listdir(path_to_data_folder)]
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
    feval = []
    err_rms=[]
    for aud in range(len(y_)):
        xb,timeInSec= block_audio(y_[aud],1024,512,fs_[aud])
        f0 =[]
        f0 = [comp_acf(xb[m],False) for m in range(xb.shape[0])]
        fl = [get_f0_from_acf(n, fs) for n in f0]
        df = pd.read_fwf(ftxt[aud],header=None)
        fchk = df[2]
        for i in range(len(fchk)-1):
            if (fchk[i]==min(fchk)):
                fl[i] = 0
                continue
        #funda.append(fchk)
        #feval.append(fl)
        err_rms.append(eval_pitchtrack())
    errCentRms = np.sqrt(np.mean(np.square(err_rms)))
    return errCentRms
        

            







    
    

    #targetPattern = "*.txt"
    #glob.glob(targetPattern)
   # list(pathlib.Path(path_to_data_folder).glob(targetPattern))
    return fl


    

