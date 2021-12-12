function Receiver_Signal_FMCW(filename)
    %%
    %clc; clear all; close all;
    %%
    StoringDirectory = "";    % saving directory
    % filename = "Sample_1";              % name of a spectrogram file
    Fs = 48000;                         % sampling rate
    T = 0.02;                           % 20 ms
    duration = 1;                       % Data Collection
    plottt = 1;                         
    saving = 1;
    %%
    winLength = round(T*Fs);
    overlapLength = 0;
    NumFFT = winLength;
    win = hann(winLength,'periodic');
    %%
    mkdir(StoringDirectory);
    %%
    [Tx, FsTx] = audioread("8-16kHz_20ms.wav");
    y = resample(Tx,Fs,FsTx);
    TransmittedSig = [];
    for i = 1:duration/T
        TransmittedSig = [TransmittedSig; y];
    end
    %%
    recObj = audiorecorder(Fs, 16, 1, 0);
    %%
    disp('Start speaking.');
    recordblocking(recObj, duration);
    disp('End of Recording.');
    CollectedData = getaudiodata(recObj);
    
    %%
    received = CollectedData .* TransmittedSig;
    %%
    if(plottt)
        figure; 
        spectrogram(received, win, overlapLength, NumFFT, Fs, 'yaxis');
        title('Spectrogram');
    end
    [Spec, frequency, time] = spectrogram(received, win, overlapLength, NumFFT, Fs, 'yaxis');
    if(saving)
        outfile = strcat(StoringDirectory, filename, ".mat");
        audiowrite(filename + ".wav", received, Fs);
        save(outfile, "Spec");
    end
    
end