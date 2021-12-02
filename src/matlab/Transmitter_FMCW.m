function Transmitter_FMCW()
    %%
    filename = "8-16kHz_20ms.wav";
    [Tx, Fs] = audioread(filename);
    %%
    duration = length(Tx)/Fs;
    while(1)
        sound(Tx, Fs);
        pause(duration);
    end
end