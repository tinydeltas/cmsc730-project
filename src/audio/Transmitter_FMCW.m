function Transmitter_FMCW()
    %%
    filename = "transmit.wav";
    [Tx, Fs] = audioread(filename);
    %%
    duration = length(Tx)/Fs;
    while(1)
        sound(Tx, Fs);
        pause(duration);
    end
end