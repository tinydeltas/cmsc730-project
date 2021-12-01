function Transmitter_FMCW()
    %%
    filename = "UpdatedResampledTx.wav";
    [Tx, Fs] = audioread(filename);
    %%
    duration = length(Tx)/Fs;
    while(1)
        sound(Tx, Fs);
        pause(duration);
    end
end