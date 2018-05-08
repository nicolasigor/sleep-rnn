function plotFourierSpectrum( signal, fs )

L = length(signal);
Y = fft( signal );
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs*(0:(L/2))/L;
figure, plot(f,P1);
xlabel('Frequency [Hz]');
ylabel('|F(f)|');
title('Power Spectrum')


