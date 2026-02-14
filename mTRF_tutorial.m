% This tutorial script demonstrates how to implement mTRF encoding model on
% neural data. In this tutorial, I will go through the application of the 
% mTRF model using an example patient data, with the audio spectrogram as 
% the stimulus feature (aka. the STRF model).
% 
% The current script applies the mTRF toolbox, which can be downloaded from
% here: https://github.com/mickcrosse/mTRF-Toolbox
%
% - Faxin Zhou, Feb. 12, 2026
% =========================================================================

% SET-UP WORKING DIRECTORY
% -------------------------------------------------------------------------
% clear all
cd /Users/faxin/Documents/Github/FLM_Paper


% LOAD NEURAL DATA
% -------------------------------------------------------------------------
y = load('Data/HG_data.mat').y;  % high-gamma data (ts * elecs)
ch = load('Data/HG_data.mat').ch;  % MNI coordinates
srate = 512;  % neural signal sampling rate


% LOAD AUDIO SPECTROGRAM
% -------------------------------------------------------------------------
% Frequencies were averaged into 10 bins for faster implementation.
X = load('Data/audio_spectrogram.mat').aud_spec;  % (ts * features)

% We can also visualize the spectrogram 
imagesc(X')
title('Audio Spectrogram')
xlabel('time (samples)')
ylabel('frequency (bins)')


% mTRF (HYPER)PARAMETERS
% -------------------------------------------------------------------------
tmin = 0; tmax = 400;  % mTRF window is 0 to 400 ms
nfold = 4;  % n fold cross validation
R_mod = 'ridge';  % methods for regularization. 'Tikhonov' or 'ridge'
R = 10000;  % regularization parameter;


% mTRF START!
% -------------------------------------------------------------------------
[r_mean, r, M] = CV_mTRF(X, y, srate, nfold, tmin, tmax, R, R_mod);


% VISUALIZE TWO CLUSTERS IN THE BRAIN WITH MITHRA TOOLBOX
% -------------------------------------------------------------------------
% brain figures have been saved to "./Results/mTRF_results.png"
% OBTAIN BRAIN TEMPLATES
P.vis_mode = 'MNI';
[VT_lh, VT_rh] = brain_plot_prep(P, '/Users/faxin/Documents/Data_Analysis/Interesting/iEEG_visualization/visualization-tools-v2/matlab');


% VISUALIZATION 
ElecColor = [0.0, 0.0, 0.0];
BrainColor = [1, 1, 1];
alpha = 0.8;
radius = 2.8;
clim = [0, 0.4];
aud_cmap = cmap_gen([0.9, 0.9, 0.9], ...
                    sscanf('a70000', '%2x%2x%2x', [1 3]) / 255);
figure,
VT_lh.PlotElecOnBrain(ch, ...
                      'ElecColor', r_mean', ...
                      'BrainColor', BrainColor, ...
                      'flag_AddFigure', false, ...
                      'FaceAlpha', alpha, ...
                      'radius', radius, ...
                      'cmap', aud_cmap,...
                      'clim', clim);


                  
                  

% =========================================================================

% mTRF TRAINGING FUNCTION
% -------------------------------------------------------------------------
function [r_mean, r, G_M] = CV_mTRF(X, Y, srate, nfold, tmin, tmax, R, R_mod)
    N = length(Y); 
    I = 1 : N;
    r = zeros(nfold, size(Y, 2));
    for n = 0 : nfold - 1
        % data split
        test_I = N * n / nfold + 1 :  N * (n + 1) / nfold;
        train_I = I; train_I(test_I) = [];
        Y_train = Y(train_I, :); Y_test = Y(test_I, :); 
        X_train = X(train_I, :); X_test = X(test_I, :);
        % model training
        M = mTRFtrain(X_train, Y_train, srate, 1, tmin, tmax, R, ...
                      'method', R_mod, 'split', 5, 'zeropad', 0);
        % model prediction
        [~, S] = mTRFpredict(X_test, Y_test, M);
        r(n + 1, :) = S.r;
        M.nfold = n + 1;
        G_M(n + 1) = M;
    end
    r_mean = mean(r);
end