% This tutorial script demonstrates how to implement non-negtive matrix
% factorization (NMF) on neural data.
%
% - Faxin Zhou, Feb. 12, 2026
% =========================================================================

% SET-UP WORKING DIRECTORY
% -------------------------------------------------------------------------
clear all
cd /Users/faxin/Documents/Github/FLM_Paper


% (HYPER)PARAMETERS
% -------------------------------------------------------------------------
fs = 100;  % the sampling rate of the neural signals
k = 2;  % number of clusters 
rng_seed = 25;  % fix the random seed for result repeatability
cond_ls = {'EN', 'FL', 'SI', 'OS'};


% FIX THE RANDOM SEED
% -------------------------------------------------------------------------
rng(rng_seed)


% DATA LOADING
% -------------------------------------------------------------------------
load('Data/NMF_data.mat', 'D');  % D is the time-series data
load('Data/NMF_data.mat', 'S');  % S is the trial-based data
load('Data/NMF_data.mat', 'ch');  % ch is the MNI coordinates of each elec.

D_concat = []; Tstamp = [];
for c = 1 : length(S)
    D_concat = [D_concat, S(c).data];
    Tstamp = [Tstamp, size(S(c).data, 2)];
end
[n, t] = size(D_concat);  % n is num of elecs; t is time


% DATA VISUALIZATION
% -------------------------------------------------------------------------
% Check how the concatenated data looks like!
figure, imagesc(D_concat, [-2, 2]); colorbar
xline(cumsum(Tstamp), 'linewidth', 2)
title('scene data v1 (unsorted)', 'Fontsize', 16)
edges = [0 cumsum(Tstamp)];             
centers = edges(1 : end-1) + Tstamp / 2;    
xticks(centers)
xticklabels(cond_ls); set(gca,'TickLabelInterpreter','none');
xlabel('conditions')
ylabel('electrodes')
ax = gca; ax.FontSize = 16;


% CONDUCT NNMF
% -------------------------------------------------------------------------
D_concat(D_concat < 0) = 0;  % remove the elements less than 0
[W, H, R] = nnmf(D_concat, k);  % W (n * k) and H (k * t)


% W/H MATRIX VISUALIZATION
% -------------------------------------------------------------------------
figure, imagesc(H, [0, 0.03]); title('1st\_NMF\_H\_Matrix'); colorbar
xline(cumsum(Tstamp), 'linewidth', 2)
xticks(centers)
xticklabels(cond_ls); set(gca,'TickLabelInterpreter','none');
ax = gca; ax.FontSize = 16;
figure, imagesc(W); title('1st\_NMF\_W\_Matrix'); colorbar
ax = gca; ax.FontSize = 16;


% VISUALIZE TWO CLUSTERS IN THE BRAIN WITH MITHRA TOOLBOX
% -------------------------------------------------------------------------
% brain figures have been saved to "./Results/NMF_results.png"
[~, id] = max(W, [], 2);  % hard threshold clustering
ids = zeros(size(W)); for d = 1 : length(id), ids(d, id(d)) = 1; end
ids = logical(ids);

% OBTAIN BRAIN TEMPLATES
P.vis_mode = 'MNI';
[VT_lh, VT_rh] = brain_plot_prep(P, '/Users/faxin/Documents/Data_Analysis/Interesting/iEEG_visualization/visualization-tools-v2/matlab');

% OBTAIN MNI COORDINATES
ch_NMF_aud = ch(ids(:, 1), :);
ch_NMF_vis = ch(ids(:, 2), :);
ch_NMF_aud_lh = ch_NMF_aud(ch_NMF_aud(:, 1) < 0, :);
ch_NMF_aud_rh = ch_NMF_aud(ch_NMF_aud(:, 1) > 0, :);
ch_NMF_vis_lh = ch_NMF_vis(ch_NMF_vis(:, 1) < 0, :);
ch_NMF_vis_rh = ch_NMF_vis(ch_NMF_vis(:, 1) > 0, :);

% OBTAIN WEIGHTS
W_NMF_aud = W(ids(:, 1), 1);
W_NMF_vis = W(ids(:, 2), 2);
W_NMF_aud = (W_NMF_aud - min(W_NMF_aud)) / (max(W_NMF_aud) - min(W_NMF_aud));
W_NMF_vis = (W_NMF_vis - min(W_NMF_vis)) / (max(W_NMF_vis) - min(W_NMF_vis));
W_NMF_aud_lh = W_NMF_aud(ch_NMF_aud(:, 1) < 0, :);
W_NMF_aud_rh = W_NMF_aud(ch_NMF_aud(:, 1) > 0, :);
W_NMF_vis_lh = W_NMF_vis(ch_NMF_vis(:, 1) < 0, :);
W_NMF_vis_rh = W_NMF_vis(ch_NMF_vis(:, 1) > 0, :);

% VISUALIZATION 
ElecColor = [0.0, 0.0, 0.0];
BrainColor = [1, 1, 1];
alpha = 0.8;
radius = 2.8;
clim = [0, 1];
vis_cmap = cmap_gen([0.9, 0.9, 0.9], ...
                    sscanf('004980', '%2x%2x%2x', [1 3]) / 255);
aud_cmap = cmap_gen([0.9, 0.9, 0.9], ...
                    sscanf('a70000', '%2x%2x%2x', [1 3]) / 255);

figure,
ax1 = subplot(2, 2, 1);
VT_rh.PlotElecOnBrain(ch_NMF_aud_rh, ...
                      'ElecColor', W_NMF_aud_rh, ...
                      'BrainColor', BrainColor, ...
                      'flag_AddFigure', false, ...
                      'FaceAlpha', alpha, ...
                      'radius', radius, ...
                      'cmap', aud_cmap,...
                      'clim', clim);
title('NMF aud RH');

ax2 = subplot(2, 2, 2);
VT_lh.PlotElecOnBrain(ch_NMF_aud_lh, ...
                      'ElecColor', W_NMF_aud_lh, ...
                      'BrainColor', BrainColor, ...
                      'flag_AddFigure', false, ...
                      'FaceAlpha', alpha, ...
                      'radius', radius, ...
                      'cmap', aud_cmap,...
                      'clim', clim);
title('NMF aud LH');

ax3 = subplot(2, 2, 3);
VT_rh.PlotElecOnBrain(ch_NMF_vis_rh, ...
                      'ElecColor', W_NMF_vis_rh, ...
                      'BrainColor', BrainColor, ...
                      'flag_AddFigure', false, ...
                      'FaceAlpha', alpha, ...
                      'radius', radius, ...
                      'cmap', vis_cmap,...
                      'clim', clim);
title('NMF vis RH');

ax4 = subplot(2, 2, 4);
VT_lh.PlotElecOnBrain(ch_NMF_vis_lh, ...
                      'ElecColor', W_NMF_vis_lh, ...
                      'BrainColor', BrainColor, ...
                      'flag_AddFigure', false, ...
                      'FaceAlpha', alpha, ...
                      'radius', radius, ...
                      'cmap', vis_cmap,...
                      'clim', clim);
title('NMF vis LH');


