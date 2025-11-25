function [TSEn] = TSMEFuDE(X, dim, nc, tau, Kmax)
% Time-Shift Multiscale Enhanced Fuzzy Dispersion Entropy (TSMEFuDE)
% Compute average entropy values for multiple time scales
%
% Input Parameters:
%   X    - Input time series (1×N vector)
%   dim  - Embedding dimension (positive integer)
%   nc   - Number of classes (positive integer)
%   tau  - Time delay (positive integer)
%   Kmax - Maximum time scale (positive integer)
%
% Output Parameters:
%   TSEn - Average entropy values at each scale (1×Kmax vector)
%
% Reference:
%   Based on Higuchi fractal dimension and ensemble fuzzy dispersion entropy
%
% Example:
%   X = randn(1,1000);
%   TSEn = TSMEFuDE(X, 3, 5, 1, 20);
%
% Version: 1.3
% Author: Juntong Li
% Date: 2024

    % Input validation
    arguments
        X (1,:) double
        dim (1,1) double {mustBeInteger, mustBePositive}
        nc (1,1) double {mustBeInteger, mustBePositive}
        tau (1,1) double {mustBeInteger, mustBePositive}
        Kmax (1,1) double {mustBeInteger, mustBePositive}
    end
    
    % Check if input data length is sufficient
    if length(X) < dim
        error('Time series length must be greater than embedding dimension dim');
    end
    
    % Adjust Kmax if it exceeds time series length
    if Kmax > length(X)
        warning('Kmax exceeds time series length, automatically adjusted to maximum value');
        Kmax = length(X);
    end
    
    TSEn = zeros(1, Kmax);
    
    % Progress display for long computations
    if Kmax > 10
        fprintf('Calculation progress: ');
    end
    
    % Main loop over all time scales
    for k = 1:Kmax
        try
            En = TSE_beta(X, k, dim, nc, tau);
            TSEn(k) = mean(En);
            
            % Display progress
            if Kmax > 10 && mod(k, ceil(Kmax/10)) == 0
                fprintf('%d%% ', round(100*k/Kmax));
            end
        catch ME
            warning('Calculation failed at k=%d: %s', k, ME.message);
            TSEn(k) = NaN;
        end
    end
    
    if Kmax > 10
        fprintf('\nCompleted!\n');
    end
end

function [En] = TSE_beta(S, K, dim, nc, tau)
% Time-Shift Entropy (TSE) - Generate subsequences and compute entropy
%
% Input Parameters:
%   S   - Original time series
%   K   - Time interval
%   dim - Embedding dimension
%   nc  - Number of classes
%   tau - Time delay
%
% Output Parameters:
%   En  - Entropy values for each subsequence (1×K vector)

    N = length(S);
    En = zeros(1, K); % Preallocate memory
    
    % Check if parallel computing toolbox is available
    use_parallel = (K > 5) && (license('test', 'Distrib_Computing_Toolbox'));
    
    if use_parallel
        % Parallel computation for better performance
        parfor m = 1:K
            indices = m:K:N;
            if length(indices) >= dim  % Ensure subsequence is long enough
                sub_series = S(indices);
                En(m) = EnsFuDE(sub_series, dim, nc, tau);
            else
                En(m) = NaN;  % Mark invalid computation
            end
        end
    else
        % Serial computation
        for m = 1:K
            indices = m:K:N;
            if length(indices) >= dim
                sub_series = S(indices);
                En(m) = EnsFuDE(sub_series, dim, nc, tau);
            else
                En(m) = NaN;
            end
        end
    end
    
    % Handle invalid values
    valid_idx = ~isnan(En);
    if sum(valid_idx) == 0
        error('All subsequences are too short for computation');
    end
end

function sub_series = generate_subseries(S, start_idx, interval, N)
% Generate subsequence for given starting index and interval
%
% Input Parameters:
%   S         - Original time series
%   start_idx - Starting index
%   interval  - Sampling interval
%   N         - Length of original time series
%
% Output Parameters:
%   sub_series - Generated subsequence

    indices = start_idx:interval:N;
    sub_series = S(indices);
end

% Optional: Visualization function
function plot_TSMEFuDE(TSEn, stdEn, Kmax)
% Plot TSMEFuDE results with error bars
%
% Input Parameters:
%   TSEn - Average entropy values
%   stdEn - Standard deviation of entropy values
%   Kmax  - Maximum time scale

    figure;
    errorbar(1:Kmax, TSEn, stdEn, 'o-', 'LineWidth', 2);
    xlabel('Time Scale K');
    ylabel('TSMEFuDE Entropy Value');
    title('Time-Shift Multiscale Enhanced Fuzzy Dispersion Entropy');
    grid on;
end