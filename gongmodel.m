%% Stable Real-Time Plate Model Simulation Exactly as in the Paper
% This script implements the plate model described in:
% "On the Limits of Real-Time Physical Modelling Synthesis with a Modular Environment"
% using the exact parameters from the paper for a stable 41×41 grid at 44.1 kHz.
% The excitation is defined as per Equation (4).

clear; close all; clc;

%% 1. Grid Setup and State Array Initialization
% Plate dimensions (meters)
Lx = 1.0;  
Ly = 1.0;  

% Grid spacing (meters)
dx = 0.025;  % Use dx = dy = 0.025 m so that the grid has 41 points (1/0.025 = 40 intervals + 1)
dy = 0.025;  

% Define halo size for the 13-point stencil (2 ghost cells on all sides)
halo = 2;  

% Number of interior grid points
Nx = round(Lx/dx) + 1;  % Should be 41
Ny = round(Ly/dy) + 1;  % Should be 41

% Total grid size including halo layers
Nx_tot = Nx + 2 * halo;
Ny_tot = Ny + 2 * halo;

% Time step parameters
Fs = 44100;       
Ts = 1 / Fs;      

% Allocate state arrays:
% u_nm1: state at time step n-1 (previous)
% u_n:   state at time step n (current)
% u_np1: state at time step n+1 (new)
u_nm1 = zeros(Ny_tot, Nx_tot);
u_n   = zeros(Ny_tot, Nx_tot);
u_np1 = zeros(Ny_tot, Nx_tot);

% Define indices for the interior (excluding halo)
i_start = halo + 1;
i_end   = Ny_tot - halo;
j_start = halo + 1;
j_end   = Nx_tot - halo;

%% 2. Finite Difference Coefficient Computation
% Physical parameters (use the values given in the paper)
rho    = 7800;      % Density (kg/m^3)
H      = 0.005;     % Plate thickness (m)
E      = 2e11;      % Young's modulus (Pa)
nu     = 0.3;       % Poisson's ratio
sigma0 = 1;         % Damping parameter σ₀
sigma1 = 0.005;     % Damping parameter σ₁

% Derived bending stiffness parameter:
% κ² = E*H² / (12*rho*(1 - ν²))
kappa2 = E * H^2 / (12 * rho * (1 - nu^2));  % ≈ 58.73

% Define scale factor for spatial terms:
% S = (T_s² * κ²) / dx⁴
S = Ts^2 * kappa2 / dx^4;
% For Ts = 1/44100, dx = 0.025, S ≈ 0.0773

% Time-stepping factors for the semi-implicit scheme:
A = 1 + sigma0 * Ts;  % ≈ 1.0000227
B = sigma1 * Ts;      % ≈ 1.1338e-07

% Compute finite difference coefficients exactly as derived in the paper:
% Coefficients for u^n (current state):
B1 = S * 4;            % Immediate (distance 1) neighbors, ≈ 0.3092
B2 = S * (-1);         % Second neighbors (distance 2 along axes), ≈ -0.0773
B3 = S * 2;            % Diagonal neighbors, ≈ 0.1546
B4 = (2 - 20 * S) / A;   % Center coefficient, ≈ (2 - 1.546)/1.0000227 ≈ 0.454

% Coefficients for u^{n-1} (state two time steps back):
C1 = (4 * B) / A;      % Contribution from immediate neighbors in past state, ≈ 4.5352e-07
C2 = (-1) / A;         % Center coefficient for the past state, ≈ -0.99998

% Display computed coefficients
fprintf('Finite Difference Coefficients:\n');
fprintf('B1 = %g\n', B1);
fprintf('B2 = %g\n', B2);
fprintf('B3 = %g\n', B3);
fprintf('B4 = %g\n', B4);
fprintf('C1 = %g\n', C1);
fprintf('C2 = %g\n', C2);

%% 3. Simulation Initialization and Excitation Setup
% Initial conditions: u_nm1 and u_n are zero.
% Define excitation parameters as per Equation (4) in the paper:
f0    = 1.0;     % Maximum force amplitude (N)
t0    = 0.01;    % Strike start time (s)
T_exc = 0.005;   % Duration of the strike (s)
q     = 2;       % Parameter (q = 2 for a strike)

% Define excitation location (Dirac delta: a single grid point, here at the center)
exc_i = round((i_start + i_end) / 2);
exc_j = round((j_start + j_end) / 2);

% Preallocate output vector (to monitor displacement at the excitation point)
Nf = Fs;   % 1 second of simulation (44100 time steps)
out = zeros(Nf, 1);

%% 4. Time-Stepping Loop with Excitation Injection
for n = 1:Nf
    % Current simulation time
    current_time = (n - 1) * Ts;
    
    % Loop over the interior grid (excluding halo) to compute u_np1:
    for i = i_start : i_end
        for j = j_start : j_end
            term1 = B1 * ( u_n(i, j-1) + u_n(i, j+1) + u_n(i-1, j) + u_n(i+1, j) );
            term2 = B2 * ( u_n(i, j-2) + u_n(i, j+2) + u_n(i-2, j) + u_n(i+2, j) );
            term3 = B3 * ( u_n(i+1, j-1) + u_n(i+1, j+1) + u_n(i-1, j-1) + u_n(i-1, j+1) );
            term4 = B4 * u_n(i, j);
            term5 = C1 * ( u_nm1(i, j-1) + u_nm1(i, j+1) + u_nm1(i-1, j) + u_nm1(i+1, j) );
            term6 = C2 * u_nm1(i, j);
            
            u_np1(i, j) = term1 + term2 + term3 + term4 + term5 + term6;
        end
    end
    
    % Inject the excitation force if within the excitation time window:
    if current_time >= t0 && current_time <= (t0 + T_exc)
        fe = (f0 / 2) * (1 - cos((q * pi * (current_time - t0)) / T_exc));
        u_np1(exc_i, exc_j) = u_np1(exc_i, exc_j) + fe;
    end
    
    % Apply clamped boundary conditions by zeroing the halo regions:
    u_np1(1:halo, :) = 0;
    u_np1(end-halo+1:end, :) = 0;
    u_np1(:, 1:halo) = 0;
    u_np1(:, end-halo+1:end) = 0;
    
    % Record output at the excitation location
    out(n) = u_np1(exc_i, exc_j);
    
    % Swap state arrays for the next time step:
    u_nm1 = u_n;
    u_n   = u_np1;
    
    % Reset u_np1 to zero for the next iteration
    u_np1(:) = 0;
end

%% 5. Verification Checks

% 5.1. Time-Domain Analysis: Plot the displacement at the excitation point
figure;
plot((0:Nf-1)*Ts, out, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Displacement');
title('Time-Domain Response at the Excitation Location');
grid on;

% Print summary statistics
fprintf('Output Statistics:\n');
fprintf('Min = %g\n', min(out));
fprintf('Max = %g\n', max(out));
fprintf('Mean = %g\n', mean(out));

% 5.2. Frequency-Domain Analysis: Compute and plot the FFT of the output
NFFT = 2^nextpow2(Nf);
Y = fft(out, NFFT)/Nf;
f_axis = Fs/2*linspace(0,1,NFFT/2+1);

figure;
plot(f_axis, 2*abs(Y(1:NFFT/2+1)), 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Frequency Spectrum of the Output');
grid on;

% 5.3. Audio Output: Normalize and write the output to a WAV file, then play it
audio_out = out / max(abs(out));   % Normalize to [-1,1]
audio_out = audio_out * 0.9;         % Scale for safety
audiowrite('plate_output.wav', audio_out, Fs);
sound(audio_out, Fs);
fprintf('Audio written to "plate_output.wav" and played.\n');
