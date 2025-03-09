#pragma once
#include <JuceHeader.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace project
{

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

    using namespace juce;
    using namespace hise;
    using namespace scriptnode;

    namespace FunctionsClasses {

        // PlateModel implements the 2D plate simulation using a 41x41 grid (with 2-cell halos).
        class PlateModel {
        public:
            PlateModel() : sampleRate(44100.0), dx(0.025f), dy(0.025f), Lx(1.0f), Ly(1.0f), halo(2),
                rho(7800.0f), H(0.005f), E(2e11f), nu(0.3f), sigma0(1.0f), sigma1(0.005f),
                stabilityThreshold(1e4f) // failsafe threshold
            {
            }

            // Call once during initialization to setup the simulation.
            void prepare(double sr) {
                sampleRate = sr;
                Ts = 1.0 / sampleRate;

                Nx = static_cast<int>(std::round(Lx / dx)) + 1;
                Ny = static_cast<int>(std::round(Ly / dy)) + 1;
                Nx_tot = Nx + 2 * halo;
                Ny_tot = Ny + 2 * halo;

                // Interior grid indices (0-indexed)
                i_start = halo;
                i_end = Ny_tot - halo - 1;
                j_start = halo;
                j_end = Nx_tot - halo - 1;

                // Excitation point at the center of the interior.
                exc_i = (i_start + i_end) / 2;
                exc_j = (j_start + j_end) / 2;

                float kappa2 = E * (H * H) / (12.0f * rho * (1.0f - nu * nu));
                S = (Ts * Ts * kappa2) / (dx * dx * dx * dx);

                // Time-stepping factors.
                A = 1.0f + sigma0 * Ts;
                B = sigma1 * Ts;

                // Factorized finite-difference coefficients.
                B1 = S * 4.0f;
                B2 = S * (-1.0f);
                B3 = S * 2.0f;
                B4 = (2.0f - 20.0f * S) / A;
                C1 = (4.0f * B) / A;
                C2 = -1.0f / A;

                // Allocate and initialize state arrays.
                size_t gridSize = static_cast<size_t>(Nx_tot * Ny_tot);
                u_nm1.assign(gridSize, 0.0f);
                u_n.assign(gridSize, 0.0f);
                u_np1.assign(gridSize, 0.0f);
            }

            // Update damping parameters and recalc coefficients (for real-time control)
            void updateDampingParameters(float newSigma0, float newSigma1) {
                sigma0 = newSigma0;
                sigma1 = newSigma1;
                A = 1.0f + sigma0 * Ts;
                B = sigma1 * Ts;
                // Only B4, C1 and C2 depend on the damping.
                B4 = (2.0f - 20.0f * S) / A;
                C1 = (4.0f * B) / A;
                C2 = -1.0f / A;
            }

            // Update stiffness (Young's modulus) parameters and recalc coefficients.
            // Physically, increasing E (stiffness) raises the plate's resonant frequencies,
            // yielding a brighter, more percussive sound.
            void updateStiffnessParameters(float newE) {
                E = newE;
                float kappa2 = E * (H * H) / (12.0f * rho * (1.0f - nu * nu));
                S = (Ts * Ts * kappa2) / (dx * dx * dx * dx);
                // Recalculate coefficients that depend on S.
                B1 = S * 4.0f;
                B2 = S * (-1.0f);
                B3 = S * 2.0f;
                B4 = (2.0f - 20.0f * S) / A;
            }

            // Reset all state arrays to zero. This can be used as a failsafe.
            void resetState() {
                std::fill(u_nm1.begin(), u_nm1.end(), 0.0f);
                std::fill(u_n.begin(), u_n.end(), 0.0f);
                std::fill(u_np1.begin(), u_np1.end(), 0.0f);
            }

            // Process one simulation time step with an unrolled update loop.
            // The excitation value (mono) is injected at the plate center.
            // Returns the displacement at the excitation point.
            inline float processSample(float excitation) {
                const int width = Nx_tot;
                // Precompute constant offsets (using row-major ordering)
                const int offL = -1;               // left
                const int offR = 1;               // right
                const int offU = -width;           // up
                const int offD = width;           // down
                const int offLL = -2;               // two left
                const int offRR = 2;               // two right
                const int offUU = -2 * width;       // two up
                const int offDD = 2 * width;       // two down
                const int offUL = offU + offL;      // up-left
                const int offUR = offU + offR;      // up-right
                const int offDL = offD + offL;      // down-left
                const int offDR = offD + offR;      // down-right

                // Unrolled update loop: iterate over each interior row
                for (int i = i_start; i <= i_end; ++i) {
                    int rowStart = i * width;
                    int j = j_start;
                    // Process two grid points per iteration.
                    for (; j <= j_end - 1; j += 2) {
                        int cp0 = rowStart + j;
                        int cp1 = rowStart + j + 1;

                        // Update for grid point cp0.
                        float u_n_cp0 = u_n[cp0];
                        float u_nm1_cp0 = u_nm1[cp0];
                        float update0 =
                            B1 * (u_n[cp0 + offL] + u_n[cp0 + offR] + u_n[cp0 + offU] + u_n[cp0 + offD]) +
                            B2 * (u_n[cp0 + offLL] + u_n[cp0 + offRR] + u_n[cp0 + offUU] + u_n[cp0 + offDD]) +
                            B3 * (u_n[cp0 + offUL] + u_n[cp0 + offUR] + u_n[cp0 + offDL] + u_n[cp0 + offDR]) +
                            B4 * u_n_cp0 +
                            C1 * (u_nm1[cp0 + offL] + u_nm1[cp0 + offR] + u_nm1[cp0 + offU] + u_nm1[cp0 + offD]) +
                            C2 * u_nm1_cp0;
                        u_np1[cp0] = update0;

                        // Update for grid point cp1.
                        float u_n_cp1 = u_n[cp1];
                        float u_nm1_cp1 = u_nm1[cp1];
                        float update1 =
                            B1 * (u_n[cp1 + offL] + u_n[cp1 + offR] + u_n[cp1 + offU] + u_n[cp1 + offD]) +
                            B2 * (u_n[cp1 + offLL] + u_n[cp1 + offRR] + u_n[cp1 + offUU] + u_n[cp1 + offDD]) +
                            B3 * (u_n[cp1 + offUL] + u_n[cp1 + offUR] + u_n[cp1 + offDL] + u_n[cp1 + offDR]) +
                            B4 * u_n_cp1 +
                            C1 * (u_nm1[cp1 + offL] + u_nm1[cp1 + offR] + u_nm1[cp1 + offU] + u_nm1[cp1 + offD]) +
                            C2 * u_nm1_cp1;
                        u_np1[cp1] = update1;
                    }
                    // If there is one remaining column (odd width), process it.
                    if (j <= j_end) {
                        int cp = rowStart + j;
                        float u_n_cp = u_n[cp];
                        float u_nm1_cp = u_nm1[cp];
                        float update =
                            B1 * (u_n[cp + offL] + u_n[cp + offR] + u_n[cp + offU] + u_n[cp + offD]) +
                            B2 * (u_n[cp + offLL] + u_n[cp + offRR] + u_n[cp + offUU] + u_n[cp + offDD]) +
                            B3 * (u_n[cp + offUL] + u_n[cp + offUR] + u_n[cp + offDL] + u_n[cp + offDR]) +
                            B4 * u_n_cp +
                            C1 * (u_nm1[cp + offL] + u_nm1[cp + offR] + u_nm1[cp + offU] + u_nm1[cp + offD]) +
                            C2 * u_nm1_cp;
                        u_np1[cp] = update;
                    }
                }

                // Inject the excitation at the plate center.
                int excIndex = exc_i * width + exc_j;
                u_np1[excIndex] += excitation;

                // Enforce clamped boundary conditions by zeroing the halo regions.
                // Top and bottom halo rows.
                for (int i = 0; i < halo; ++i) {
                    int rowStart = i * width;
                    std::fill(u_np1.begin() + rowStart, u_np1.begin() + rowStart + width, 0.0f);
                }
                for (int i = Ny_tot - halo; i < Ny_tot; ++i) {
                    int rowStart = i * width;
                    std::fill(u_np1.begin() + rowStart, u_np1.begin() + rowStart + width, 0.0f);
                }
                // Left and right halo columns.
                for (int i = halo; i < Ny_tot - halo; ++i) {
                    int rowStart = i * width;
                    for (int j = 0; j < halo; ++j)
                        u_np1[rowStart + j] = 0.0f;
                    for (int j = width - halo; j < width; ++j)
                        u_np1[rowStart + j] = 0.0f;
                }

                // Capture the output displacement at the excitation point.
                float output = u_np1[excIndex];

                // Failsafe: If the output is non-finite or exceeds the stability threshold,
                // reset the entire state to recover from numerical instability.
                if (!std::isfinite(output) || std::abs(output) > stabilityThreshold) {
                    resetState();
                    output = 0.0f;
                }

                // Rotate state arrays for the next time step.
                u_nm1.swap(u_n);
                u_n.swap(u_np1);
                std::fill(u_np1.begin(), u_np1.end(), 0.0f);

                return output;
            }

        private:
            double sampleRate;
            float Ts;
            // Plate geometry.
            float Lx, Ly, dx, dy;
            int Nx, Ny;
            int halo;
            int Nx_tot, Ny_tot;
            int i_start, i_end, j_start, j_end;
            int exc_i, exc_j;

            // Physical parameters.
            float rho, H, E, nu;
            float sigma0, sigma1;

            // Finite-difference coefficients.
            float S, A, B;
            float B1, B2, B3, B4, C1, C2;

            // Failsafe stability threshold.
            float stabilityThreshold;

            // State arrays (2D grid stored in 1D).
            std::vector<float> u_nm1;
            std::vector<float> u_n;
            std::vector<float> u_np1;
        };

    } // namespace FunctionsClasses

    // pre C++20

    // You cannot change the way this node is templated.
    template <int NV>
    struct Simple_Plate : public data::base
    {
        SNEX_NODE(Simple_Plate);

        struct MetadataClass
        {
            SN_NODE_ID("Simple_Plate");
        };

        //==============================================================================
        // Node Properties 
        //==============================================================================
        static constexpr bool isModNode() { return false; }
        static constexpr bool isPolyphonic() { return NV > 1; }
        static constexpr bool hasTail() { return false; }
        static constexpr bool isSuspendedOnSilence() { return false; }
        static constexpr int getFixChannelAmount() { return 2; }

        static constexpr int NumTables = 0;
        static constexpr int NumSliderPacks = 0;
        static constexpr int NumAudioFiles = 0;
        static constexpr int NumFilters = 0;
        static constexpr int NumDisplayBuffers = 0;

        //==============================================================================
        // Audio Effect Class 
        // (Not used in this implementation since we use a single shared plate model)
        class AudioEffect
        {
        public:
            AudioEffect() {}
            void prepare(double sampleRate) {}
            inline float process(float excitation) { return excitation; }
            void updateParams(float newParam) {}
        };

        //==============================================================================
        // Main Processing Functions
        //==============================================================================
        // Initialization: prepare the plate model.
        void prepare(PrepareSpecs specs)
        {
            double sampleRate = specs.sampleRate;
            plate.prepare(sampleRate);
        }

        // Reset (called when the plugin is reloaded).
        void reset() {}

        // Process audio blocks. The input channels are summed to mono,
        // then each sample is used to update the plate simulation.
        // The same (mono) output is sent to both output channels.
        template <typename ProcessDataType>
        inline void process(ProcessDataType& data)
        {
            auto& fixData = data.template as<ProcessData<getFixChannelAmount()>>();
            auto audioBlock = fixData.toAudioBlock();

            const int numSamples = data.getNumSamples();
            auto* leftChannelData = audioBlock.getChannelPointer(0);
            auto* rightChannelData = audioBlock.getChannelPointer(1);

            for (int n = 0; n < numSamples; ++n)
            {
                // Sum channels and apply excitation gain.
                float excitation = (leftChannelData[n] + rightChannelData[n]) * excitationGain;
                // Process the plate simulation for this sample.
                float outSample = plate.processSample(excitation);
                // Apply output gain.
                float finalSample = outSample * outputGain;
                leftChannelData[n] = finalSample;
                rightChannelData[n] = finalSample;
            }
        }

        //==============================================================================
        // Parameter Handling
        //==============================================================================
        // Update parameters (using rigid syntax).
        template <int P>
        void setParameter(double v)
        {
            if (P == 0) {
                excitationGain = static_cast<float>(v);
            }
            else if (P == 1) {
                outputGain = static_cast<float>(v);
            }
            else if (P == 2) {
                // Damping0 parameter update.
                damping0 = static_cast<float>(v);
                plate.updateDampingParameters(damping0, damping1);
            }
            else if (P == 3) {
                // Damping1 parameter update.
                damping1 = static_cast<float>(v);
                plate.updateDampingParameters(damping0, damping1);
            }
            else if (P == 4) {
                // Stiffness parameter update.
                stiffness = static_cast<float>(v);
                plate.updateStiffnessParameters(stiffness);
            }
        }

        // Create GUI parameters.
        void createParameters(ParameterDataList& data)
        {
            {
                parameter::data p("ExcitationGain", { 0.0, 2.0, 0.001 });
                registerCallback<0>(p);
                p.setDefaultValue(1.0);
                data.add(std::move(p));
            }
            {
                // Increased output gain range.
                parameter::data p("OutputGain", { 0.0, 10.0, 0.001 });
                registerCallback<1>(p);
                p.setDefaultValue(1.0);
                data.add(std::move(p));
            }
            {
                parameter::data p("Damping0", { 0.0, 10.0, 0.001 });
                registerCallback<2>(p);
                p.setDefaultValue(1.0);
                data.add(std::move(p));
            }
            {
                parameter::data p("Damping1", { 0.0, 0.05, 0.0001 });
                registerCallback<3>(p);
                p.setDefaultValue(0.005);
                data.add(std::move(p));
            }
            {
                // Stiffness parameter (Young's modulus) range from 1e10 to 1e12.
                parameter::data p("Stiffness", { 1e10, 1e12, 1e9 });
                registerCallback<4>(p);
                p.setDefaultValue(2e11);
                data.add(std::move(p));
            }
        }

        //==============================================================================
        // External Data Handling
        //==============================================================================
        void setExternalData(const ExternalData& ed, int index)
        {
            // Not used in this implementation.
        }

        //==============================================================================
        // Event Handling (e.g. MIDI) 
        //==============================================================================
        void handleHiseEvent(HiseEvent& e)
        {
            // No MIDI handling needed for this model.
        }

        //==============================================================================
        // Frame Processing (not used for block processing)
        //==============================================================================
        template <typename FrameDataType>
        void processFrame(FrameDataType& data) {}

    private:
        // Single plate simulation instance.
        FunctionsClasses::PlateModel plate;
        // Parameter-controlled gains.
        float excitationGain{ 1.0f };
        float outputGain{ 1.0f };
        // Real-time damping parameters.
        float damping0{ 1.0f };
        float damping1{ 0.005f };
        // Real-time stiffness (Young's modulus) parameter.
        float stiffness{ 2e11f };
    };

} // namespace project
