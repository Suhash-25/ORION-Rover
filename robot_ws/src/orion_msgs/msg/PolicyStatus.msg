# Status of the active controller in orion_policy.
uint8 MODE_IDLE=0
uint8 MODE_BASELINE=1
uint8 MODE_POLICY=2
uint8 MODE_PROBE=3
uint8 MODE_ESCAPE=4

std_msgs/Header header
uint8 mode                   # MODE_*
float32 entrapment_risk      # [0, 1], critic output rho (NaN when no critic is loaded)
float32[8] latent            # adaptation module output z-hat (zeros for the baseline)
string policy_sha            # short sha256 of the loaded ONNX policy, empty for the baseline
string obs_spec_hash         # hash of learning/orion_learning/obs_spec.py layout
bool critic_gating           # true when entrapment_risk can switch the rover to ESCAPE
bool escapes_exhausted       # true from the give-up (max_escape_attempts escapes failed, mode
                             # IDLE, zero command) until the next SetMode; the trial runner
                             # then ends the trial ENTRAPPED (SPEC section 8)
