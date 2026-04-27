import pytest
import torch
import math
from ioptics.nonlinear import Abs, AbsSquared, SquareActivation, SPMActivation, ElectroOpticActivation, SigmoidLikeActivation, TanhLikeActivation

def test_abs_activation():
    act = Abs()
    x = torch.tensor([1.0 + 1j, 2.0 - 2j, 0.0 + 0j], dtype=torch.complex64)
    expected = torch.tensor([math.sqrt(2), math.sqrt(8), 0.0], dtype=torch.float32)
    torch.testing.assert_close(act(x), expected)

def test_abs_squared_activation():
    act = AbsSquared()
    x = torch.tensor([1.0 + 1j, 2.0 - 2j, 0.0 + 0j], dtype=torch.complex64)
    expected = torch.tensor([2.0, 8.0, 0.0], dtype=torch.float32)
    torch.testing.assert_close(act(x), expected)

def test_square_activation():
    act = SquareActivation()
    x = torch.tensor([1.0 + 1j, 2.0 - 2j, 0.0 + 0j], dtype=torch.complex64)
    # (1+i)^2 = 2i, (2-2i)^2 = -8i
    expected = torch.tensor([0 + 2j, 0 - 8j, 0 + 0j], dtype=torch.complex64)
    torch.testing.assert_close(act(x), expected)

def test_spm_activation():
    gain = 0.5
    act = SPMActivation(gain=gain)
    x = torch.tensor([1.0 + 0j, 2.0 + 0j], dtype=torch.complex64)

    res = act(x)

    for i in range(len(x)):
        z = x[i]
        abs_sq = torch.abs(z)**2
        expected = z * torch.exp(-1j * gain * abs_sq)
        torch.testing.assert_close(res[i], expected)

def test_eo_direct_correctness():
    g = 0.5
    phi_b = 1.0
    alpha = 0.1
    act = ElectroOpticActivation(g=g, phi_b=phi_b, alpha=alpha)
    x = torch.tensor([1.0 + 0j], dtype=torch.complex64)

    res = act(x)

    abs_sq = torch.abs(x[0])**2
    arg = 0.5 * (g * abs_sq + phi_b)
    expected = 1j * math.sqrt(1 - alpha) * torch.exp(-1j * arg) * torch.cos(arg) * x[0]
    torch.testing.assert_close(res[0], expected)

def test_eo_physical_equivalence():
    # Compare physical params with explicit g, phi_b
    alpha = 0.1
    resp = 0.8
    area = 1.0
    V_pi = 10.0
    V_bias = 10.0
    R = 1e3
    imp = 120 * math.pi
    act_phys = ElectroOpticActivation(
        alpha=alpha, responsivity=resp, area=area, V_pi=V_pi, V_bias=V_bias, R=R, impedance=imp
    )

    g_derived = math.pi * alpha * R * resp * area * 1e-12 / 2 / V_pi / imp
    phi_b_derived = math.pi * V_bias / V_pi
    act_dir = ElectroOpticActivation(g=g_derived, phi_b=phi_b_derived, alpha=alpha)
    x = torch.tensor([1.0 + 1j, 0.5 + 0.2j], dtype=torch.complex64)
    torch.testing.assert_close(act_phys(x), act_dir(x))

def test_nonlinear_complex128():
    x = torch.tensor([1.0 + 1j], dtype=torch.complex128)

    # SPM
    act_spm = SPMActivation(gain=0.5)
    act_eo = ElectroOpticActivation()
    res_spm = act_spm(x)
    res_eo = act_eo(x)
    assert res_spm.dtype == torch.complex128
    assert res_eo.dtype == torch.complex128

def test_eo_validation():
    # Paired parameters validation
    with pytest.raises(ValueError, match="g and phi_b must be provided together"):
        ElectroOpticActivation(g=0.5)

    with pytest.raises(ValueError, match="g and phi_b must be provided together"):
        ElectroOpticActivation(phi_b=1.0)

    # Alpha validation
    with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\)"):
        ElectroOpticActivation(alpha=1.0)

    with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\)"):
        ElectroOpticActivation(alpha=-0.1)

    # Physical params validation
    with pytest.raises(ValueError, match="V_pi/impedance must be positive"):
        ElectroOpticActivation(V_pi=0)

def test_spm_validation():
    with pytest.raises(ValueError, match="gain must be a finite real number"):
        SPMActivation(gain=float('inf'))

def test_sigmoid_like_deterministic():
    # Deterministic check when noise_std=0
    midpoint = 0.5
    steepness = 10.0
    act = SigmoidLikeActivation(midpoint=midpoint, steepness=steepness, noise_std=0)

    # Use complex inputs to verify complex output and power-domain behavior
    x = torch.tensor([0.1 + 0.1j, 0.5 + 0.5j, 1.0 + 0j], dtype=torch.complex64)

    p = torch.abs(x)**2
    p_act = torch.sigmoid(torch.tensor(steepness, dtype=torch.float32) * (p - midpoint))
    eps = 1e-12
    expected = x * torch.sqrt(p_act / (p + eps))

    torch.testing.assert_close(act(x), expected)

def test_sigmoid_like_phase_preservation():
    # Verify that the phase of complex input is preserved
    act = SigmoidLikeActivation(noise_std=0)
    x = torch.tensor([1.0 + 1j, -0.5 + 0.2j, 0.0 + 1j], dtype=torch.complex64)

    res = act(x)

    # Phase is preserved if angle(res) == angle(x)
    # We use torch.angle to compare phases
    torch.testing.assert_close(torch.angle(res), torch.angle(x))

def test_sigmoid_like_noise():
    # Output noise affects result when noise_std > 0
    torch.manual_seed(42)
    midpoint = 0.5
    steepness = 10.0
    noise_std = 0.1
    act = SigmoidLikeActivation(midpoint=midpoint, steepness=steepness, noise_std=noise_std)
    x = torch.tensor([0.5 + 0j], dtype=torch.complex64)

    res1 = act(x)
    res2 = act(x)

    # With noise, two calls on same input should be different
    assert not torch.allclose(res1, res2)

def test_sigmoid_like_reproducibility():
    # Seeded noise reproducibility check
    midpoint = 0.5
    steepness = 10.0
    noise_std = 0.1
    act = SigmoidLikeActivation(midpoint=midpoint, steepness=steepness, noise_std=noise_std)
    x = torch.tensor([0.5 + 0j], dtype=torch.complex64)

    torch.manual_seed(42)
    res1 = act(x)

    torch.manual_seed(42)
    res2 = act(x)

    torch.testing.assert_close(res1, res2)

def test_sigmoid_like_complex128():
    # Dtype behavior with complex128 input
    x = torch.tensor([1.0 + 1j], dtype=torch.complex128)
    act = SigmoidLikeActivation()
    res = act(x)
    assert res.dtype == torch.complex128

def test_sigmoid_like_validation():
    # Parameter validation checks
    with pytest.raises(ValueError, match="steepness must be a finite real number"):
        SigmoidLikeActivation(steepness=float('inf'))

    with pytest.raises(ValueError, match="noise_std must be non-negative"):
        SigmoidLikeActivation(noise_std=-0.1)


def test_tanh_like_calibration_points():
    # Default gain is chosen so that at P=1 => P_out≈0.8
    act = TanhLikeActivation()

    # Use real-positive fields so P = |x|^2 is exact: x=1 -> P=1, x=2 -> P=4
    x = torch.tensor([1.0 + 0j, 2.0 + 0j], dtype=torch.complex64)
    y = act(x)
    p_out = torch.abs(y) ** 2

    # Calibration targets
    assert torch.isclose(p_out[0], torch.tensor(0.8, dtype=p_out.dtype), atol=5e-3)
    assert p_out[1] > 0.99
    assert p_out[1] < 1.0


def test_tanh_like_saturating_increment():
    # As input power increases, output increments should shrink (saturation)
    act = TanhLikeActivation()
    p = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)
    x = torch.sqrt(p).to(torch.complex64)

    y = act(x)
    p_out = torch.abs(y) ** 2

    inc_12 = p_out[1] - p_out[0]
    inc_24 = p_out[2] - p_out[1]
    assert inc_24 < inc_12


def test_tanh_like_phase_preservation():
    act = TanhLikeActivation()
    x = torch.tensor([1.0 + 1j, -0.5 + 0.2j, 0.0 + 1j], dtype=torch.complex64)
    y = act(x)

    torch.testing.assert_close(torch.angle(y), torch.angle(x))


def test_tanh_like_validation():
    with pytest.raises(ValueError, match="gain must be a finite positive real number"):
        TanhLikeActivation(gain=float('inf'))

    with pytest.raises(ValueError, match="gain must be a finite positive real number"):
        TanhLikeActivation(gain=0.0)
