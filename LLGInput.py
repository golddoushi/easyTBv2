from collections.abc import Callable, Sequence
import numpy as np
from LLG import IFTDevice, SOTDevice, sample_tangent_torque_map
import matplotlib.pyplot as plt

def gaussian_pulse(peak_field: float, center_time: float, intensity_fwhm: float) -> Callable[[float], float]:
    """
    Return a Gaussian electric-field envelope in V/angstrom.

    ``intensity_fwhm`` is the FWHM of ``|E(t)|^2``.  Consequently, the field
    amplitude contains ``exp(-2 ln(2) (t-center)^2/FWHM^2)``.
    """
    peak_field = float(peak_field)
    center_time = float(center_time)
    intensity_fwhm = float(intensity_fwhm)
    if not np.isfinite(peak_field):
        raise ValueError("peak_field must be finite.")
    if not np.isfinite(center_time):
        raise ValueError("center_time must be finite.")
    if not np.isfinite(intensity_fwhm) or intensity_fwhm <= 0.0:
        raise ValueError("intensity_fwhm must be finite and positive.")

    def envelope(time):
        reduced_time = (time - center_time) / intensity_fwhm
        return peak_field * np.exp(-2.0 * np.log(2.0) * reduced_time**2)

    return envelope

def step_pulse(t_pulse_on, t_pulse_off) -> Callable[[float], float]:
    """
    Return a step pulse, which is unit between t_pulse_on and t_pulse_off, otherwise zero
    """
    def evelope(time):
        return float(t_pulse_on <= time < t_pulse_off)
    return evelope

def run_ift_example(
    data_directory="mcalc/MnBi2Te4",
    omega0=1.0,
    polarization="xy+",
    show_torque_map=True,
):
    """
    Plot one optical-pulse trajectory from a saved angular electronic map.

    The carrier oscillation is already integrated out of the rectified IFT.
    The supplied pulse is therefore the slowly varying electric-field
    envelope, and ``intensity_fwhm`` is its intensity FWHM.
    """
    pulse = gaussian_pulse(
        peak_field=0.01,
        center_time=100000.0,
        intensity_fwhm=80000.0,
    )
    device = IFTDevice(
        data_directory=data_directory,
        omega0=omega0,
        polarization=polarization,
        torque_projection=1,
        light_pulse=pulse,
        alpha=0.1,
        theta0=1*np.pi/180,
        phi0=0*np.pi/180,
    )
    time_axis, trajectory = device.run(t_total=500000.0, time_step=1000.0)

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(time_axis, trajectory[:, 0], label=r"m$_x$")
    axes[0].plot(time_axis, trajectory[:, 1], label=r"m$_y$")
    axes[0].plot(time_axis, trajectory[:, 2], label=r"m$_z$")
    axes[0].set_ylabel("Magnetic orientation")
    axes[0].legend()
    axes[1].plot(
        time_axis,
        [abs(pulse(time)) ** 2 for time in time_axis],
    )
    axes[1].set_ylabel(r"$|E(t)|^2$ [$(V/\AA)^2$]")
    axes[1].set_xlabel("Time [fs]")
    if show_torque_map:
        plot_tangent_torque_map(device)
    plt.show()


def plot_tangent_torque_map(
    device,
    theta_count=31,
    phi_count=61,
    drive_amplitude=1.0,
):
    """
    Plot the unit-drive source torque over magnetic orientation space.

    Arrow components are expressed in the local orthonormal tangent basis,
    with ``T_phi`` horizontal and ``T_theta`` vertical.  Color gives the
    Cartesian tangent-torque magnitude.
    """
    torque_map = sample_tangent_torque_map(
        device,
        theta_count=theta_count,
        phi_count=phi_count,
        drive_amplitude=drive_amplitude,
    )
    figure, axis = plt.subplots()
    arrows = axis.quiver(
        torque_map.phi,
        torque_map.theta,
        torque_map.phi_component,
        torque_map.theta_component,
        torque_map.magnitude,
        cmap="viridis",
        pivot="mid",
    )
    axis.set_xlim(0.0, 2.0 * np.pi)
    axis.set_ylim(np.pi, 0.0)
    axis.set_xticks(
        [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi],
        ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )
    axis.set_yticks(
        [0.0, 0.5 * np.pi, np.pi],
        ["0", r"$\pi/2$", r"$\pi$"],
    )
    axis.set_xlabel(r"$\phi$")
    axis.set_ylabel(r"$\theta$")
    figure.colorbar(arrows, ax=axis, label="Tangent torque magnitude")
    figure.tight_layout()
    return torque_map


def run_sot_example(show_torque_map=True):
    pulse=step_pulse(
        t_pulse_on= 50.0,
        t_pulse_off=150.0,
    )
    t_total = 200.0

    device = SOTDevice(
        H_K=np.array([0, 0, 1]),
        H_ext=np.array([-0.3, 0.0, 0.0]),
        H_FL=0.10,
        H_DL=0.30,
        sigma=np.array([0, 1, 0]),
        current_pulse=pulse,
        gamma=1.0,
        alpha=0.1,
        theta0=0.05,
        phi0=0.0,
    )
    time_axis, trajectory = device.run(t_total=t_total, time_step=0.1)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(time_axis, trajectory[:, 0], label=r"m$_x$")
    axes[0].plot(time_axis, trajectory[:, 1], label=r"m$_y$")
    axes[0].plot(time_axis, trajectory[:, 2], label=r"m$_z$")
    axes[0].set_ylabel("Magnetic orientation")
    axes[0].legend()
    axes[1].plot(time_axis, [pulse(time) for time in time_axis])
    axes[1].set_ylabel("Current density")
    axes[1].set_xlabel("Time")
    if show_torque_map:
        plot_tangent_torque_map(device)
    plt.show()


if __name__ == "__main__":
    #run_sot_example()
    run_ift_example(show_torque_map=True)
