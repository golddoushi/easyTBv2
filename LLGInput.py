from collections.abc import Callable, Sequence
import numpy as np
from LLG import IFTDevice, PulseTimeStepSchedule, SOTDevice, Field, sample_tangent_torque_map
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light, epsilon_0

def plot_tangent_torque_map(mainDir, device, theta_count=31, phi_count=61, drive_amplitude=1.0, trajectory=None):
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
        angles="xy",
        scale_units="xy",
    )
    if trajectory is not None:
        path = np.asarray(trajectory, dtype=float)
        if path.ndim != 2 or path.shape[1] != 3:
            raise ValueError("trajectory must have shape (ntime, 3).")
        path = path / np.linalg.norm(path, axis=1)[:, None]
        path_theta = np.arccos(np.clip(path[:, 2], -1.0, 1.0))
        path_phi = np.mod(np.arctan2(path[:, 1], path[:, 0]), 2.0 * np.pi)
        axis.scatter(path_phi, path_theta, s=8, c="black", alpha=0.55, label="trajectory",)
        axis.scatter(path_phi[0], path_theta[0], s=45, facecolors="white", edgecolors="black", label="start", zorder=3)
        axis.scatter(path_phi[-1], path_theta[-1], s=55, marker="x", c="red", label="end", zorder=3)
        axis.legend()
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
    plt.savefig(mainDir+"/torqueDistr.png",dpi=300)
    plt.close()
    return torque_map

def plot_mag_evolution(mainDir, time_axis:Sequence, trajectory:Sequence, pulse:Callable, pulse_title:str):
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(time_axis, trajectory[:, 0], label=r"m$_x$")
    axes[0].plot(time_axis, trajectory[:, 1], label=r"m$_y$")
    axes[0].plot(time_axis, trajectory[:, 2], label=r"m$_z$")
    axes[0].set_ylabel("Magnetic orientation")
    axes[0].legend()
    axes[0].set_xlim(time_axis[0],time_axis[-1])
    axes[1].plot(time_axis, [abs(pulse(time)) ** 2 for time in time_axis])
    axes[1].set_ylabel(pulse_title)
    axes[1].set_xlabel("Time [fs]")
    axes[1].set_xlim(time_axis[0],time_axis[-1])
    plt.savefig(mainDir+"/magEvolution.png",dpi=300)
    plt.close()

def gaussian_pulse(peak_field: float, center_times: Sequence[float], intensity_fwhm: float) -> Callable[[float], float]:
    """
    Return a Gaussian electric-field envelope in V/angstrom.

    ``intensity_fwhm`` is the FWHM of ``|E(t)|^2``.  Consequently, the field
    amplitude contains ``exp(-2 ln(2) (t-center)^2/FWHM^2)``.
    """
    peak_field = float(peak_field)
    intensity_fwhm = float(intensity_fwhm)
    if not np.isfinite(peak_field):
        raise ValueError("peak_field must be finite.")
    if not np.isfinite(intensity_fwhm) or intensity_fwhm <= 0.0:
        raise ValueError("intensity_fwhm must be finite and positive.")

    def envelope(time):
        field_tot=0.
        log2=np.log(2.0)
        for center in center_times:
            reduced_time = (time - center) / intensity_fwhm
            field_tot += peak_field * np.exp(-2.0 * log2 * reduced_time**2)
        return field_tot

    return envelope

def step_pulse(t_pulse_on, t_pulse_off) -> Callable[[float], float]:
    """
    Return a step pulse, which is unit between t_pulse_on and t_pulse_off, otherwise zero
    """
    def evelope(time):
        return float(t_pulse_on <= time < t_pulse_off)
    return evelope

def run_ift_example(data_directory="mcalc/MnBi2Te4", omega0=1.0, polarization="xy+",show_torque_map=True):
    """
    Plot one optical-pulse trajectory from a saved angular electronic map.

    The carrier oscillation is already integrated out of the rectified IFT.
    The supplied pulse is therefore the slowly varying electric-field
    envelope, and ``intensity_fwhm`` is its intensity FWHM.
    """
    center_times = [10_000.0 + 100_000.0 * ipulse for ipulse in range(40)]
    peak_field=0.1
    intensity_fwhm = 50.0
    intensity=0.5*speed_of_light*epsilon_0*(peak_field*1E10)**2
    fluence=intensity*intensity_fwhm*np.sqrt(np.pi/(4*np.log(2)))*1E-15
    print(f"Fluence of one pulse {fluence:.3f} J/m2 {fluence*0.1:.3f} mJ/cm2")
    
    pulse = gaussian_pulse(
        peak_field=peak_field,
        center_times=center_times,
        intensity_fwhm=intensity_fwhm,
    )
    time_step = PulseTimeStepSchedule.gaussian(
        center_times=center_times,
        intensity_fwhm=intensity_fwhm,
        fine_step=10.0,
        coarse_step=2500.0,
        window_half_width_in_fwhm=3.0,
    )
    device = IFTDevice(
        data_directory=data_directory,
        omega0=omega0,
        polarization=polarization,
        torque_projection=1,
        light_pulse=pulse,
        external_field=np.array([0.0, 0.0, 0.0]),  # in Tesla
        alpha=0.001,
        theta0=5*np.pi/180,
        phi0=0*np.pi/180,
    )
    time_axis, trajectory = device.run(
        t_total=8_000_000.0,
        time_step=time_step,
    )
    plot_mag_evolution(mainDir=data_directory,time_axis=time_axis,trajectory=trajectory,pulse=pulse,pulse_title=r"$|E(t)|^2$ [$(V/\AA)^2$]")
    if show_torque_map: plot_tangent_torque_map(mainDir=data_directory, device=device, trajectory=trajectory)
    plt.show()

def run_sot_example(mainDir="./",show_torque_map=True):
    pulse=step_pulse(
        t_pulse_on= 200_000.0,
        t_pulse_off=250_000.0,
    )
    t_total = 800_000.0

    device = SOTDevice(
        H_K=np.array([0, 0, 1]),
        H_ext=np.array([-0.3, 0.0, 0.0]),
        H_FL=0.10,
        H_DL=0.30,
        sigma=np.array([0, 1, 0]),
        current_pulse=pulse,
        alpha=0.1,
        theta0=5*np.pi/180,
        phi0=0.0*np.pi/180,
    )
    time_axis, trajectory = device.run(t_total=t_total, time_step=5000)

    plot_mag_evolution(mainDir=mainDir,time_axis=time_axis,trajectory=trajectory,pulse=pulse,pulse_title="Current density")
    if show_torque_map: plot_tangent_torque_map(device=device, trajectory=trajectory)
    plt.show()


if __name__ == "__main__":
    #run_sot_example()
    run_ift_example(show_torque_map=True)
