from collections.abc import Callable, Sequence
import numpy as np
from LLG import IFTDevice, PulseTimeStepSchedule, SOTDevice, Field, TangentTorqueMap
from LLG import sample_raw_tangent_torque_map, sample_tangent_torque_at_directions, sample_tangent_torque_map
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    moment_magnitude = getattr(device, "moment_magnitude_muB", None)
    if moment_magnitude is not None:
        moment_magnitude = float(moment_magnitude)
        torque_map = TangentTorqueMap(
            theta=torque_map.theta,
            phi=torque_map.phi,
            direction=torque_map.direction,
            cartesian=moment_magnitude*torque_map.cartesian,
            theta_component=moment_magnitude*torque_map.theta_component,
            phi_component=moment_magnitude*torque_map.phi_component,
        )
        colorbar_label = r"IFT efficiency of $\tau_{\theta}$ ($\mu_B$/fs)/(V/$\AA$)$^2$"
    else:
        colorbar_label = r"Drive contribution to $\dot{\theta}$ (fs$^{-1}$)"
    with open(f"{mainDir}/torqueDistr.txt","w") as fout:
        fout.write("             Phi           Theta         tau_phi       tau_theta\n")
        for itheta in range(theta_count):
            for iphi in range(phi_count):
                fout.write(f"{torque_map.phi[itheta,iphi]:16.8E}{torque_map.theta[itheta,iphi]:16.8E}"
                           f"{torque_map.phi_component[itheta,iphi]:16.8E}{torque_map.theta_component[itheta,iphi]:16.8E}\n")
    figure, axis = plt.subplots()
    arrows = axis.quiver(
        torque_map.phi,
        torque_map.theta,
        torque_map.phi_component,
        torque_map.theta_component,
        torque_map.theta_component,
        cmap="RdBu",
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
    colorMagnitude=max([abs(torque_map.theta_component).max(),abs(-torque_map.theta_component).max()])    
    arrows.set_clim(vmin=-colorMagnitude, vmax=colorMagnitude)
    figure.colorbar(arrows, ax=axis, label=colorbar_label)
    figure.tight_layout()
    plt.savefig(mainDir+"/torqueDistr.png",dpi=300)
    plt.close()
    return torque_map

def plot_raw_tangent_torque_map(mainDir, device, drive_amplitude=1.0, trajectory=None, arrow_stride=1):
    """
    Plot the raw torque and its pointwise comparison with the VSH fit.

    The vector origins are the saved Fibonacci sampling directions in
    ``AngleDependentDynamicsMap.npz``. No VSH evaluation or angular-grid
    interpolation is used for ``rawTorqueDistr.png``. The additional
    ``rawVSHComparison.png`` overlays black raw arrows and colored VSH arrows
    evaluated at exactly the same orientations. Its color is the local
    tangent-vector residual normalized by the global RMS raw torque.
    """
    arrow_stride = int(arrow_stride)
    if arrow_stride < 1:
        raise ValueError("arrow_stride must be at least 1.")
    torque_map = sample_raw_tangent_torque_map(device, drive_amplitude=drive_amplitude)
    moment_magnitude = float(device.moment_magnitude_muB)
    torque_map = TangentTorqueMap(
        theta=torque_map.theta,
        phi=torque_map.phi,
        direction=torque_map.direction,
        cartesian=moment_magnitude*torque_map.cartesian,
        theta_component=moment_magnitude*torque_map.theta_component,
        phi_component=moment_magnitude*torque_map.phi_component,
    )
    fitted_map = sample_tangent_torque_at_directions(
        device, torque_map.direction, drive_amplitude=drive_amplitude
    )
    fitted_map = TangentTorqueMap(
        theta=fitted_map.theta,
        phi=fitted_map.phi,
        direction=fitted_map.direction,
        cartesian=moment_magnitude*fitted_map.cartesian,
        theta_component=moment_magnitude*fitted_map.theta_component,
        phi_component=moment_magnitude*fitted_map.phi_component,
    )
    residual_phi = fitted_map.phi_component-torque_map.phi_component
    residual_theta = fitted_map.theta_component-torque_map.theta_component
    absolute_error = np.hypot(residual_phi, residual_theta)
    raw_rms = np.sqrt(np.mean(torque_map.phi_component**2+torque_map.theta_component**2))
    normalized_error = np.zeros_like(absolute_error) if raw_rms == 0.0 else 100.0*absolute_error/raw_rms

    with open(f"{mainDir}/rawTorqueDistr.txt", "w") as fout:
        fout.write("             Phi           Theta         tau_phi       tau_theta\n")
        for phi, theta, tau_phi, tau_theta in zip(
            torque_map.phi, torque_map.theta, torque_map.phi_component, torque_map.theta_component
        ):
            fout.write(f"{phi:16.8E}{theta:16.8E}{tau_phi:16.8E}{tau_theta:16.8E}\n")

    with open(f"{mainDir}/rawVSHComparison.txt", "w") as fout:
        fout.write(
            "             Phi           Theta     raw_tau_phi   raw_tau_theta"
            "     fit_tau_phi   fit_tau_theta     delta_phi   delta_theta"
            "       abs_error error_pct_raw_rms\n"
        )
        rows = zip(
            torque_map.phi, torque_map.theta, torque_map.phi_component, torque_map.theta_component,
            fitted_map.phi_component, fitted_map.theta_component, residual_phi, residual_theta,
            absolute_error, normalized_error,
        )
        for row in rows:
            fout.write("".join(f"{value:16.8E}" for value in row)+"\n")

    sample = slice(None, None, arrow_stride)
    figure, axis = plt.subplots()
    arrows = axis.quiver(
        torque_map.phi[sample],
        torque_map.theta[sample],
        torque_map.phi_component[sample],
        torque_map.theta_component[sample],
        torque_map.theta_component[sample],
        cmap="RdBu",
        pivot="mid",
        angles="xy",
        scale_units="xy",
    )
    if trajectory is not None:
        path = np.asarray(trajectory, dtype=float)
        if path.ndim != 2 or path.shape[1] != 3:
            raise ValueError("trajectory must have shape (ntime, 3).")
        path = path/np.linalg.norm(path, axis=1)[:, None]
        path_theta = np.arccos(np.clip(path[:, 2], -1.0, 1.0))
        path_phi = np.mod(np.arctan2(path[:, 1], path[:, 0]), 2.0*np.pi)
        axis.scatter(path_phi, path_theta, s=8, c="black", alpha=0.55, label="trajectory")
        axis.scatter(path_phi[0], path_theta[0], s=45, facecolors="white", edgecolors="black", label="start", zorder=3)
        axis.scatter(path_phi[-1], path_theta[-1], s=55, marker="x", c="red", label="end", zorder=3)
        axis.legend()
    axis.set_xlim(0.0, 2.0*np.pi)
    axis.set_ylim(np.pi, 0.0)
    axis.set_xticks(
        [0.0, 0.5*np.pi, np.pi, 1.5*np.pi, 2.0*np.pi],
        ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )
    axis.set_yticks([0.0, 0.5*np.pi, np.pi], ["0", r"$\pi/2$", r"$\pi$"])
    axis.set_xlabel(r"$\phi$")
    axis.set_ylabel(r"$\theta$")
    axis.set_title("Raw electronic IFT samples")
    color_magnitude = np.max(np.abs(torque_map.theta_component))
    if color_magnitude == 0.0: color_magnitude = 1.0
    arrows.set_clim(vmin=-color_magnitude, vmax=color_magnitude)
    figure.colorbar(
        arrows,
        ax=axis,
        label=r"IFT efficiency of $\tau_{\theta}$ ($\mu_B$/fs)/(V/$\AA$)$^2$",
    )
    figure.tight_layout()
    figure.savefig(f"{mainDir}/rawTorqueDistr.png", dpi=300)
    plt.close(figure)

    raw_magnitude = np.hypot(torque_map.phi_component[sample], torque_map.theta_component[sample])
    fitted_magnitude = np.hypot(fitted_map.phi_component[sample], fitted_map.theta_component[sample])
    combined_magnitude = np.concatenate((raw_magnitude, fitted_magnitude))
    reference_magnitude = np.percentile(combined_magnitude, 95.0)
    shared_scale = reference_magnitude/0.12 if reference_magnitude > 0.0 else 1.0

    figure, axis = plt.subplots()
    axis.quiver(
        torque_map.phi[sample], torque_map.theta[sample], torque_map.phi_component[sample],
        torque_map.theta_component[sample], color="green", pivot="mid", angles="xy", scale_units="xy",
        scale=shared_scale, width=0.002, alpha=1, zorder=1,
    )
    fitted_arrows = axis.quiver(
        torque_map.phi[sample], torque_map.theta[sample], fitted_map.phi_component[sample],
        fitted_map.theta_component[sample], absolute_error[sample], cmap="plasma", pivot="mid",
        angles="xy", scale_units="xy", scale=shared_scale, width=0.002, zorder=2,
    )
    color_magnitude = np.max(np.abs(absolute_error))
    if color_magnitude == 0.0: color_magnitude = 1.0
    fitted_arrows.set_clim(vmin=0,vmax=color_magnitude)
    axis.set_xlim(0.0, 2.0*np.pi)
    axis.set_ylim(np.pi, 0.0)
    axis.set_xticks(
        [0.0, 0.5*np.pi, np.pi, 1.5*np.pi, 2.0*np.pi],
        ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )
    axis.set_yticks([0.0, 0.5*np.pi, np.pi], ["0", r"$\pi/2$", r"$\pi$"])
    axis.set_xlabel(r"$\phi$")
    axis.set_ylabel(r"$\theta$")
    axis.set_title("Raw electronic IFT and VSH fit")
    axis.legend(
        handles=[
            Line2D([0], [0], color="green", linewidth=2.5, label="Raw electronic IFT"),
            Line2D([0], [0], color=plt.get_cmap("magma")(0.65), linewidth=2.5, label="VSH fit"),
        ],
        loc="best",
    )
    figure.colorbar(
        fitted_arrows, ax=axis,
        #label="Local VSH residual (% of global RMS raw torque)",
        label=r"Local VSH residual |$\tau$| ($\mu_B$/fs)/(V/$\AA$)$^2$",
    )
    figure.tight_layout()
    figure.savefig(f"{mainDir}/rawVSHComparison.png", dpi=600)
    plt.close(figure)
    return torque_map

def plot_mag_evolution(mainDir, time_axis:Sequence, trajectory:Sequence, pulse:Callable, pulse_title:str):
    with open(f"{mainDir}/magEvolution.txt","w") as fout:
        fout.write(f"        Time(fs)              mx              my              mz {pulse_title:20s}\n")
        for time, (mx,my,mz) in zip(time_axis,trajectory):
            fout.write(f"{time:16.8E}{mx:16.8E}{my:16.8E}{mz:16.8E} {pulse(time):20.8E}\n")
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(time_axis, trajectory[:, 0], label=r"m$_x$")
    axes[0].plot(time_axis, trajectory[:, 1], label=r"m$_y$")
    axes[0].plot(time_axis, trajectory[:, 2], label=r"m$_z$")
    axes[0].axhline(y= np.cos(np.radians(5)),c='black')
    axes[0].axhline(y=-np.cos(np.radians(5)),c='black')
    axes[0].set_ylabel("Magnetic orientation")
    axes[0].legend()
    axes[0].set_xlim(time_axis[0],time_axis[-1])
    axes[1].plot(time_axis, [pulse(time) for time in time_axis])
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

def step_pulse(t_pulse_on, t_pulse_off, efield = 1.0) -> Callable[[float], float]:
    """
    Return a step pulse, which is unit between t_pulse_on and t_pulse_off, otherwise zero
    """
    def evelope(time):
        return efield*float(t_pulse_on <= time < t_pulse_off)
    return evelope

def run_ift_example(mainDir, omega0, polarization="xy+",show_torque_map=True):
    """
    Plot one optical-pulse trajectory from a saved angular electronic map.

    The carrier oscillation is already integrated out of the rectified IFT.
    The supplied pulse is therefore the slowly varying electric-field
    envelope, and ``intensity_fwhm`` is its intensity FWHM.
    """
    center_times = [100.0 + 100_000.0 * ipulse for ipulse in range(20)]
    peak_field=0.06
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
        coarse_step=1500.0,
        window_half_width_in_fwhm=3.0,
    )

    #pulse = step_pulse(t_pulse_on = 0_000.0, t_pulse_off = 10_000_000.0, 
    #                   efield=5*1E-4)
    #time_step = 1500.0

    device = IFTDevice(
        data_directory=mainDir,
        omega0=omega0,
        polarization=polarization,
        torque_projection=0,
        light_pulse=pulse,
        external_field=np.array([0.0, 0.0, 0.0]),  # in Tesla
        use_DFT_mae=False,
        extra_uniaxial_mae=0.125E-3,               # in eV
        extra_uniaxial_mae_direction=(0, 0, 1), 
        alpha=0.001,
        theta0=5*np.pi/180,
        phi0=0*np.pi/180,
    )
    time_axis, trajectory = device.run(
        t_total=   104_000.0,
        #t_total=3_000_000.0,
        time_step=time_step,
    )
    plot_mag_evolution(mainDir=mainDir,time_axis=time_axis,trajectory=trajectory,pulse=pulse,pulse_title=r"|E(t)|[$(V/\AA)$]")
    if show_torque_map:
        plot_tangent_torque_map(mainDir=mainDir, device=device, trajectory=None)
        plot_raw_tangent_torque_map(mainDir=mainDir, device=device, trajectory=None)
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
    run_ift_example(mainDir="mcalc/MnBi2Te4",omega0=0.8,polarization="xy-",show_torque_map=True)
