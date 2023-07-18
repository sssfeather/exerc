import numpy as np

from obspy import Trace
import matplotlib.pyplot as plt
from scipy import fftpack
import numpy.fft as fftlib
from scipy import signal, fftpack
from scipy.signal import get_window
from scipy.fftpack import rfft, irfft


AMP_LIMIT = 1e-5  


def plot_fft(x: np.array, y: np.array, strict_length: bool = True,
             reconstruct: bool = True, log_y: bool = True,
             log_x: bool = True, plot_phase: bool = False,
             fft_len: int = None):
    N = len(y)
    if strict_length:
        fft_len = N
    elif not fft_len:
        fft_len = fftpack.next_fast_len(N)
    dt = x[1] - x[0]
    yf = fftlib.fft(y, n=fft_len)
    xf = np.fft.fftfreq(fft_len, dt)[:fft_len//2]
    yr = fftlib.ifft(yf)
    amplitude_spectrum = np.abs(yf[:fft_len//2])
    phase_spectrum = np.angle(yf[:fft_len//2], deg=True)
    amplitude_spectrum *= 2./fft_len
    nrows = 2
    if reconstruct:
        nrows += 1
    if plot_phase:
        nrows += 1
    ts_row, amp_row, phase_row, rs_row = 0, 1, -2, -1
    fig, ax = plt.subplots(nrows=nrows)
    ax[ts_row].plot(x, y, label="Time series")
    ax[ts_row].set_xlabel("Time (s)")
    ax[ts_row].set_ylabel("Amplitude")
    ax[ts_row].autoscale(enable=True, axis='both', tight=True)
    ax[ts_row].legend()

    if log_y and log_x:
        ax[amp_row].loglog(xf, amplitude_spectrum, label="Amplitude spectra")
    elif log_x:
        ax[amp_row].semilogx(xf, amplitude_spectrum, label="Amplitude spectra")
    else:
        ax[amp_row].plot(xf, amplitude_spectrum, label="Amplitude spectra")
    ax[amp_row].set_xlabel("Frequency (Hz)")
    ax[amp_row].set_ylabel("Normalised \namplitude")
    ax[amp_row].autoscale(enable=True, axis='both', tight=True)
    ax[amp_row].legend()
    ax[amp_row].grid("on")

    if plot_phase:
        ax[amp_row].get_shared_x_axes().join(ax[amp_row], ax[phase_row])
        if log_x:
            ax[phase_row].semilogx(xf, phase_spectrum, label="Phase spectra")
        else:
            ax[phase_row].plot(xf, phase_spectrum, label="Phase spectra")
        ax[phase_row].set_xlabel("Frequency (Hz)")
        ax[phase_row].set_ylabel("Phase \n(degrees)")
        ax[phase_row].autoscale(enable=True, axis='x', tight=True)
        ax[phase_row].set_ylim(-180, 180)
        ax[phase_row].legend()
        ax[phase_row].grid("on")

    if reconstruct:
        ax[ts_row].get_shared_x_axes().join(ax[ts_row], ax[rs_row])
        # Plot the reconstructed time-series
        ax[rs_row].plot(x, np.real(yr)[0:len(x)],
                        label="Reconstructed Time-series")
        ax[ts_row].set_xlabel("Time (s)")
        ax[rs_row].set_ylabel("Amplitude")
        ax[rs_row].autoscale(enable=True, axis='both', tight=True)
        ax[rs_row].legend()

    return fig


def resample_and_plot(tr: Trace, sampling_rate: float):
    factor = tr.stats.sampling_rate / float(sampling_rate)
    tr_out = tr.copy()
    data_in = tr.data
    max_time = tr.stats.npts * tr.stats.delta
    dt = tr.stats.delta
    N = tr.stats.npts
    x = rfft(tr_out.data.newbyteorder("="))
    x = np.insert(x, 1, x.dtype.type(0))
    if tr_out.stats.npts % 2 == 0:
        x = np.append(x, [0])
    x_r = x[::2]
    x_i = x[1::2]
    large_w = np.fft.ifftshift(
        get_window('hann', tr_out.stats.npts))
    x_r *= large_w[:tr_out.stats.npts // 2 + 1]
    x_i *= large_w[:tr_out.stats.npts // 2 + 1]
    num = int(tr_out.stats.npts / factor)
    df = 1.0 / (tr_out.stats.npts * tr_out.stats.delta)
    d_large_f = 1.0 / num * sampling_rate
    f = df * np.arange(0, tr_out.stats.npts // 2 + 1, dtype=np.int32)
    n_large_f = num // 2 + 1
    large_f = d_large_f * np.arange(0, n_large_f, dtype=np.int32)
    large_y = np.zeros((2 * n_large_f))
    large_y[::2] = np.interp(large_f, f, x_r)
    large_y[1::2] = np.interp(large_f, f, x_i)

    large_y = np.delete(large_y, 1)
    if num % 2 == 0:
        large_y = np.delete(large_y, -1)
    tr_out.data = irfft(large_y) * (float(num) / float(tr_out.stats.npts))
    tr_out.stats.sampling_rate = sampling_rate

    # 绘图
    fig, axes = plt.subplots(ncols=2)
    axes[0].plot(np.arange(0, max_time, dt), data_in)
    axes[1].semilogx(
        np.linspace(0.0, 1.0 / (2. * dt), int(N / 2)),
        2./N * np.abs(x[:N//2]), label="Original")
    axes[1].semilogx(
        np.linspace(dt, dt + 1.0 / (2. * tr_out.stats.delta), num // 2),
        2./N * np.abs(large_y[:num//2]), label="Resampled")
    axes[0].plot(np.arange(0, max_time, 1 / sampling_rate), tr_out.data)
    axes[1].legend()
    axes[0].set_xlabel("Time (s)")
    axes[1].set_xlabel("Frequency (Hz)")
    return tr_out, fig


def filter_and_plot(data, dt, filt):

    N = len(data)
    filt_time = fftpack.ifft(filt)
    x_time = np.arange(0, N * dt, dt)

    fft_len = fftpack.next_fast_len(N)
    data_freq = fftpack.fft(data, n=fft_len)
    filtered_freq = data_freq * filt
    filtered = fftpack.ifft(filtered_freq)

    x_freq = np.linspace(0.0, 1.0 / (2. * dt), int(N / 2))

    fig, axes = plt.subplots(nrows=2, ncols=3)

    axes[0][0].plot(x_time, data)
    axes[0][0].set_title("Input data")
    axes[0][1].plot(np.arange(0, len(filt_time) * dt, dt), np.real(filt_time))
    axes[0][1].set_title("Filter")
    axes[0][2].plot(x_time, np.real(filtered))
    axes[0][2].set_title("Filtered")

    axes[1][0].semilogx(x_freq, 2./N * np.abs(data_freq[:N//2]))
    axes[1][1].semilogx(x_freq, 2./N * np.abs(filt[:N//2]))
    axes[1][2].semilogx(x_freq, 2./N * np.abs(filtered_freq[:N//2]))

    for ax in axes[0]:
        ax.set_xlabel("Time (s)")
        ax.autoscale(enable=True, axis='both', tight=True)
    for ax in axes[1]:
        ax.set_xlabel("Frequency (Hz)")
        ax.autoscale(enable=True, axis='both', tight=True)
    return fig

