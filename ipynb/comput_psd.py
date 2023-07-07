import math
import numpy as np
from obspy.core.inventory.inventory import Inventory


def trace_psd(tr, metadata,
              psd_periods=None,
              smooth_on_all_periods=False,
              period_smoothing_width_octaves=1.0,
              period_step_octaves=0.125,
              special_handling=None):
    """Calculate the power spectral density (PSD) of the given
    trace `tr`, and returns the values in dB at the given `psd_periods`.

    :param tr: ObsPy Trace
    :param metadata: Response information of instrument. It must be
        a :class:`~obspy.core.inventory.inventory.Inventory`
    :param psd_periods: numeric list/array of periods (in second) or None.
    :param smooth_on_all_periods: boolean (default: False).
    :param period_smoothing_width_octaves: float. Ignored if `psd_periods`
        is None.
    :param period_step_octaves: float (default=0.125). Ignored if `psd_periods`
        is None or `smooth_on_all_periods` is False.
    :param special_handling: sensor details
    """
    try:
        tr.data[tr.data.mask] = 0.0
    except AttributeError:
        pass

    ppsd_length = tr.stats.endtime - tr.stats.starttime  # float, seconds
    stats = tr.stats
    sampling_rate = stats.sampling_rate

    nfft = ppsd_length * sampling_rate
    nfft = nfft / 4.0
    nfft = int(math.pow(2, math.floor(math.log(nfft, 2))))
    nlap = int(0.75 * nfft)

    spec, _freq = psd(tr.data, nfft, sampling_rate, detrend=detrend_linear,
                      window=fft_taper, noverlap=nlap, sides='onesided',
                      scale_by_freq=True)

    spec = spec[1:]
    freq = _freq[1:]

    spec = spec[::-1]

    if special_handling == "ringlaser":
        spec /= metadata['sensitivity'] ** 2
    else:
        try:
            resp = _get_response(tr, metadata, nfft)
        except Exception as e:
            msg = ("Error getting response from provided metadata:\n"
                   "%s: %s\n"
                   "Skipping time segment(s).")
            msg = msg % (e.__class__.__name__, str(e))
            raise ValueError(msg)

        resp = resp[1:]
        resp = resp[::-1]
        respamp = np.absolute(resp * np.conjugate(resp))

        w = 2.0 * math.pi * freq
        w = w[::-1]
        if special_handling == "hydrophone":
            spec = spec / respamp
        else:
            spec = (w ** 2) * spec / respamp
    dtiny = np.finfo(0.0).tiny
    spec[spec < dtiny] = dtiny

    # go to dB
    spec = np.log10(spec)
    spec *= 10

    smoothed_psd = []
    _psd_periods = 1.0 / freq[::-1]

    if psd_periods is None:
        return spec, _psd_periods

    psd_periods = np.asarray(psd_periods)

    if smooth_on_all_periods:
        period_bin_centers = []
        period_limits = (_psd_periods[0], _psd_periods[-1])
        # calculate smoothed periods
        for periods_bins in \
                _setup_yield_period_binning(psd_periods,
                                            period_smoothing_width_octaves,
                                            period_step_octaves, period_limits):
            period_bin_left, period_bin_center, period_bin_right = periods_bins
            _spec_slice = spec[(period_bin_left <= _psd_periods) &
                               (_psd_periods <= period_bin_right)]
            smoothed_psd.append(_spec_slice.mean())
            period_bin_centers.append(period_bin_center)
        val = np.interp(
            np.log10(psd_periods),
            np.log10(period_bin_centers),
            smoothed_psd
        )
        val[psd_periods < period_bin_centers[0]] = np.nan
        val[psd_periods > period_bin_centers[-1]] = np.nan
    else:
        for period_bin_left, period_bin_right in \
                _yield_period_binning(psd_periods,
                                      period_smoothing_width_octaves):
            _spec_slice = spec[(period_bin_left <= _psd_periods) &
                               (_psd_periods <= period_bin_right)]
            smoothed_psd.append(_spec_slice.mean() if len(_spec_slice)
                                else np.nan)

        val = np.array(smoothed_psd)

    return val, psd_periods


###################
# PSD COMPUTATION #
###################


def psd(x, nfft=None, fs=None, detrend=None, window=None,
        noverlap=None, pad_to=None, sides=None, scale_by_freq=None):
    """
    Compute the power spectral density.
    """
    Pxx, freqs = _spectral_helper(x=x, y=None, NFFT=nfft, Fs=fs,
                                  detrend_func=detrend, window=window,
                                  noverlap=noverlap, pad_to=pad_to,
                                  sides=sides, scale_by_freq=scale_by_freq,
                                  mode='psd')

    if Pxx.ndim == 2:
        if Pxx.shape[1] > 1:
            Pxx = Pxx.mean(axis=1)
        else:
            Pxx = Pxx[:, 0]
    return Pxx.real, freqs


def _spectral_helper(x, y=None, NFFT=None, Fs=None, detrend_func=None,  # noqa
                     window=None, noverlap=None, pad_to=None,  # noqa
                     sides=None, scale_by_freq=None, mode=None):
    """
    Private helper implementing the common parts between the psd, csd
    (cross spectral density), spectrogram and complex, magnitude, angle, and
    phase spectra.
    """
    if y is None:
        same_data = True
    else:
        same_data = y is x

    if Fs is None:
        Fs = 2
    if noverlap is None:
        noverlap = 0
    if detrend_func is None:
        detrend_func = detrend_none
    if window is None:
        window = window_hanning

    # if NFFT is set to None use the whole signal
    if NFFT is None:
        NFFT = 256  # noqa

    if mode is None or mode == 'default':
        mode = 'psd'
    else:
        lst = ['default', 'psd', 'complex', 'magnitude', 'angle', 'phase']
        if mode not in lst:
            raise ValueError('mode "%s" not in %s' % (str(mode), str(lst)))

    if not same_data and mode != 'psd':
        raise ValueError("x and y must be equal if mode is not 'psd'")

    x = np.asarray(x)
    if not same_data:
        y = np.asarray(y)

    if sides is None or sides == 'default':
        if np.iscomplexobj(x):
            sides = 'twosided'
        else:
            sides = 'onesided'
    else:
        lst = ['default', 'onesided', 'twosided']
        if sides not in lst:
            raise ValueError('sides "%s" not in %s' % (str(sides), str(lst)))

    # zero pad x and y up to NFFT if they are shorter than NFFT
    if len(x) < NFFT:
        n = len(x)
        x = np.resize(x, NFFT)
        x[n:] = 0

    if not same_data and len(y) < NFFT:
        n = len(y)
        y = np.resize(y, NFFT)
        y[n:] = 0

    if pad_to is None:
        pad_to = NFFT

    if mode != 'psd':
        scale_by_freq = False
    elif scale_by_freq is None:
        scale_by_freq = True

    # For real x, ignore the negative frequencies unless told otherwise
    if sides == 'twosided':
        numFreqs = pad_to
        if pad_to % 2:
            freqcenter = (pad_to - 1)//2 + 1
        else:
            freqcenter = pad_to//2
        scaling_factor = 1.
    elif sides == 'onesided':
        if pad_to % 2:
            numFreqs = (pad_to + 1)//2
        else:
            numFreqs = pad_to//2 + 1
        scaling_factor = 2.

    if not np.iterable(window):
        window = window(np.ones(NFFT, x.dtype))
    if len(window) != NFFT:
        raise ValueError(
            "The window length must match the data's first dimension")

    result = stride_windows(x, NFFT, noverlap, axis=0)
    result = detrend(result, detrend_func, axis=0)
    result = result * window.reshape((-1, 1))
    result = np.fft.fft(result, n=pad_to, axis=0)[:numFreqs, :]
    freqs = np.fft.fftfreq(pad_to, 1/Fs)[:numFreqs]

    if not same_data:
        # if same_data is False, mode must be 'psd'
        resultY = stride_windows(y, NFFT, noverlap)
        resultY = detrend(resultY, detrend_func, axis=0)
        resultY = resultY * window.reshape((-1, 1))
        resultY = np.fft.fft(resultY, n=pad_to, axis=0)[:numFreqs, :]
        result = np.conj(result) * resultY
    elif mode == 'psd':
        result = np.conj(result) * result
    elif mode == 'magnitude':
        result = np.abs(result) / np.abs(window).sum()
    elif mode == 'angle' or mode == 'phase':
        # we unwrap the phase later to handle the onesided vs. twosided case
        result = np.angle(result)
    elif mode == 'complex':
        result /= np.abs(window).sum()

    if mode == 'psd':

        # if we have a even number of frequencies, don't scale NFFT/2
        if not NFFT % 2:
            slc = slice(1, -1, None)
        # if we have an odd number, just don't scale DC
        else:
            slc = slice(1, None, None)

        result[slc] *= scaling_factor

        if scale_by_freq:
            result /= Fs
            result /= (np.abs(window)**2).sum()
        else:
            result /= np.abs(window).sum()**2

    if sides == 'twosided':
        freqs = np.roll(freqs, -freqcenter, axis=0)
        result = np.roll(result, -freqcenter, axis=0)
    elif not pad_to % 2:
        freqs[-1] *= -1

    if mode == 'phase':
        result = np.unwrap(result, axis=0)

    return result, freqs


def stride_windows(x, n, noverlap=None, axis=0):
    """
    Get all windows of x with length n as a single array,
    using strides to avoid data duplication.
    """
    if noverlap is None:
        noverlap = 0

    if noverlap >= n:
        raise ValueError('noverlap must be less than n')
    if n < 1:
        raise ValueError('n cannot be less than 1')

    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError('only 1-dimensional arrays can be used')
    if n == 1 and noverlap == 0:
        if axis == 0:
            return x[np.newaxis]
        else:
            return x[np.newaxis].transpose()
    if n > x.size:
        raise ValueError('n cannot be greater than the length of x')

    noverlap = int(noverlap)
    n = int(n)

    step = n - noverlap
    if axis == 0:
        shape = (n, (x.shape[-1]-noverlap)//step)
        strides = (x.strides[0], step*x.strides[0])
    else:
        shape = ((x.shape[-1]-noverlap)//step, n)
        strides = (step*x.strides[0], x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


#############################
# PSD COMPUTATION (detrend) #
#############################


def detrend(x, key=None, axis=None):
    """
    Return x with its trend removed.
    """
    if key is None or key in ['constant', 'mean', 'default']:
        return detrend(x, key=detrend_mean, axis=axis)
    elif key == 'linear':
        return detrend(x, key=detrend_linear, axis=axis)
    elif key == 'none':
        return detrend(x, key=detrend_none, axis=axis)
    elif callable(key):
        x = np.asarray(x)
        if axis is not None and axis + 1 > x.ndim:
            raise ValueError(f'axis(={axis}) out of bounds')
        if (axis is None and x.ndim == 0) or (not axis and x.ndim == 1):
            return key(x)

        try:
            return key(x, axis=axis)
        except TypeError:
            return np.apply_along_axis(key, axis=axis, arr=x)
    else:
        raise ValueError(
            f"Unknown value for key: {key!r}, must be one of: 'default', "
            f"'constant', 'mean', 'linear', or a function")


def detrend_mean(x, axis=None):
    """
    Return x minus the mean(x).
    """
    x = np.asarray(x)

    if axis is not None and axis+1 > x.ndim:
        raise ValueError('axis(=%s) out of bounds' % axis)

    return x - x.mean(axis, keepdims=True)


def detrend_none(x, axis=None):
    """
    Return x: no detrending.
    """
    return x


def detrend_linear(y):
    """
    Return x minus best fit line; 'linear' detrending.
    """
    # This is faster than an algorithm based on linalg.lstsq.
    y = np.asarray(y)

    if y.ndim > 1:
        raise ValueError('y cannot have ndim > 1')

    # short-circuit 0-D array.
    if not y.ndim:
        return np.array(0., dtype=y.dtype)

    x = np.arange(y.size, dtype=float)

    C = np.cov(x, y, bias=1)
    b = C[0, 1]/C[0, 0]

    a = y.mean() - b*x.mean()
    return y - (b*x + a)


####################
# TAPERING WINDOWS #
####################


def fft_taper(data):
    """Cosine taper, 10 percent at each end
    """
    return data * cosine_taper(len(data), 0.2)


def cosine_taper(npts, p=0.1, freqs=None, flimit=None, halfcosine=True,
                 sactaper=False):
    """
    Cosine Taper. Copied from ObsPy to avoid importing unnecessary stuff from
    the invsim module (import in ObsPy can be quite slow)
    """
    if p < 0 or p > 1:
        msg = "Decimal taper percentage must be between 0 and 1."
        raise ValueError(msg)
    if p == 0.0 or p == 1.0:
        frac = int(npts * p / 2.0)
    else:
        frac = int(npts * p / 2.0 + 0.5)

    if freqs is not None and flimit is not None:
        fl1, fl2, fl3, fl4 = flimit
        idx1 = np.argmin(abs(freqs - fl1))
        idx2 = np.argmin(abs(freqs - fl2))
        idx3 = np.argmin(abs(freqs - fl3))
        idx4 = np.argmin(abs(freqs - fl4))
    else:
        idx1 = 0
        idx2 = frac - 1
        idx3 = npts - frac
        idx4 = npts - 1
    if sactaper:
        idx2 += 1
        idx3 -= 1

    if idx1 == idx2:
        idx2 += 1
    if idx3 == idx4:
        idx3 -= 1

    cos_win = np.zeros(npts)
    if halfcosine:
        cos_win[idx1:idx2 + 1] = 0.5 * (
            1.0 - np.cos((np.pi * (np.arange(idx1, idx2 + 1) - float(idx1)) /
                          (idx2 - idx1))))
        cos_win[idx2 + 1:idx3] = 1.0
        cos_win[idx3:idx4 + 1] = 0.5 * (
            1.0 + np.cos((np.pi * (float(idx3) - np.arange(idx3, idx4 + 1)) /
                          (idx4 - idx3))))
    else:
        cos_win[idx1:idx2 + 1] = np.cos(-(
            np.pi / 2.0 * (float(idx2) -
                           np.arange(idx1, idx2 + 1)) / (idx2 - idx1)))
        cos_win[idx2 + 1:idx3] = 1.0
        cos_win[idx3:idx4 + 1] = np.cos((
            np.pi / 2.0 * (float(idx3) -
                           np.arange(idx3, idx4 + 1)) / (idx4 - idx3)))

    if idx1 == idx2:
        cos_win[idx1] = 0.0
    if idx3 == idx4:
        cos_win[idx3] = 0.0
    return cos_win


def window_hanning(x):
    """
    Return x times the hanning window of len(x).
    """
    return np.hanning(len(x))*x

##########################
# RESPONSE-RELATED STUFF #
##########################


def _get_response(tr, metadata, nfft):
    """Return the response from the given trace and the given metadata
    """
    if isinstance(metadata, Inventory):
        return _get_response_from_inventory(tr, metadata, nfft)

    msg = "Unexpected type for `metadata`: %s" % type(metadata)
    raise TypeError(msg)


def _get_response_from_inventory(tr, metadata, nfft):
    """Alias of
    :meth:`~obspy.signal.spectral_estimation.PPSD._get_response_from_inventory`
    """
    inventory = metadata
    delta = 1.0 / tr.stats.sampling_rate
    id_ = "%(network)s.%(station)s.%(location)s.%(channel)s" % tr.stats
    response = inventory.get_response(id_, tr.stats.starttime)

    resp, _ = get_evalresp_response(response, t_samp=delta, nfft=nfft,
                                    output="VEL")
    return resp


def get_evalresp_response(response, t_samp, nfft, output="VEL",
                          start_stage=None, end_stage=None):
    """Alias of
    :meth:`~obspy.core.inventory.response.Response.get_evalresp_response`
    """
    fy = 1 / (t_samp * 2.0)
    freqs = np.linspace(0, fy, nfft // 2 + 1).astype(np.float64)

    response = get_evalresp_response_for_frequencies(response,
                                                     freqs, output=output,
                                                     start_stage=start_stage,
                                                     end_stage=end_stage)
    return response, freqs


def get_evalresp_response_for_frequencies(response, frequencies, output="VEL",
                                          start_stage=None, end_stage=None):
    """Alias of
    :meth:`~obspy.core.inventory.response.Response.get_evalresp_response_for_frequencies`
    """
    output, chan = response._call_eval_resp_for_frequencies(
        frequencies, output=output, start_stage=start_stage,
        end_stage=end_stage, hide_sensitivity_mismatch_warning=True)
    return output


#################################
# SMOOTHING WINDOWS COMPUTATION #
#################################


def _yield_period_binning(psd_periods, period_smoothing_width_octaves):

    period_smoothing_width_factor = \
        2 ** period_smoothing_width_octaves
    period_smoothing_width_factor_sqrt = \
        (period_smoothing_width_factor ** 0.5)
    for psd_period in psd_periods:
        per_left = (psd_period /
                    period_smoothing_width_factor_sqrt)
        per_right = per_left * period_smoothing_width_factor
        yield per_left, per_right


def _setup_yield_period_binning(psd_periods, period_smoothing_width_octaves,
                                period_step_octaves, period_limits):
    """
    Set up period binning
    """
    if period_limits is None:
        period_limits = (psd_periods[0], psd_periods[-1])
    period_step_factor = 2 ** period_step_octaves
    period_smoothing_width_factor = \
        2 ** period_smoothing_width_octaves

    per_left = (period_limits[0] /
                (period_smoothing_width_factor ** 0.5))
    per_right = per_left * period_smoothing_width_factor
    per_center = math.sqrt(per_left * per_right)

    previous_periods = per_left, per_center, per_right

    idx = np.argwhere(psd_periods > per_center)[0][0]
    psdlen = len(psd_periods)

    while per_center < period_limits[1] and idx < psdlen:
        per_left *= period_step_factor

        per_right = per_left * period_smoothing_width_factor

        per_center = math.sqrt(per_left * per_right)
        # yield if:
        if previous_periods[1] <= psd_periods[idx] and \
                per_center >= psd_periods[idx]:
            yield previous_periods
            yield per_left, per_center, per_right
            idx += 1

        previous_periods = per_left, per_center, per_right
