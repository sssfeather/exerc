import numpy as np
from scipy.fftpack import fft, next_fast_len
from obspy.geodetics.base import locations2degrees
from obspy.taup import TauPyModel


def rms(d):
    return np.sqrt(np.mean(d**2))


def get_snr(d, t, dist, vmin, vmax, extend=0, offset=20, axis=1,
            getwindow=False, db=False, side="a", shorten_noise=False):
    """
    Get SNRs of the data with given distance, vmin, and vmax. The
    signal window will be computed using vmin and vmax. The noise
    window will be the same length as the signal window shifted
    toward the end with the given offset.

    d,t,dist,vmin,vmax: REQUIRED. data, time vector, distance, minimum
                        velocity, maximum velocity.
    extend: extend the window length from the computed window based on
            vmin and vmax. default is 20.
    offset: offset between noise and signal windows, in seconds. default is 20.
    axis: axis for the calculation. default 1.
    db: Decibel or not. Default is False.
    getwindow: return the indices of the signal and noise windows. only
                the start and end indices. Default False.
    side: negative (n) and/or positive (p) or both sides (a) for the
          given data (time vector). Default: "a"
    shorten_noise: force noise window to fit the data length after the signal
                   window. Default False.
            If True, the noise window will be smaller than the signal window.

    RETURNS
    snr: [negative, positive]
    [sig_idx_p,noise_idx_p],[sig_idx_n,noise_idx_n]: only return these windows
    when getwindow is True and side=="a".
    When side != "a" only returns the corresponding window indices.
    """
    d = np.array(d)
    # get window index:
    tmin = dist/vmax
    tmax = extend + dist/vmin
    dt = np.abs(t[1]-t[0])
    shift = int(offset/dt)
    if side.lower() == "a":
        halfn = int(len(t)/2) + 1
    else:
        halfn = 0
    sig_idx_p = [int(tmin/dt) + halfn, int(tmax/dt)+halfn]
    winlen = sig_idx_p[1] - sig_idx_p[0] + 1
    noise_idx_p = [sig_idx_p[0] + shift + winlen,
                   sig_idx_p[1] + shift + winlen]
    if noise_idx_p[1] > len(t) - 1:
        if shorten_noise:
            print("Noise window end [%d]is larger than the data length \
                  [%d]. Force it to stop at the end."
                  % (noise_idx_p[1], len(t)-1))
            noise_idx_p[1] = len(t) - 1
        else:
            raise ValueError("Noise window end [%d]is larger than the data \
                             length [%d]. Please adjust it."
                             % (noise_idx_p[1], len(t)-1))

    sig_idx_n = [len(t) - sig_idx_p[1], len(t) - sig_idx_p[0]]
    noise_idx_n = [len(t) - noise_idx_p[1], len(t) - noise_idx_p[0]]

    if d.ndim == 1:
        # axis is not used in this case
        if side.lower() == "a":
            snr_n = rms(np.abs(d[sig_idx_n[0]:sig_idx_n[1]+1])) / \
                rms(np.abs(d[noise_idx_n[0]:noise_idx_n[1]+1]))
            snr_p = rms(np.abs(d[sig_idx_p[0]:sig_idx_p[1]+1])) / \
                rms(np.abs(d[noise_idx_p[0]:noise_idx_p[1]+1]))
            snr = [snr_n**2, snr_p**2]
        elif side.lower() == "n":
            snr_n = rms(np.abs(d[sig_idx_n[0]:sig_idx_n[1]+1])) / \
                rms(np.abs(d[noise_idx_n[0]:noise_idx_n[1]+1]))
            snr = snr_n**2
        elif side.lower() == "p":
            snr_p = rms(np.abs(d[sig_idx_p[0]:sig_idx_p[1]+1])) / \
                rms(np.abs(d[noise_idx_p[0]:noise_idx_p[1]+1]))
            snr = snr_p**2
        else:
            raise ValueError(side+" is not supported. use one of: xcorr_sides")
    elif d.ndim == 2:
        if axis == 1:
            dim = 0
        else:
            dim = 1
        if side.lower() == "a":
            snr = np.ndarray((d.shape[dim], 2))
            for i in range(d.shape[dim]):
                snr_n = rms(np.abs(d[i, sig_idx_n[0]:sig_idx_n[1]+1])) / \
                    rms(np.abs(d[i, noise_idx_n[0]:noise_idx_n[1]+1]))
                snr_p = rms(np.abs(d[i, sig_idx_p[0]:sig_idx_p[1]+1])) / \
                    rms(np.abs(d[i, noise_idx_p[0]:noise_idx_p[1]+1]))
                snr[i, :] = [snr_n**2, snr_p**2]
        elif side.lower() == "n":
            snr = np.ndarray((d.shape[dim], 1))
            for i in range(d.shape[dim]):
                snr_n = rms(np.abs(d[i, sig_idx_n[0]:sig_idx_n[1]+1])) / \
                        rms(np.abs(d[i, noise_idx_n[0]:noise_idx_n[1]+1]))
                snr[i] = snr_n**2
        elif side.lower() == "p":
            snr = np.ndarray((d.shape[dim], 1))
            for i in range(d.shape[dim]):
                snr_p = rms(np.abs(d[i, sig_idx_p[0]:sig_idx_p[1]+1])) / \
                        rms(np.abs(d[i, noise_idx_p[0]:noise_idx_p[1]+1]))
                snr[i] = snr_p**2
        else:
            raise ValueError(side+" is not supported. use one of: xcorr_sides")
    else:
        raise ValueError("Only handles ndim <=2.")
        snr = None
    if db:
        snr = 10*np.log10(snr)
    if getwindow:
        if side.lower() == "a":
            return snr, [sig_idx_p, noise_idx_p], [sig_idx_n, noise_idx_n]
        elif side.lower() == "n":
            return snr, [sig_idx_n, noise_idx_n]
        elif side.lower() == "p":
            return snr, [sig_idx_p, noise_idx_p]
    else:
        return snr


def get_tt(event_lat, event_long, sta_lat, sta_long, depth_km,
           model="iasp91", type='first'):
    """
    Get the seismic phase arrival time of the specified earthquake
    at the station.
    """
    sta_t = locations2degrees(event_lat, event_long, sta_lat, sta_long)
    taup = TauPyModel(model=model)
    arrivals = taup.get_travel_times(source_depth_in_km=depth_km,
                                     distance_in_degree=sta_t)
    if type == 'first':
        tt = arrivals[0].time
        ph = arrivals[0].phase
    else:
        phase_found = False
        phaseall = []
        for i in range(len(arrivals)):
            phaseall.append(arrivals[i].phase.name)
            if arrivals[i].phase.name == type:
                tt = arrivals[i].time
                ph = type
                phase_found = True
                break
        if not phase_found:
            raise ValueError('phase <'+type+' > not found in '+str(phaseall))

    return tt, ph


def psd(d, s, axis=-1, db=False):
    """
    Compute power spectral density. The power spectrum is normalized by
    frequency resolution.

    PARAMETERS:
    d: numpy ndarray containing the data.
    s: sampling frequency (samples per second)
    axis: axis to computer PSD. default is the last dimension (-1).

    RETURNS:
    f: frequency array
    psd: power spectral density
    """
    if isinstance(d, list):
        d = np.array(d)
    if d.ndim > 2:
        print('data has >2 dimension. skip demean.')
    else:
        d = detrend(demean(d))
    if d.ndim == 1:
        axis = 0
    elif d.ndim == 2:
        axis = 1
    Nfft = int(next_fast_len(int(d.shape[axis])))
    Nfft2 = int(Nfft//2)
    ft = fft(d, Nfft, axis=axis)
    psd = np.square(np.abs(ft))/s
    f = np.linspace(0, s/2, Nfft2)
    if d.ndim == 1:
        psd = psd[:Nfft2]
    elif d.ndim == 2:
        psd = psd[:, :Nfft2]
    if db:
        psd = 10*np.log10(np.abs(psd))
    return f, psd


def detrend(data):
    '''
    this function removes the signal trend based on QR decomposion
    NOTE: QR is a lot faster than the least square inversion used by
    scipy (also in obspy).

    PARAMETERS:
    data: input data matrix
    RETURNS:
    data: data matrix with trend removed
    '''
    if data.ndim == 1:
        npts = data.shape[0]
        X = np.ones((npts, 2))
        X[:, 0] = np.arange(0, npts)/npts
        Q, R = np.linalg.qr(X)
        rq = np.dot(np.linalg.inv(R), Q.transpose())
        coeff = np.dot(rq, data)
        data = data-np.dot(X, coeff)
    elif data.ndim == 2:
        npts = data.shape[1]
        X = np.ones((npts, 2))
        X[:, 0] = np.arange(0, npts)/npts
        Q, R = np.linalg.qr(X)
        rq = np.dot(np.linalg.inv(R), Q.transpose())
        for ii in range(data.shape[0]):
            coeff = np.dot(rq, data[ii])
            data[ii] = data[ii] - np.dot(X, coeff)
    return data


def demean(data, axis=-1):
    '''
    this function remove the mean of the signal
    PARAMETERS:
    data: input data matrix
    axis: axis to operate.

    RETURNS:
    data: data matrix with mean removed
    '''
    if data.ndim == 1:
        data = data-np.mean(data)
    elif data.ndim == 2:
        m = np.mean(data, axis=axis)
        for ii in range(data.shape[0]):
            if axis == -1:
                data[ii] = data[ii]-m[ii]
            else:
                data[:, ii] = data[:, ii]-m[ii]

    return data


def remove_empty_trace(stream, date_info):
    """
    this function checks sampling rate and find gaps of all traces in stream.

    PARAMETERS:
    stream: obspy stream object.
    date_info: dict of starting and ending time of the stream

    RETURENS:
    stream: List of good traces in the stream
    """
    # remove empty/big traces
    if len(stream) == 0 or len(stream) > 100:
        stream = []
        return stream

    # remove traces with big gaps
    if portion_gaps(stream, date_info) > 0.3:
        stream = []
        return stream

    freqs = []
    for tr in stream:
        freqs.append(int(tr.stats.sampling_rate))
    freq = max(freqs)
    for tr in stream:
        if int(tr.stats.sampling_rate) != freq:
            stream.remove(tr)
        if tr.stats.npts < 10:
            stream.remove(tr)

    return stream


def portion_gaps(stream, date_info):
    '''
    this function tracks the gaps (npts) from the accumulated difference
    between starttime and endtime of each stream trace. it removes trace
    with gap length > 30% of trace size.

    PARAMETERS:
    stream: obspy stream object
    date_info: dict of starting and ending time of the stream

    RETURNS:
    pgaps: proportion of gaps/all_pts in stream
    '''

    starttime = date_info['starttime']
    endtime = date_info['endtime']
    npts = (endtime-starttime)*stream[0].stats.sampling_rate

    pgaps = 0
    # loop through all trace to accumulate gaps
    for ii in range(len(stream)-1):
        pgaps += (stream[ii+1].stats.starttime-stream[ii].stats.endtime) * \
                            stream[ii].stats.sampling_rate
    if npts != 0:
        pgaps = pgaps/npts
    if npts == 0:
        pgaps = 1
    return pgaps


def xcorr(x, y, maxlags=10):
    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')

    c = np.correlate(x, y, mode=2)

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maxlags must be None or strictly positive < %d' % Nx)

    c = c[Nx - 1 - maxlags:Nx + maxlags]

    return c
