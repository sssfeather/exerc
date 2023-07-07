import numpy as np
from scipy.fftpack import fft, next_fast_len
from obspy.geodetics.base import locations2degrees
from obspy.taup import TauPyModel


def rms(d):
    return np.sqrt(np.mean(d**2))


def get_snr(trace, pick, pre_wl=1e-3, post_wl=10e-3):
    """
    PARAMETERS:
    tr - Obspy trace
    Pick Time - in Obspy UTCDateTime
    preWl - Length of pre-window in seconds
    postWl - Length of post-window in seconds

    RETURNS:
    SNR - Signal to noise ratio
    """

    tr = trace

    sr = tr.stats.sampling_rate
    st = tr.stats.starttime
    et = tr.stats.endtime
    ps = int((pick - st) * sr)
    n_pre = int(pre_wl * sr)
    n_post = int(post_wl * sr)

    if pick + post_wl > et:
        energy_s = np.var(tr.data[ps:])
    else:
        energy_s = np.var(tr.data[ps:ps + n_post])

    if pick - pre_wl < st:
        energy_n = np.var(tr.data[:ps])
    else:
        energy_n = np.var(tr.data[ps - n_pre:ps])

    if (energy_n == 0) | (energy_s == 0):
        return 0

    snr = 10 * np.log10(energy_s / energy_n)

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
