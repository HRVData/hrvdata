#-------------------------------------------------------------------------------
# Name:        pwelch
    # Purpose:     Estimates the Modified Periodogram (Welch Peridogram) of a Signal
    #
    # Author:      Rhenan Bartels Ferreira
    #
    # Created:     25/03/2013
    # Copyright:   (c) Rhenan 2013
    # Licence:     <your licence>
    #-----------------------------------------------------------------------------
    #!/usr/bin/env python

import numpy as np
import scipy.stats
import math


def pwelch(data, D=256, overlap=0, fs=1.0, nfft=None, side='onesided',
           confint = False, alpha = 0.05):
    """
    Parameters
    ----------
    data: array_like, 1-D
        Signal.
    D: interger
        Segment size.
    overlap: interger
        overlap of adjancent segments.
   fs: integer
       sampling frequency
   nfft: integer
       Size of Zero-Padding
   side: str or None, optional
       Must be 'onesided' or 'twosided'. This determinates the length of the
       outputs. If 'onesided' then 'f' and 'Pxx' goes from 0 to fs / 2 + 1,
       else, 'f' and 'Pxx' goes from 0 to fs -1.

   Returns
   ------
   f: array_like, 1-D
       frequency content.
   Pxx: array_like, 1_D
       Power Spectral Density
   ---------------------------------------------------------------------------

    If data x[0], x[1], ..., x[N - 1] of N samples is divided into P segments
    of D samples each, with a shift os S samples between adjacents segments
    (S<=D), then the maximum number of segments P is given by the integer part
    of: P = (N - D) / S + 1. The  periodogram estivative is defined by;

                  Pxx(p) = (1 / (U * D * T)) * abs(X(f))**2

    Where p is the range of segments O <= p <= P - 1 over the frequency
    -1/2*T <= f <= 1/2*T, where X(f) is the DFT of the pth segment.
    U is the discrete-time window energy and T is the sample interval.
    X(f) =  T * x[n] * np.exp(-1j * 2 * np.pi * f * n * T)
    T = 1 / fs
    U = T * np.sum(w[n]**2)

    """
    if not isinstance(data, np.ndarray):
        raise Exception("data must be a array_like")
    elif D != int(D) or D <= 0:
        raise Exception("D must be an integer greater than zero")
    elif overlap != int(overlap) or overlap < 0:
        raise Exception("overlap must be an interger greater than zero")
    elif fs != int(fs) or fs <= 0:
        raise Exception("fs must be an integer greater than zero")
    elif not isinstance(side, str):
        raise Exception("fs must be a string. 'one-sided' or 'twosided'")
    elif nfft is not None and (nfft != int(nfft) or nfft <= 0):
        raise ValueError("nfft must be None or an integer greater than zero")
    elif nfft is not None and (nfft <= D):
        raise Exception("nfft must be greater than D")
    elif overlap >= D:
        raise Exception("overlap must be smaller than D")
    if len(data) < D:
        raise Exception("D must be smaller than the length of data")

    start = 0
    end = D

    psd_len = D / 2.0 + 1

    S = D - overlap
    P = int((len(data) - D) / S) + 1
    H = np.hanning(D)
    U = np.dot(H.T, H) / D

    Sxx = 0
    if nfft is not None:
        zplen = nfft - D
        zp = np.zeros(zplen)
        U = (1.0 / nfft) * sum(np.hanning(nfft) ** 2)
    else:
        zp = []

    scale = int(P) * D * fs * U

    for p in xrange(P):
        data_temp = np.concatenate((data[start:end], zp), axis=0)
        data_dc_han = (data_temp - np.mean(data_temp)) *\
                       np.hanning(len(data_temp))
        data_dc2_han = data_dc_han - np.mean(data_dc_han)
        xf = np.fft.fft(data_dc_han)
        Sxx = Sxx + np.real(xf * np.conj(xf))
        start += S
        end += S

    if side == 'onesided':
        Pxx = 2 * (Sxx / scale)
        Pxx = Pxx[0:psd_len]
        f = np.linspace(0, fs / 2, psd_len)
    else:
        Pxx = np.sum(PSD, axis=0) / (P * U)
        f = np.linspace(0, fs, len(Pxx))
    if confint:
        Pxxlow, Pxxsup = welchconfint(P, alpha, Pxx)
        return f, Pxx, Pxxlow, Pxxsup

    return f, Pxx


def welchconfint(ddof, alpha, Pxx):

    alpha1 = alpha / 2
    alpha2 = 1 - (alpha / 2)

    #stand deviations to calculate
    sigma1 = math.sqrt(scipy.stats.chi2.ppf(alpha1, 1))
    sigma2 = math.sqrt(scipy.stats.chi2.ppf(alpha2, 1))

    #confidence intervals these sigmas represent:
    conf_int1 = scipy.stats.chi2.cdf(sigma1 ** 2, 1)
    conf_int2 = scipy.stats.chi2.cdf(sigma2 ** 2, 1)


    #degrees of freedom to calculate
    chi_squared1 =scipy.stats.chi2.ppf(conf_int1, ddof)
    chi_squared2 =scipy.stats.chi2.ppf(conf_int2, ddof)

    infBoundary = 2 * ddof * Pxx / chi_squared1
    upBoundary = 2 * ddof * Pxx / chi_squared2
    return infBoundary, upBoundary

    def chi2conf(alpha, k):

        v = 2 * k
        c = chiinv([1 - alpha / 2, alpha / 2])
        return  v / c




