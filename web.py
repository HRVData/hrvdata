#!/usr/bin/env python
# coding: utf-8

import argparse
import json

import numpy as np

from flask import Flask, render_template, request, Response
from scipy import interpolate

from P_welch import pwelch


app = Flask(__name__)


def psd(input_data):
    t = np.cumsum(input_data) / 1000.0
    t = t - t[0]
    tx = np.arange(t[0], t[-1], 1.0 / 4.0)
    temp = interpolate.splrep(t, input_data, s=0)
    rrix = interpolate.splev(tx, temp, der=0)
    rrix = rrix - np.mean(rrix)
    Fxx, Pxx = pwelch(rrix, 256, 128, 4.0)
    media = '{:.2f}'.format(float(np.mean(input_data)))
    sdnn = '{:.2f}'.format(float(np.std(input_data)))
    rmssd = '{:.2f}'.format(float(np.sqrt(sum((np.diff(input_data)) ** 2) /
                                  (len(input_data) - 1))))
    vlf, hf, lf = aucpsd(Fxx, Pxx)
    retorno = {'signal': zip(t, input_data),'data': zip(Fxx, Pxx), 'linhas': len(input_data),
               'indices': {'data-media': media, 'data-sdnn': sdnn, 'data-rmssd': rmssd, 'data-vlf':vlf,
               'data-lf':lf,'data-hf':hf}}
    return retorno

def aucpsd(fxx, pxx):
    df = fxx[1] - fxx[0]
    vlf = sum(pxx[np.logical_and(fxx >= 0, fxx < 0.04)]) * df
    lf = sum(pxx[np.logical_and(fxx >= 0.04, fxx < 0.15)]) * df
    hf = sum(pxx[np.logical_and(fxx >= 0.15, fxx < 0.4)]) * df

    vlf = '{:.2f}'.format(vlf)
    lf =  '{:.2f}'.format(lf)
    hf =  '{:.2f}'.format(hf)
    return vlf, lf, hf


@app.route("/")
def hello(name=None):
    return render_template('index.html')


@app.route("/process/", methods=['POST'])
def process_function():
    if 'userfile' not in request.files or not request.files['userfile'].filename:
        return Response(response=json.dumps({'erro': 'Bad request'}),
                        status=400,
                        content_type='application/json')

    fobj = request.files['userfile'].stream
    input_data = np.array([float(line.strip()) for line in fobj if line.strip()])
    retorno = psd(input_data)
    return Response(response=json.dumps(retorno),
                    status=200,
                    content_type='application/json')

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--http-host', default='127.0.0.1')
    args.add_argument('--http-port', default=5000, type=int)
    args.add_argument('--debug', action='store_true',
                      help='Habilita debugging no servidor Web')
    args.add_argument('--processes', type=int, default=16)
    argv = args.parse_args()

    app.config["DEBUG"] = argv.debug
    app.run(host=argv.http_host, port=argv.http_port, processes=argv.processes)
