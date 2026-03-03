# -*- coding: utf-8 -*-
from math import sin, cos, radians, pi, degrees, asin, acos, exp, log
import os
import math
import numpy as np
from osgeo import gdal
import glob
import time
from scipy.linalg import expm,logm
import torch
import torch.nn as nn
import datetime
import multiprocessing
from scipy.ndimage import zoom

def calculate(i, j):


        # Boltzmann constant
        BoltzmannConstant = 0.0000000567

        Cp = 1014

        DDSR = (b26 - b22)

        DDLR = (b28 - b24)

        DLST_ta = DTa * FTa / FLST
        DLST_ta[np.where(DLST_ta == None)] = 0
        DLST_ta[np.isnan(DLST_ta)] = 0

        DLST_DSR = DDSR * FDSR / FLST
        DLST_DSR[np.where(DLST_DSR == None)] = 0
        DLST_DSR[np.isnan(DLST_DSR)] = 0

        DLST_DLR = DDLR * FDLR / FLST
        DLST_DLR[np.where(DLST_DLR == None)] = 0
        DLST_DLR[np.isnan(DLST_DLR)] = 0

        # 计算偏导数,FLST为加过负号的，直接比就可以
        FLST = BoltzmannConstant * b8 * 4 * b18 * b18 * b18 + (1 + (1 / b16)) * airdensity * Cp / ra / (
                1 - β) * MOD_QC
        FDSR = (1 - b15) * MOD_QC
        FDSR[np.where(FDSR == None)] = 0
        FDSR[np.isnan(FDSR)] = 0

        air_emiss = 0.92*0.00001*b6*b6
        air_emiss =np.where(air_emiss >= 1, 1, air_emiss)

        FDLR = (b8 +(b24**(-0.75))*0.25*(1+1/b16)*airdensity * Cp/(1 - β)/ra/BoltzmannConstant**0.25/air_emiss**0.25
                ) * MOD_QC
        FTa = (1 + (1 / b16)) * airdensity * Cp / ra / (1 - β) * MOD_QC
        FTa[np.where(FTa == None)] = 0
        FTa[np.isnan(FTa)] = 0

        FBB = (b18 - b36) * airdensity * Cp / ra / (1 - β) / b16 / b16
        FB = FBB / FLST * MOD_QC
        FB[np.where(FB == None)] = 0
        FB[np.isnan(FB)] = 0

        FwindSpeed = -(b18 - b36) * (1 + (1 / b16)) * airdensity * Cp / ra / (1 - β) / b7
        Fwind = FwindSpeed / FLST * MOD_QC
        Fwind[np.where(Fwind == None)] = 0
        Fwind[np.isnan(Fwind)] = 0

        Fraa = (1 + (1 / b16)) * airdensity * Cp * (b18 - b36) / (1 - β) / ra / ra
        Fra = Fraa / FLST * MOD_QC
        Fra[np.where(Fra == None)] = 0
        Fra[np.isnan(Fra)] = 0


        DSR_contribuction =DDSR * FDSR / FLST
        DSR_contribuction = np.where(DSR_contribuction > 100, 0, DSR_contribuction)
        DSR_contribuction = np.where(DSR_contribuction< -100, 0, DSR_contribuction)
        DSR_contribuction[np.where(DSR_contribuction == None)] = 0
        DSR_contribuction[np.isnan(DSR_contribuction)] = 0

        DLR_contribuction =DDLR * FDLR / FLST
        DLR_contribuction = np.where(DLR_contribuction > 100, 0, DLR_contribuction)
        DLR_contribuction = np.where(DLR_contribuction < -100, 0, DLR_contribuction)
        DLR_contribuction[np.where(DLR_contribuction == None)] = 0
        DLR_contribuction[np.isnan(DLR_contribuction)] = 0

    return


