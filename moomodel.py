import math
import yaml, os, argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from skimage.color import hsv2rgb
import cv2
from collections import deque
from scipy.special import softmax


class Mooveemodel:
    def __init__(self, x_init, y_init, mu_s, sigma_speed, sigma_angular_velocity, theta_speed, theta_angular_velocity, border=None, side=None):
        self.mu = np.array([mu_s,0.])
        self.theta = np.array([theta_speed,theta_angular_velocity])
        self.sigma = np.array([sigma_speed,sigma_angular_velocity])
        self.v = np.array(self.mu)
        self.dt = np.ones(2)
        self.rng = np.random.default_rng()
        self.pos = np.array([x_init,y_init])
        self.angle = 0.
        self.os = np.array(self.mu)
        self.s = 0
        self.updateSpeed()
        if border:
            self.border = border
            if side:
                self.side = side
            if not side:
                print(f'You need to provide argument \'side\' for the border conditions {borderMethod}')

    def updateSpeed(self):
        os1 = self.os
        mu1 = self.mu
        theta1 = self.theta
        dt1 = self.dt
        sigma1 = self.sigma
        rng1 = self.rng

        self.os = (os1
            + theta1 * (mu1 - os1) * dt1
            + sigma1 * rng1.normal(0,np.sqrt(dt1),2)
        )

        self.angle = self.angle + self.os[1] * dt1[1]
        #self.s = np.log1p(np.exp(self.os[0])) #softplus cause it to get stuck in 0.
        self.s = abs(self.os[0])
        self.v[0] = self.s*np.cos(self.angle)
        self.v[1] = self.s*np.sin(self.angle)

        return self.v

    def updatePosition(self):
        new_pos = self.pos + self.v * self.dt
        if not self.border:
            return self.pos
        elif self.border=='periodic':
            self.pos = new_pos % self.side
            is_same_panel = True if np.all(new_pos == self.pos) else False
            return self.pos, is_same_panel


    def getDirection(self):
        return np.degrees(np.arctan2(self.v[1],self.v[0]))
