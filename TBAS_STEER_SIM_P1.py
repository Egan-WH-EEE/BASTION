import sys

import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import os


from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSlider, QLabel, QFrame, QSizePolicy
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap, QIcon

import pygame, random, math


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BASTION")

        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
        self.setWindowIcon(QIcon(logo_path))

        self.view = QLabel()
        self.view.setMinimumSize(640, 384)
        self.view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # logo
        logo = QLabel()
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = QPixmap(logo_path)
        if not pix.isNull():
            logo.setPixmap(pix.scaledToWidth(200, Qt.TransformationMode.SmoothTransformation))
        else:
            logo.setText("BASTION")   # fallback if the file isn't found

        # controls
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(
            lambda on: self.pause_btn.setText("Resume" if on else "Pause"))

        wind_label = QLabel("WIND SPEED")
        wind_label.setObjectName("heading")
        self.wind_slider = QSlider(Qt.Orientation.Horizontal)
        self.wind_slider.setRange(0, 20)

        self.readout = QLabel()
        self.readout.setObjectName("readout")
        self.readout.setMinimumWidth(240)

        # side panel: logo first, then controls
        panel = QVBoxLayout()
        panel.setSpacing(10)
        panel.addWidget(logo)
        panel.addSpacing(8)
        panel.addWidget(self.pause_btn)
        panel.addWidget(wind_label)
        panel.addWidget(self.wind_slider)
        panel.addWidget(self.readout)
        panel.addStretch()

        side_panel = QWidget()
        side_panel.setLayout(panel)
        side_panel.setFixedWidth(280)      # panel width, adjust to taste

        layout = QHBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)
        layout.addWidget(self.view, 1)     # stretch 1: takes all spare space
        layout.addWidget(side_panel, 0)    # stretch 0: stays at its fixed width

        # Stripes across the bottom
        flag = QVBoxLayout()
        flag.setContentsMargins(0, 0, 0, 0)
        flag.setSpacing(0)
        for name, weight in (("flag_yellow", 2), ("flag_blue", 2), ("flag_red", 2)):
            stripe_bar = QFrame()
            stripe_bar.setObjectName(name)
            flag.addWidget(stripe_bar, weight)

        flag_holder = QWidget()
        flag_holder.setLayout(flag)
        flag_holder.setFixedHeight(48)     # total flag height

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addLayout(layout)
        outer.addWidget(flag_holder)

        central = QWidget()
        central.setLayout(outer)
        self.setCentralWidget(central)


# ---------------- WIND ----------------
k, c = 2.21, 7.72
turbulence_intensity = 0.05 # 5% turbulence intensity

base_wind_speed = c * math.gamma(1 + 1/k)
base_wind_speed_stdev = turbulence_intensity * base_wind_speed

base_wind_angle = 270 # deg, travelling towards 270 deg (west)
base_wind_angle_stdev = turbulence_intensity * base_wind_angle # deg

class Wind:
    def __init__(self, speed, speed_std, angle, angle_std, tau_speed=5.0, tau_angle=20.0):
        self.speed0, self.speed_std = speed, speed_std
        self.angle0, self.angle_std = angle, angle_std
        self.tau_s, self.tau_a = tau_speed, tau_angle
        self.ds = 0.0   # speed deviation
        self.da = 0.0   # angle deviation

    def step(self, dt):
        # discrete Gauss-Markov: stationary stdev equals the value you pass in
        self.ds += -self.ds / self.tau_s * dt + self.speed_std * math.sqrt(2 * dt / self.tau_s) * random.gauss(0, 1)
        self.da += -self.da / self.tau_a * dt + self.angle_std * math.sqrt(2 * dt / self.tau_a) * random.gauss(0, 1)
        return max(0.0, self.speed0 + self.ds), self.angle0 + self.da

wind = Wind(base_wind_speed, base_wind_speed_stdev, base_wind_angle, base_wind_angle_stdev)
# ---------------- CURRENT ----------------
base_current_speed = 1 # m/s
base_current_speed_stdev = 0.1 # m/s
base_current_angle = 0 # deg, travelling towards 0 deg (north)
base_current_angle_stdev = 5 # deg

# ------------- CONSTANTS --------------

# Pygame display constants
scale = 20            # pixels per metre
cx, cy = 600, 360     # screen centre (the camera follows the boat)

dt = 0.05 # s

mass = 80 # kg

e = 0.8

AR_sail = 5 # Aspect ratio of the sail, placeholder
AR_rudder = 2 # Aspect ratio of the rudder, placeholder
AR_keel = 3 # Aspect ratio of the keel, placeholder

stall_angle_sail = 20 # deg
stall_angle_rudder = 15 # deg
stall_angle_keel = 15 # deg

CD0_sail = 0.1 # zero-lift drag coefficient for sail, placeholder
CD0_rudder = 0.01 # zero-lift drag coefficient for rudder, placeholder
CD0_keel = 0.01 # zero-lift drag coefficient for keel, placeholder

CD_max_sail = 1.2 # max drag coefficient for sail, placeholder
CD_max_rudder = 1.0 # max drag coefficient for rudder, placeholder
CD_max_keel = 1.0 # max drag coefficient for keel, placeholder

air_density = 1.225 # kg/m^3
water_density = 1000 # kg/m^3

sail_area = 10 # m^2, placeholder
rudder_area = 0.2 # m^2, placeholder
keel_area = 0.2 # m^2, placeholder

# Saturation limits
min_sail, max_sail = -90, 90 # deg
min_rudder, max_rudder = -45, 45 # deg
max_sail_rotation_rate = 10 # deg/s
max_rudder_rotation_rate = 20 # deg/s

# Foil positions in body frame
sail_pos = (0, 0)
rudder_pos = [(-2, 1), (-2, -1)]
keel_pos = (0, 0)


trim_heading = 250

# ---------- LIVE VARIABLES ------------
class Simulation:
    def __init__(self):
        self.heading = 0
        self.yaw_rate = 0

        # Coordinates, m
        self.pos_x = 0
        self.pos_y = 0

        self.boat_velocity_x = 0
        self.boat_velocity_y = 0

        # Acceleration, m/s^2
        self.acceleration_x = 0
        self.acceleration_y = 0

        self.sail_angle = 0
        self.rudder_angle = 0

        self.net_force_x = 0
        self.net_force_y = 0

        self.moment = 0

        self.wind_speed = 0
        self.wind_angle = 0

    def update(self):
        update_sim()

sim = Simulation()

I_z = 1.3 * mass * (0.3 * 5)**2   # kg·m²
moment = 0.0


sail_angle_rate = 0 # deg/s
rudder_angle_rate = 0 # deg/s

# ----------- GENERAL -------------
# Map any angle (deg) into [-180, 180)
def wrap_angle(angle):
    return (angle + 180.0) % 360.0 - 180.0

# Rotate position according to heading change
def world_offset(px, py, heading_deg):
    h = math.radians(heading_deg)
    return px * math.cos(h) - py * math.sin(h), px * math.sin(h) + py * math.cos(h)

# Calculate relative flow at a foil given fluid velocity, boat velocity, and yaw rate
def foil_flow(fluid_x, fluid_y, rx, ry):
    return (fluid_x - sim.boat_velocity_x + sim.yaw_rate * ry, fluid_y - sim.boat_velocity_y - sim.yaw_rate * rx)

# Calculate force and moment for a foil
def foil_total(pos, fluid_x, fluid_y, density, foil_angle, area, AR, stall, CD0, CDmax):
    rx, ry = world_offset(pos[0], pos[1], sim.heading)
    fx, fy = foil_flow(fluid_x, fluid_y, rx, ry)
    lx, ly, dx, dy = calc_foil_forces(density, fx, fy, sim.heading, foil_angle, area, AR, stall, CD0, CDmax)
    Fx, Fy = lx + dx, ly + dy
    return Fx, Fy, rx * Fy - ry * Fx   # force x, force y, moment

# ----------- SYSTEM -------------
# Update globals based on current state
def update_system(): 
    # Calculate acceleration
    sim.acceleration_x = sim.net_force_x / mass
    sim.acceleration_y = sim.net_force_y / mass
    
    # Update velocity
    sim.boat_velocity_x += sim.acceleration_x * dt
    sim.boat_velocity_y += sim.acceleration_y * dt
    
    # Update position
    sim.pos_x += sim.boat_velocity_x * dt
    sim.pos_y += sim.boat_velocity_y * dt

    # Update heading and yaw rate based on rudder force (simplified model)
    sim.yaw_rate += (sim.moment / I_z) * dt
    sim.heading = wrap_angle(sim.heading + math.degrees(sim.yaw_rate) * dt)

# Give fluctuating wind speed/angle
def wind_fluctuate(base_speed, base_angle):
    wind_speed = base_speed + base_wind_speed_stdev * math.cos(math.radians(random.randint(0,180)))
    wind_angle = base_angle + base_wind_angle_stdev * math.cos(math.radians(random.randint(0,180)))
    return wind_speed, wind_angle

# Give fluctuating current speed/angle
def current_fluctuate(base_current_speed, base_current_angle):
    current_speed = base_current_speed + base_current_speed_stdev * math.cos(math.radians(random.randint(0,180)))
    current_angle = base_current_angle + base_current_angle_stdev * math.cos(math.radians(random.randint(0,180)))
    return current_speed, current_angle

# Calculate lift and drag coeffs
def calc_gains(angle_of_attack, AR, stall_angle, CD0, CD_max):
    global e
    if angle_of_attack > 90:
        angle_of_attack -= 180
    elif angle_of_attack < -90:
        angle_of_attack += 180
    aoa = abs(angle_of_attack)
    a = math.radians(aoa)

    # Pre and post stall lift coeff formulas
    if aoa < stall_angle:
        CL = 2 * math.pi * a / (1 + 2 / AR)
        CD = CD0 + (CL**2) / (math.pi * e * AR)
    else:
        CL = CD_max * math.sin(a) * math.cos(a)
        CD = CD_max * (math.sin(a)**2)
    return math.copysign(CL, angle_of_attack), CD  # Preserve the sign of the angle of attack

# Calculate lift and drag forces for a foil
def calc_foil_forces(fluid_density, flow_x, flow_y, heading, foil_angle, foil_area, AR, stall_angle, CD0, CD_max):
    flow_angle = math.degrees(math.atan2(flow_y, flow_x))
    angle_of_attack = wrap_angle(flow_angle - (heading + foil_angle))

    CL, CD = calc_gains(angle_of_attack, AR, stall_angle, CD0, CD_max)

    v_squared = flow_x**2 + flow_y**2
    lift_force = 0.5 * fluid_density * v_squared * foil_area * CL
    drag_force = 0.5 * fluid_density * v_squared * foil_area * CD

    lift_x, lift_y = lift_force * math.cos(math.radians(flow_angle + 90)), lift_force * math.sin(math.radians(flow_angle + 90))
    drag_x, drag_y = drag_force * math.cos(math.radians(flow_angle)), drag_force * math.sin(math.radians(flow_angle))

    return lift_x, lift_y, drag_x, drag_y

# ----------- PYGAME --------------

# Return (x, y) for a given centre point, angle (deg) and radius.
def point(angle, r, centrex, centrey):
    x = centrex + r * math.sin(math.radians(angle))
    y = centrey - r * math.cos(math.radians(angle))
    return (x, y)

# Map into screen coordinates from world coordinates
def world_to_screen(wx, wy):
    return cx + (wx - sim.pos_x) * scale, cy - (wy - sim.pos_y) * scale   # y flipped here only

# Draw a circle (for compass display)
def draw_arrow(colour, cx, cy, angle, length):
    bearing = 90 - angle
    end = point(bearing, length, cx, cy)
    pygame.draw.line(screen, colour, (cx, cy), end, 2)


# ----------- CONTROL -------------
# Operating point - LQR only works for a single operating point
# vx, vy, heading, r, sail_angle, rudder_angle


# Coefficients of state variables
def net_forces_and_moments(vx, vy, heading, r, sail_angle, rudder_angle, wind_speed=None, wind_angle=None):
    if wind_speed is None: wind_speed = base_wind_speed
    if wind_angle is None: wind_angle = base_wind_angle

    apparent_wind_speed_x = wind_speed * math.cos(math.radians(wind_angle)) - vx
    apparent_wind_speed_y = wind_speed * math.sin(math.radians(wind_angle)) - vy

    water_flow_x = base_current_speed * math.cos(math.radians(base_current_angle)) - vx
    water_flow_y = base_current_speed * math.sin(math.radians(base_current_angle)) - vy

    sail_lx, sail_ly, sail_dx, sail_dy = calc_foil_forces(air_density, apparent_wind_speed_x, apparent_wind_speed_y, heading, sail_angle, sail_area, AR_sail, stall_angle_sail, CD0_sail, CD_max_sail)
    keel_lx, keel_ly, keel_dx, keel_dy = calc_foil_forces(water_density, water_flow_x, water_flow_y, heading, 0, keel_area, AR_keel, stall_angle_keel, CD0_keel, CD_max_keel)

    # rudders: loop over both positions
    rud_fx = rud_fy = moment = 0
    for pos in rudder_pos:
        rx, ry = world_offset(pos[0], pos[1], heading)
        fx_flow = water_flow_x + r * ry
        fy_flow = water_flow_y - r * rx
        lx, ly, dx, dy = calc_foil_forces(water_density, fx_flow, fy_flow,
                                          heading, rudder_angle, rudder_area, AR_rudder,
                                          stall_angle_rudder, CD0_rudder, CD_max_rudder)
        fx, fy = lx + dx, ly + dy
        rud_fx += fx
        rud_fy += fy
        moment += rx * fy - ry * fx

    net_force_x = sail_lx + sail_dx + rud_fx + keel_lx + keel_dx
    net_force_y = sail_ly + sail_dy + rud_fy + keel_ly + keel_dy

    # hull drag (quadratic) on water-relative velocity, plus yaw damping
    yaw_damping = 50.0 # N·m·s/rad, placeholder
    rel_x, rel_y = water_flow_x, water_flow_y          # water velocity relative to boat
    rel_speed = math.hypot(rel_x, rel_y)
    hull_cd_area = 0.05                                 # m^2, placeholder
    hull_fx = 0.5 * water_density * hull_cd_area * rel_speed * rel_x
    hull_fy = 0.5 * water_density * hull_cd_area * rel_speed * rel_y
    net_force_x += hull_fx
    net_force_y += hull_fy
    moment += -yaw_damping * r  

    return net_force_x / mass, net_force_y / mass, moment / I_z

def jacobian(func, point, h=0.001):
    point = np.array(point, dtype=float)
    n_in = len(point)
    n_out = len(func(*point))
    J = np.zeros((n_out, n_in))
    for j in range(n_in):
        plus = point.copy();  plus[j] += h
        minus = point.copy(); minus[j] -= h
        J[:, j] = (np.array(func(*plus)) - np.array(func(*minus))) / (2 * h)
    return J

def residual(z):
    vx, vy, sail = z
    ax, ay, rdot = net_forces_and_moments(vx, vy, trim_heading, 0, sail, 0)
    return [ax, ay, rdot]

def residual_h(z, hdg):
    vx, vy, sail = z
    return list(net_forces_and_moments(vx, vy, hdg, 0, sail, 0))

for hdg in range(0, 360, 10):
    for guess in ([0.5, 0.5, 10], [-0.5, -0.5, 10], [1, -1, 15], [-1, 1, 15], [0, 0, 5]):
        sol, info, ier, msg = fsolve(residual_h, guess, args=(hdg,), full_output=True)
        res = np.linalg.norm(residual_h(sol, hdg))
        if ier == 1 and res < 1e-8 and abs(sol[2]) < 18:
            print("OK heading", hdg, np.round(sol, 3))
            break

vx0, vy0, sail0 = fsolve(residual, [0.5, 0.5, 15])
operating_point = [vx0, vy0, trim_heading, 0, sail0, 0]
print(operating_point, residual([vx0, vy0, sail0]))

x0 = np.array(operating_point, dtype=float)
x0[2] = math.radians(x0[2])
x0[4] = math.radians(x0[4])
x0[5] = math.radians(x0[5])

J = jacobian(net_forces_and_moments, operating_point)

A = np.zeros((6, 6))

# J rows: 0=ax, 1=ay, 2=rdot       -> A rows 0, 1, 3
# J cols: 0=vx, 1=vy, 2=heading, 3=r, 4=sail, 5=rudder -> A cols 0, 1, 2, 3, 4, 5
rows = [0, 1, 3]
cols = [0, 1, 2, 3, 4, 5]
for i in range(3):
    for j in range(6):
        A[rows[i], cols[j]] = J[i, j]

# per-degree -> per-radian for heading, sail, rudder columns
A[:, 2] *= 180 / math.pi
A[:, 4] *= 180 / math.pi
A[:, 5] *= 180 / math.pi

A[2, 3] = 1   # psi_dot = r
print(np.round(A, 3))
# A = np.array([
#     [dax/dvx, dax/dvy, dax/dpsi, dax/dr, dax/ddeltar, dax/ddeltas],
#     [day/dvx, day/dvy, day/dpsi, day/dr, day/ddeltar, day/ddeltas],
#     [0, 0, 0, 1, 0, 0],
#     [drdot/dvx, drdot/dvy, drdot/dpsi, drdot/dr, drdot/ddeltar, drdot/ddeltas],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0]
# ])

# X = np.array([boat_velocity_x, boat_velocity_y, heading, yaw_rate, sail_angle, rudder_angle])
# Xdot = np.array([net_force_x / mass, net_force_y / mass, yaw_rate, moment / I_z, sail_angle_rate, rudder_angle_rate])

# Coefficients of control inputs
B = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 1],
])

U = np.array([sail_angle_rate, rudder_angle_rate])

# State cost matrix
Q = np.diag([4, 4, 1000, 364.76, 8.21, 3.65])

 # Control cost matrix
R = np.diag([14.59, 3.65])

P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
print(np.linalg.eigvals(A - B @ K))


# ----------- SENSORS -------------
# Accelerometer, gyroscope, magnetometer, wind sensor, GPS
# Resolution, RMS noise, ODR, bias, ARW

class SensorSpec:
    name: str
    rms_noise: float # White noise, same units as measurement
    odr: float # Hz
    bias_init_std: float = 0.0 # Typical bias at t=0
    bias_walk: float = 0.0 # Random walk of bias (gyros only)
    resolution: float = 0.0 # Smallest measurable quantity

# SENSORS = {
#     SensorSpec("gyros", rms_noise=0, odr=100, bias_init_std=0.2, bias_walk=0.01),
#     SensorSpec("accel", rms_noise=0, odr=50,),
#     SensorSpec("compass", rms_noise=0),
#     SensorSpec("gps_x", rms_noise=0),
#     SensorSpec("gps_y"),
#     SensorSpec("wind")
# }

def read_sensor(true_value, res, rms_noise, bias, bias_walk):
    bias += random.gauss(0, bias_walk * math.sqrt(dt))
    sensor_value = round(true_value + random.gauss(0, rms_noise) / res) * res
    return (sensor_value + bias)

# ----------- MAIN FUNCS -------------
def draw_foil(body_pos, foil_angle, length, colour, width):
    ox, oy = world_offset(body_pos[0], body_pos[1], sim.heading)
    a = math.radians(sim.heading + foil_angle)
    start = (sim.pos_x + ox, sim.pos_y + oy)
    end = (start[0] - length * math.cos(a), start[1] - length * math.sin(a))
    pygame.draw.line(screen, colour, world_to_screen(*start), world_to_screen(*end), width)

def update_sim():

    x = np.array([
        sim.boat_velocity_x,
        sim.boat_velocity_y,
        math.radians(sim.heading),
        sim.yaw_rate,
        math.radians(sim.sail_angle),
        math.radians(sim.rudder_angle)
    ])

    dx = x - x0
    dx[2] = (dx[2] + math.pi) % (2 * math.pi) - math.pi

    u = -K @ dx

    sail_rate = max(
        min(math.degrees(u[0]), max_sail_rotation_rate),
        -max_sail_rotation_rate
    )

    rudder_rate = max(
        min(math.degrees(u[1]), max_rudder_rotation_rate),
        -max_rudder_rotation_rate
    )

    sim.sail_angle = max(
        min(sim.sail_angle + sail_rate * dt, max_sail),
        min_sail
    )

    sim.rudder_angle = max(
        min(sim.rudder_angle + rudder_rate * dt, max_rudder),
        min_rudder
    )

    sim.wind_speed, sim.wind_angle = wind.step(dt)

    ax, ay, rdot = net_forces_and_moments(
        sim.boat_velocity_x,
        sim.boat_velocity_y,
        sim.heading,
        sim.yaw_rate,
        sim.sail_angle,
        sim.rudder_angle,
        sim.wind_speed,
        sim.wind_angle
    )

    sim.net_force_x = ax * mass
    sim.net_force_y = ay * mass
    sim.moment = rdot * I_z

    update_system()

    global sim_t

    sim_t += dt

    log_t.append(sim_t)
    log_heading.append(sim.heading)
    log_yaw.append(math.degrees(sim.yaw_rate))
    log_sail.append(sim.sail_angle)
    log_rudder.append(sim.rudder_angle)

    trail.append((sim.pos_x, sim.pos_y))
    if len(trail) > 400:
        trail.pop(0)

def update_display():
    global trail

    # background
    screen.fill((20, 40, 70))

    # grid (the camera follows the boat, so the grid is what shows motion)
    step = 5
    gx0 = int(sim.pos_x // step) * step
    gy0 = int(sim.pos_y // step) * step
    for i in range(-8, 9):
        sx, _ = world_to_screen(gx0 + i * step, 0)
        pygame.draw.line(screen, (35, 60, 95), (sx, 0), (sx, 720))
    for j in range(-5, 6):
        _, sy = world_to_screen(0, gy0 + j * step)
        pygame.draw.line(screen, (35, 60, 95), (0, sy), (1200, sy))

    # trail
    if len(trail) > 1:
        pygame.draw.lines(screen, (255, 255, 0), False,
                        [world_to_screen(x, y) for x, y in trail], 2)

    # hull
    hull_body = [(3, 0), (0.5, 1.2), (-2.5, 1.2), (-2.5, -1.2), (0.5, -1.2)]
    hull_pts = []
    for px, py in hull_body:
        ox, oy = world_offset(px, py, sim.heading)
        hull_pts.append(world_to_screen(sim.pos_x + ox, sim.pos_y + oy))
    pygame.draw.polygon(screen, (200, 200, 200), hull_pts)

    # sail and rudders
    draw_foil(sail_pos, sim.sail_angle, 3.0, (255, 255, 255), 4)
    for rp in rudder_pos:
        draw_foil(rp, sim.rudder_angle, 0.8, (255, 120, 80), 3)

    # wind (white) and current (blue) arrows
    draw_arrow((255, 255, 255), 80, 80, sim.wind_angle, 40)
    draw_arrow((80, 200, 255), 80, 160, base_current_angle, 40)

    # heading dial: blue = target, red = actual
    pygame.draw.circle(screen, (30, 34, 44), (hd_cx, hd_cy), dial_r)
    pygame.draw.circle(screen, (100, 100, 100), (hd_cx, hd_cy), dial_r, 3)
    draw_arrow((0, 0, 255), hd_cx, hd_cy, desired_heading, dial_r)
    draw_arrow((255, 0, 0), hd_cx, hd_cy, sim.heading, dial_r)

    # rudder dial
    pygame.draw.circle(screen, (30, 34, 44), (rd_cx, rd_cy), dial_r)
    pygame.draw.circle(screen, (100, 100, 100), (rd_cx, rd_cy), dial_r, 3)
    pygame.draw.line(screen, (0, 255, 0), (rd_cx, rd_cy),
                    point(-sim.rudder_angle, dial_r, rd_cx, rd_cy), 3)

    # Wind dial
    pygame.draw.circle(screen, (30, 34, 44), (rd_cx + 900, rd_cy - 200), dial_r)
    pygame.draw.circle(screen, (100, 100, 100), (rd_cx + 900, rd_cy - 200), dial_r, 3)
    pygame.draw.line(screen, (0, 255, 0), (rd_cx + 900, rd_cy - 200),
                    point(sim.wind_angle, dial_r, rd_cx + 900, rd_cy - 200), 3)


    # # labels
    # screen.blit(font.render(f"Heading: {heading:.2f}", True, (255, 255, 255)), (40, 270))
    # screen.blit(font.render(f"Rudder: {rudder_angle:.2f}", True, (255, 255, 255)), (40, 520))
    # screen.blit(font.render(f"Sail: {sail_angle:.2f}", True, (255, 255, 255)), (40, 550))


def qt_update():
    if not window.pause_btn.isChecked():
        update_sim()

    update_display()

    data = pygame.image.tostring(screen, "RGBA")
    image = QImage(data, 1200, 720, 1200 * 4, QImage.Format.Format_RGBA8888)
    pix = QPixmap.fromImage(image).scaled(
        window.view.size(),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    window.view.setPixmap(pix)

    window.readout.setText(
        f"t: {sim_t:.1f} s\n"
        f"Heading: {sim.heading:.1f}°\n"
        f"Speed: {math.hypot(sim.boat_velocity_x, sim.boat_velocity_y):.2f} m/s\n"
        f"Sail: {sim.sail_angle:.1f}°\n"
        f"Rudder: {sim.rudder_angle:.1f}°\n"
        f"Wind: {sim.wind_speed:.1f} m/s"
    )

# ---------- THEME ----------
# Swap these hex values for your team colours
PALETTE = {
    "bg":        "#05024E",   # window background
    "panel":     "#1426cd",   # cards / rectangles
    "border":    "#2a4ac3",
    "text":      "#ffffff",
    "primary":   "#1f6fcb",   # buttons
    "accent":    "#36dcb0",   # slider fill, readout text
    "highlight": "#ffffff",   # headings, slider handle
    "alert":     "#e55c4b",   # paused button
}

STYLE_TEMPLATE = """
QMainWindow, QWidget {
    background-color: @bg;
    color: @text;
    font-family: "Segoe UI", "Helvetica Neue", Arial;
    font-size: 14px;
}
QFrame#card {
    background-color: @panel;
    border: 1px solid @border;
    border-radius: 8px;
}
QFrame#stripe {
    background-color: @highlight;
    border: none;
    border-radius: 3px;
}
QLabel { background: transparent; }
QLabel#readout {
    background-color: @panel;
    border: 1px solid @border;
    border-radius: 8px;
    padding: 12px;
    color: @accent;
    font-family: Menlo, Consolas, monospace;
    font-size: 15px;
}
QLabel#heading {
    color: @highlight;
    font-weight: bold;
    font-size: 12px;
    letter-spacing: 2px;
}
QPushButton {
    background-color: @primary;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 14px;
    font-weight: bold;
}
QPushButton:hover { background-color: @accent; color: @bg; }
QPushButton:checked { background-color: @alert; }
QSlider::groove:horizontal { height: 6px; background: @border; border-radius: 3px; }
QSlider::sub-page:horizontal { background: @accent; border-radius: 3px; }
QSlider::handle:horizontal {
    background: @highlight; width: 16px; margin: -6px 0; border-radius: 8px;
}
QFrame#flag_yellow { background-color: #FCD116; border: none; }
QFrame#flag_blue   { background-color: #003893; border: none; }
QFrame#flag_red    { background-color: #CE1126; border: none; }
"""

STYLE = STYLE_TEMPLATE
for name, value in PALETTE.items():
    STYLE = STYLE.replace("@" + name, value)

app = QApplication(sys.argv)
app.setStyle("Fusion")
app.setStyleSheet(STYLE)

# ----------- STARTUP -------------
# Start pygame, init display, clock, font
pygame.init()
screen = pygame.Surface((1200, 720))
font = pygame.font.SysFont("menlo", 18)
running = True

# logging lists for the plots at the end
log_t, log_heading, log_yaw, log_sail, log_rudder = [], [], [], [], []
sim_t = 0.0

# trail of past positions, stored in world coordinates
trail = []

# dial positions (renamed so they don't clash with cx, cy)
hd_cx, hd_cy = 120, 150
rd_cx, rd_cy = 120, 400
dial_r = 100

desired_heading = math.degrees(x0[2])   # the trim heading is your "desired heading" now


window = MainWindow()
window.wind_slider.setValue(round(base_wind_speed))
window.wind_slider.valueChanged.connect(lambda v: setattr(wind, "speed0", float(v)))

# Starting values for the boat
sim.heading = trim_heading + 10
sim.boat_velocity_x = vx0 + 0.2
sim.boat_velocity_y = vy0

sim.sail_angle = sail0

# ---------- MAIN LOOP ------------
timer = QTimer()

timer.timeout.connect(qt_update)

timer.start(int(dt * 1000))

window.show()

sys.exit(app.exec())