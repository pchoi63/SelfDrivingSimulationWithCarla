#Younghoon Choi
#Professor Ozgur Izmirli
#Fall of 2022
"""
Welcome to CARLA manual control.
Use ARROWS or WASD keys for control.
    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    TAB          : change sensor position
    `            : next sensor
    [1-9]        : change to sensor [1-9]
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle
    R            : toggle recording images to disk
    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)
    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import io

try:
    sys.path.append(glob.glob('carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla



from carla import ColorConverter as cc
import cv2
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

from matplotlib import pyplot as plt

import numpy as np
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm


try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
    #from agents.navigation.global_route_planner import GlobalRoutePlanner
    #from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
    
    from carla import ColorConverter as cc
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Global and Junghwan Kim functions -----------------------------------------
# ==============================================================================

def sumMatrix(A, B):
    A = np.array(A)
    B = np.array(B)
    answer = A + B
    return answer.tolist()

count30 = 0
count60 = 0
count90 = 0

def signDetect(cutimage, indexIm):
    print("Final Project")
    #print(count30)

    os.chdir('C:/Users/pchoi/Carla/PythonApi/')
    Categories = ['speed30', 'speed60', 'speed90']

    model = pickle.load(open('img_modell.p', 'rb'))

    img_resized = resize(cutimage, (120,120,3))
    imgFlattened.append(img_resized.flatten())
    flat_data = np.array(imgFlattened)
    y_out = model.predict(flat_data)
    y_out = Categories[y_out[0]]

    print(indexIm)
    
    org = (320, 140)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if(y_out =='speed30'):
        #os.chdir(dir30)
        #cv2.imwrite('image30'+ haha +'.jpg', cutimage)
        speed_con == 0

    elif(y_out =='speed60'):
        #os.chdir(dir60)
        #cv2.imwrite('image60'+ haha + '.jpg', cutimage)
        speed_con == 1

    elif(y_out =='speed90'):
        #os.chdir(dir90)
        #cv2.imwrite('image90'+ haha +'.jpg', cutimage)
        speed_con == 2

    elif(y_out =='backSign'):
        #os.chdir(bSign)
        #cv2.imwrite('bSign'+ haha +'.jpg', cutimage)
        print("backsign")
    
    
    #cv2.imshow("Detected Sign",cutimage)
    

    return y_out


test_con = 0  # test function
speed_con = 0

IM_WIDTH = 640
IM_HEIGHT = 480

imgFlattened = []

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


template30 = cv2.imread('30.jpg',cv2.IMREAD_COLOR)
template60 = cv2.imread('60.jpg',cv2.IMREAD_COLOR)
template90 = cv2.imread('90.jpg',cv2.IMREAD_COLOR)

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def relative_location(frame, location):
        origin = frame.location
        forward = frame.get_forward_vector()
        right = frame.get_right_vector()
        up = frame.get_up_vector()
        disp = location - origin
        x = np.dot([disp.x, disp.y, disp.z], [forward.x, forward.y, forward.z])
        y = np.dot([disp.x, disp.y, disp.z], [right.x, right.y, right.z])
        z = np.dot([disp.x, disp.y, disp.z], [up.x, up.y, up.z])
        
        return carla.Vector3D(x, y, z)

def control_pure_pursuit(vehicleLoc, vehicle_tr, waypoint_tr, max_steer, wheelbase):
        # TODO: convert vehicle transform to rear axle transform
        origin = vehicle_tr.location
        forward = vehicle_tr.get_forward_vector()
        right = vehicle_tr.get_right_vector()
        up = vehicle_tr.get_up_vector()
        disp = waypoint_tr.location - origin
        x = np.dot([disp.x, disp.y, disp.z], [forward.x, forward.y, forward.z])
        y = np.dot([disp.x, disp.y, disp.z], [right.x, right.y, right.z])
        z = np.dot([disp.x, disp.y, disp.z], [up.x, up.y, up.z])
        
        hehe = carla.Vector3D(x, y, z)
        wp_loc_rel = hehe + carla.Vector3D(wheelbase, 0, 0)
        wp_ar = [wp_loc_rel.x, wp_loc_rel.y]
        d2 = wp_ar[0]**2 + wp_ar[1]**2
        steer_rad = math.atan(2 * wheelbase * wp_loc_rel.y / d2)
        steer_deg = math.degrees(steer_rad)
        steer_deg = np.clip(steer_deg, -max_steer, max_steer)
        
        return steer_deg / max_steer


def tempMatch(frameIm):
    signs = [template30, template60, template90]
    which = ["30 MPH: ", "60 MPH: ", "90 MPH: "]

    org = (320, 140)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    currVal = 0
    currInd = 0
    img_gray = cv2.cvtColor(frameIm, cv2.COLOR_BGR2GRAY)
    i = 0

    for i in range(len(signs)):
        template = signs[i]
        temIm = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(temIm, dsize=(img_gray.shape[1],img_gray.shape[0]))

        res = cv2.matchTemplate(img_gray,im,cv2.TM_CCOEFF_NORMED)
        print("Similarity for " + which[i]+ str(res))
        threshold = 0.2
        

        if (currVal < res):
            currVal = res
            currInd = i
        
    if(currVal > threshold):
        print("Current Value:  " + str(currVal))
        if (currInd == 0):
            ans = "30 MPH Detected"
        elif (currInd == 1):
            ans = "60 MPH Detected"
        else:
            ans = "90 MPH Detected"

        print("From Index: " + ans)
        haha = signDetect(frameIm, currVal)
        #cv2.putText(frameIm, ans, org, font, 0.7, (0, 0, 255), 2)
        print(haha)
        return haha

def redDetect(image):
    newImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #lower_red = np.array([141, 155, 84], dtype = "uint8") 

    lower_red = np.array([0,  100, 20], dtype = "uint8") 
    upper_red= np.array([10, 255, 255], dtype = "uint8")

    lower_red2 = np.array([160,  100, 20], dtype = "uint8") 
    upper_red2 = np.array([179, 255, 255], dtype = "uint8")

    lower_green = np.array([25,  52, 72], dtype = "uint8") 
    upper_green= np.array([102, 255, 255], dtype = "uint8")

    lower_green2 = np.array([160,  100, 20], dtype = "uint8") 
    upper_green2 = np.array([179, 255, 255], dtype = "uint8")

    gmask = cv2.inRange(newImage, lower_green, upper_green)
    gmask2 = cv2.inRange(newImage, lower_green2, upper_green2)

    rmask = cv2.inRange(newImage, lower_red, upper_red)
    rmask2 = cv2.inRange(newImage, lower_red2, upper_red2)

    gfullMask = gmask + gmask2

    rfullMask = rmask + rmask2
    
    hasGreen = np.sum(gfullMask)

    hasred = np.sum(rfullMask)
    global speed_con
    if(hasred > 0):
        print("whatred"+ str(hasred))
        print("Red Light Detected")
        
        speed_con = 3
        return "RED"
    elif (hasGreen > 0):
        print("green"+ str(hasGreen))
        print("Green Light Detected")
        speed_con = 0
        return "GREEN"

    else:
        speed_con = 0
        return ""
    
    
    #detected_output = cv2.bitwise_and(image, image, mask = rfullMask) 
    #cv2.imshow("red color detection", detected_output) 
     
def houghCircle(frameIm):
    gray = cv2.cvtColor(frameIm, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.medianBlur(gray, 5)
    
    org = (200, 140)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=1, maxRadius=30)
    
    i = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print("Circle Detected")
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            xVal1 = center[0]-radius
            xVal2 = center[0] + radius
            yVal = center[1] - radius
            yVal2 = center[1] + radius
            testIm = frameIm[yVal:yVal2, xVal1:xVal2]


            height1 = round(testIm.shape[0] *0.75)
            height2 = round(testIm.shape[0] * 0.25)
            width1 = round(testIm.shape[1] * 0.75)
            width2 = round(testIm.shape[1] * 0.25)

            redIm = testIm[height2:height1,width2:width1]
            value = redDetect(redIm)
            if (value == ""):
                #cv2.imshow("test", testIm)
                whatSign = tempMatch(testIm)

                return "speed60"
            #print("template matching succesful")
            else:
                return value
    
    cv2.waitKey(1)


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, actor_filter, actor_role_name='hero'):
        self.world = carla_world
        self.actor_role_name = actor_role_name
        self.map = self.world.get_map()
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        vehicle = self.player

    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 1
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()

        while self.player is None:
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            #spawn_point = carla.Transform(carla.Location(x=248.546844, y=-392.229828, z=0.281942), carla.Rotation(pitch=0.000000, yaw=180, roll=0.000000))
            #spawn_point = carla.Transform(carla.Location(x=32.640914, y=-337.974518, z=0.001780), carla.Rotation(pitch=0.000027, yaw=133.213440, roll=0.007565))
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            global hipo
            hipo = self.player.get_location()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)


        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)

        

        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        #################################################################################
        self.hud.render(display)
        #################################################################################

    def vehicleLoc(self):
        return self.player.get_location()

    def vehicleTrans(self, t):
        self.player.set_transform(t)
    def vehicleTransform(self):
        return self.player.get_transform()
    
    def vehiclePhy(self):
        return self.player.get_physics_control()

    def vehicle(self):
        return self.player

    def currSpeed(self):
        print(self.player.get_velocity())
        #return self.player.get_acceleration()
    def speedSet(self):
        self.set_velocity(8.3) 
        print("sets speed to 30 km/h (8.3 m/s)")


    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    currentIndex = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(currentIndex)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not (pygame.key.get_mods() & KMOD_CTRL):
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())

                ############################### 차량 제어 파트 ###############################
                
                
                
                global test_con
                global speed_con

                #print(speed_con)
                #print(str(test_con) + " test con")
                '''
                if speed_con == 0:
                    self._control.throttle = 0.1
                elif speed_con == 1:
                    self._control.throttle = 0.13
                elif speed_con == 2:
                    self._control.throttle = 0.15
                elif speed_con ==3:
                    #full brake
                    self._control.brake = 10


                if test_con== -2 :
                    self._control.steer = -0.035
                elif test_con == -1 :
                    self._control.steer = -0.015
                elif test_con == 1 :
                    self._control.steer = 0.015
                elif test_con == 2 :
                    self._control.steer = 0.035

                '''
                ##############################################################################
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        fonts = pygame.font.get_default_font()
        print(fonts)
    
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame_number - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.tesla.model3')
        #vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt(
                (l.x - t.location.x) ** 2 + (l.y - t.location.y) ** 2 + (l.z - t.location.z) ** 2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame_number, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '5000')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            # Original code Don't touch
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            #################################################
            # it's my code
            pt1_sum_ri = (0, 0)
            pt2_sum_ri = (0, 0)
            pt1_avg_ri = (0, 0)
            count_posi_num_ri = 0

            pt1_sum_le = (0, 0)
            pt2_sum_le = (0, 0)
            pt1_avg_le = (0, 0)

            count_posi_num_le = 0

            test_im = np.array(image.raw_data)
            test_im = test_im.copy()
            test_im = test_im.reshape((image.height, image.width, 4))
            test_im = test_im[:, :, :3]
            size_im = cv2.resize(test_im, dsize=(640, 480)) 
            roi = size_im[240:480, 108:532] #[240:480, 108:532][380:430, 330:670]   [y:y+b, x:x+a]
            roi_im = cv2.resize(roi, (424, 240))  # (a of x, b of y)
            #480, 640
            
            
            hVal = image.height //2
            wVal = image.height //2
            trafficImage = size_im[:hVal, wVal :]

            signImage = size_im[:hVal,:]

            hehe = houghCircle(signImage)
            
            
            org = (320, 100)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            cv2.putText(size_im, hehe, org, font, 0.7, (0, 0, 255), 2)

            #cv2.imshow("plz", trafficImage)
            #################################################
            # Gaussian Blur Filter
            Blur_im = cv2.bilateralFilter(roi_im, d=-1, sigmaColor=3, sigmaSpace=3)
            #################################################

            #################################################
            # Canny edge detector
            edges = cv2.Canny(Blur_im, 50, 100)
            #cv2.imshow("edges", edges)
            #################################################

            #################################################
            # Hough Transformation
            #lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180.0, threshold=80, minLineLength=30, maxLineGap=50)
            # rho, theta는 1씩 변경하면서 검출하겠다는 의미, np.pi/180 라디안 = 1'
            # threshold 숫자가 작으면 정밀도↓ 직선검출↑, 크면 정밀도↑ 직선검출↓
            # min_line_len 선분의 최소길이
            # max_line,gap 선분 사이의 최대 거리
            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180.0, threshold=22, minLineLength=10, maxLineGap=20)

            N = lines.shape[0]

            '''
            print('range_N=',range(N))
            if range(N) == 0 :
                print("bad")
            elif range(N) == 0 : print("good")
            '''

            for line in range(N):
                # for line in lines:

                # x1, y1, x2, y2 = line[0]

                x1 = lines[line][0][0]
                y1 = lines[line][0][1]
                x2 = lines[line][0][2]
                y2 = lines[line][0][3]

                if x2 == x1:
                    a = 1
                else:
                    a = x2 - x1

                b = y2 - y1

                radi = b / a 

                theta_atan = math.atan(radi) * 180.0 / math.pi
                # print('theta_atan=', theta_atan)

                pt1_ri = (x1 + 108, y1 + 240)
                pt2_ri = (x2 + 108, y2 + 240)
                pt1_le = (x1 + 108, y1 + 240)
                pt2_le = (x2 + 108, y2 + 240)

                if theta_atan > 30.0 and theta_atan < 80.0:
                    count_posi_num_ri += 1

                    pt1_sum_ri = sumMatrix(pt1_ri, pt1_sum_ri)
                    pt2_sum_ri = sumMatrix(pt2_ri, pt2_sum_ri)
                if theta_atan < -30.0 and theta_atan > -80.0:
                    count_posi_num_le += 1

                    pt1_sum_le = sumMatrix(pt1_le, pt1_sum_le)
                    pt2_sum_le = sumMatrix(pt2_le, pt2_sum_le)
            
            pt1_avg_ri = pt1_sum_ri // np.array(count_posi_num_ri)
            pt2_avg_ri = pt2_sum_ri // np.array(count_posi_num_ri)
            pt1_avg_le = pt1_sum_le // np.array(count_posi_num_le)
            pt2_avg_le = pt2_sum_le // np.array(count_posi_num_le)

            #################################################
            # 차석인식의 흔들림 보정
            # right-----------------------------------------------------------
            x1_avg_ri, y1_avg_ri = pt1_avg_ri
            x2_avg_ri, y2_avg_ri = pt2_avg_ri

            a_avg_ri = ((y2_avg_ri - y1_avg_ri) / (x2_avg_ri - x1_avg_ri))
            b_avg_ri = (y2_avg_ri - (a_avg_ri * x2_avg_ri))

            pt2_y2_fi_ri = 480


            if a_avg_ri > 0:
                pt2_x2_fi_ri = int((pt2_y2_fi_ri - b_avg_ri) // a_avg_ri)
            else:
                pt2_x2_fi_ri = 0

            pt2_fi_ri = (pt2_x2_fi_ri, pt2_y2_fi_ri)

            # left------------------------------------------------------------
            x1_avg_le, y1_avg_le = pt1_avg_le
            x2_avg_le, y2_avg_le = pt2_avg_le

            a_avg_le = ((y2_avg_le - y1_avg_le) / (x2_avg_le - x1_avg_le))
            b_avg_le = (y2_avg_le - (a_avg_le * x2_avg_le))

            pt1_y1_fi_le = 480
            if a_avg_le < 0:
                pt1_x1_fi_le = int((pt1_y1_fi_le - b_avg_le) // a_avg_le)
            else:
                pt1_x1_fi_le = 0

            pt1_fi_le = (pt1_x1_fi_le, pt1_y1_fi_le)
            #################################################

            #################################################
            # lane painting
            # right-----------------------------------------------------------
            cv2.line(size_im, tuple(pt1_avg_ri), tuple(pt2_fi_ri), (0, 255, 0), 2)  # right lane
            # left-----------------------------------------------------------
            cv2.line(size_im, tuple(pt1_fi_le), tuple(pt2_avg_le), (0, 255, 0), 2)  # left lane
            # center-----------------------------------------------------------
            cv2.line(size_im, (320, 480), (320, 360), (0, 228, 255), 1)  # middle lane
            #################################################

            #################################################
            # possible lane
            #################################################
            FCP_img = np.zeros(shape=(480, 640, 3), dtype=np.uint8) + 0
            FCP = np.array([pt2_avg_le, pt1_fi_le, pt2_fi_ri, pt1_avg_ri])
            cv2.fillConvexPoly(FCP_img, FCP, color=(255, 242, 213))  # BGR
            alpha = 0.9
            size_im = cv2.addWeighted(size_im, alpha, FCP_img, 1 - alpha, 0)
            #################################################

            #################################################
            # lane center 및 steering 계산 (320, 360)
            lane_center_y_ri = 360
            if a_avg_ri > 0:
                lane_center_x_ri = int((lane_center_y_ri - b_avg_ri) // a_avg_ri)
            else:
                lane_center_x_ri = 0

            lane_center_y_le = 360
            if a_avg_le < 0:
                lane_center_x_le = int((lane_center_y_le - b_avg_le) // a_avg_le)
            else:
                lane_center_x_le = 0

            # caenter left lane (255, 90, 185)
            cv2.line(size_im, (lane_center_x_le, lane_center_y_le - 10), (lane_center_x_le, lane_center_y_le + 10),
                     (0, 228, 255), 1)
                     
            # caenter right lane
            cv2.line(size_im, (lane_center_x_ri, lane_center_y_ri - 10), (lane_center_x_ri, lane_center_y_ri + 10),
                     (0, 228, 255), 1)

            # caenter middle lane
            lane_center_x = ((lane_center_x_ri - lane_center_x_le) // 2) + lane_center_x_le
            cv2.line(size_im, (lane_center_x, lane_center_y_ri - 10), (lane_center_x, lane_center_y_le + 10),
                     (0, 228, 255), 1)

            # print('lane_center_x=', lane_center_x)

            text_left = 'Turn Left'
            text_right = 'Turn Right'
            text_center = 'Center'
            text_non = ''
            org = (320, 440)
            font = cv2.FONT_HERSHEY_SIMPLEX

            global test_con

            if 0 < lane_center_x <= 286:
                cv2.putText(size_im, "Sharp Left", org, font, 0.7, (0, 0, 255), 2)
                test_con = -2

            elif 286 < lane_center_x <= 316:
                cv2.putText(size_im, text_left, org, font, 0.7, (0, 0, 255), 2)
                test_con = -1

            elif 316 < lane_center_x < 324:
                cv2.putText(size_im, text_center, org, font, 0.7, (0, 0, 255), 2)
                test_con = 0
            
            elif 324 < lane_center_x < 354:
                cv2.putText(size_im, text_right, org, font, 0.7, (0, 0, 255), 2)
                test_con = 1

            elif lane_center_x >= 354:
                cv2.putText(size_im, "Sharp Right", org, font, 0.7, (0, 0, 255), 2)
                test_con = 2

            elif lane_center_x == 0:
                cv2.putText(size_im, text_non, org, font, 0.7, (0, 0, 255), 2)
            #################################################

            
            #test_con = 1
            #print('test_con=', test_con)

            # 변수 초기화
            count_posi_num_ri = 0

            pt1_sum_ri = (0, 0)
            pt2_sum_ri = (0, 0)
            pt1_avg_ri = (0, 0)
            pt2_avg_ri = (0, 0)

            count_posi_num_le = 0

            pt1_sum_le = (0, 0)
            pt2_sum_le = (0, 0)
            pt1_avg_le = (0, 0)
            pt2_avg_le = (0, 0)

            cv2.imshow('frame_size_im', size_im)
            cv2.waitKey(1)
            # cv2.imshow("test_im", test_im) # original size image
            # cv2.waitKey(1)

        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame_number)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(35.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)

        # Changing The Map
        world = World(client.load_world('Town02'), hud, args.filter,args.rolename)  


        '''
        vertex_distance = 2.0  # in meters
        max_road_length = 500.0 # in meters
        wall_height = 0.0      # in meters
        extra_width = 0.6      # in meters

        f = io.open("C:/Users/pchoi/Carla/PythonAPI/map.xodr", encoding="utf-8")

        xodr_xml = f.read()

        world = client.generate_opendrive_world(xodr_xml, carla.OpendriveGenerationParameters(
            vertex_distance=vertex_distance,
            max_road_length=max_road_length,
            wall_height=wall_height,
            additional_width=extra_width,
            smooth_junctions=True,
            enable_mesh_visibility=True))

        '''


        # Town04 ,Town06 is highway | Town07 is country
        world = World(client.get_world(), hud, args.filter, args.rolename)
        
        
        controller = KeyboardControl(world, args.autopilot)
        map = world.world.get_map()
        clock = pygame.time.Clock()
        count = 0
        
        w = map.get_waypoint(hipo)
        print(world.vehicleLoc())

        '''

        worldd = client.get_world()

        blueprint = worldd.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        blueprint.set_attribute('image_size_x', '1920')
        blueprint.set_attribute('image_size_y', '1080')
        blueprint.set_attribute('fov', '110')
        # Provide the position of the sensor relative to the vehicle.
        transform = carla.Transform(carla.Location(carla.Location(x=0.8, z=1.7)))
        # Tell the world to spawn the sensor, don't forget to attach it to your vehicle actor.
        #sensor = worldd.spawn_actor(blueprint, transform, attach_to=world.vehicle())


        '''
        # Subscribe to the sensor stream by providing a callback function, this function is
        # called each time a new image is generated by the sensor.

        #sensor.listen(lambda data: showImage(data))


        #world.vehicle(carla.Transform(carla.Location(x=46.149979, y=330.459991, z=0.001598)), carla.Rotation(pitch=0.000027, yaw=133.213440, roll=0.007565))
        world.vehicleTrans(carla.Transform(carla.Location(x=-3.679999, y=121.209999, z=0.199949), carla.Rotation(pitch=0.000000, yaw=-89.999817, roll=0.000000)))
        print(world.vehicleTransform())

        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()
            #skip
            #print(world.vehicleTransform())
            print(world.currSpeed())
            world.speedSet()
            '''
            if count % 10 == 0:
                
                nexts = list(w.next(1.0))
                #print('Next(1.0) --> %d waypoints' % len(nexts))
                if not nexts:
                    raise RuntimeError("No more waypoints!")
                w = random.choice(nexts)
                text = "road id = %d, lane id = %d, transform = %s"
                #print(text % (w.road_id, w.lane_id, w.transform))
                if count % 40 == 0:
                    #draw_waypoints(world.world, w)
                    count = 0
                t = w.transform
                #carla.Vehicle.set_transform(t)
                #print(t)

                print(str(world.vehicleTransform()))

                physics_control = world.vehiclePhy()
                max_steer = physics_control.wheels[0].max_steer_angle
                rear_axle_center = (physics_control.wheels[2].position + physics_control.wheels[3].position)/200
                offset = rear_axle_center - world.vehicleLoc()
                wheelbase = np.linalg.norm([offset.x, offset.y, offset.z])
                world.vehicleTrans(t)
                steer = control_pure_pursuit(world.vehicleLoc(), world.vehicleTransform(),  t, max_steer, wheelbase)
                control = carla.VehicleControl(0.9, steer)
                
            count += 1
            '''

    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()

def draw_waypoints(world, waypoint, depth=0):
    if depth < 0:
        return
    for w in waypoint.next(4.0):
        t = w.transform
        begin = t.location + carla.Location(z=0.5)
        angle = math.radians(t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.4, life_time=1.0)
        draw_waypoints(world, w, depth - 1)



def change_lane(waypoint, n):
    if (n > 0):
        return get_left_lane_nth(waypoint, n)
    else:
        return get_right_lane_nth(waypoint, n)


def get_right_lane_nth(waypoint, n):
    out_waypoint = waypoint
    for i in range(n):
        out_waypoint = out_waypoint.get_right_lane()
    return out_waypoint

def get_left_lane_nth(waypoint, n):
    out_waypoint = waypoint
    for i in range(n):
        out_waypoint = out_waypoint.get_left_lane()
    return out_waypoint
    
# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='640x480',  # '1280x720'
        help='window resolution (default: 640x480)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.tesla.model3',
        help='actor filter (default: "vehicle.tesla.model3")')
    # default='vehicle.*',
    # help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()