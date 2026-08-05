import streamlit as st
# import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from copy import copy
from math import cos, sin, radians, degrees, tan, atan
import cv2


gravity = 9.81 # m/s^2
air_density = 1.225 # kg/m^3
ball_mass = 0.0459 # kg
ball_cross_section = 0.00143 # m^2
drag_coeff = 0.23 # assumed constant
lift_coeff = 0.18 # assumed constant

CM_PER_M = 100
CM_PER_INCH = 2.54
INCHES_PER_FOOT = 12

CONV_FAC_M_TO_FT = CM_PER_M / (CM_PER_INCH * INCHES_PER_FOOT)

m2f = lambda meters: meters * CONV_FAC_M_TO_FT
m2y = lambda meters: m2f(meters) / 3


DKGRN = (30,125,10)
RED = (0,0,255)
BLUE = (255,0,0)
MGTA = (255,0,255)
BLK = (0,0,0)
WHT = (255,255,255)
MDYLW = (0,160,160)

LTGRN = (60,175,25)
FILLED = -1
toint = lambda x: int(0.5+ x)


club_loft_lookup = {
        "D": 10.5,
        "4I": 22.5,
        "5I": 26,
        "6I": 30,
        "7I": 33,
        "8I": 37,
        "9I": 42,
        "PW": 46,
        "AW": 52,
        "SW": 56,
        "LW": 60    
    }

smash_factor_lookup = {
        "D": 1.46,
        "4I": 1.42,
        "5I": 1.4,
        "6I": 1.37,
        "7I": 1.35,
        "8I": 1.31,
        "9I": 1.27,
        "PW": 1.24,
        "AW": 1.18,
        "SW": 1.12,
        "LW": 1.06    
    }

st.title("Golf Shot Shape Simulator")
st.write("Adjust the variables on the left to see the shot shape change in real-time.")
with st.expander("📘 How to Use This Simulator & Master the Face-to-Path Rule"):
    st.markdown("""
    ### 🎯 The Goal: Hit a Predictable Draw or Fade
    In golf physics, a ball's curvature is dictated entirely by the relationship between your **Club Path** and your **Face Angle** at the moment of impact.
    
    * **The Face Angle** dictates roughly 75% to 80% of where the ball *starts* (Launch Direction).
    * **The Club Path** interacting with that face is what creates the spin axis tilt that makes the ball *curve*.

    ---

    ### 🔑 The Golden Rule: The 30% to 50% Sweet Spot
    To hit a functional, controllable shot that curves back toward your target without over-hooking or blocking, aim for a **Face-to-Path ratio between 30% and 50%**.

    #### 🟢 How to Simulate a Perfect Draw (In-to-Out Swing)
    1. Set your **Club Path** to a positive value (e.g., `+6.0°` swinging out to the right).
    2. Adjust your **Face Angle** so it sits between **30% and 50%** of that path number (e.g., `+2.0°` to `+3.0°`).
    3. **The Result:** The ball launches safely to the right of your target, then gently draws back to the center of the green.

    #### 🔵 How to Simulate a Perfect Fade (Out-to-In Swing)
    1. Set your **Club Path** to a negative value (e.g., `-6.0°` swinging left).
    2. Adjust your **Face Angle** so it stays between **30% and 50%** of that path (e.g., `-2.0°` to `-3.0°`).
    3. **The Result:** The ball launches to the left, then fades cleanly back to the flag.

    ---

    ### ⚠️ What Happens When You Miss the Ratio?
    * **Below 30% (The Over-Hook / Over-Slice):** The face is pointing too far away from your swing path. This tilts the spin axis aggressively, causing a violent hook or a massive slice that completely misses the green.
    * **Above 50% (The Push / Pull Block):** The face angle is too close to your swing path. The ball will launch offline but won't have enough spin axis tilt to curve back toward the target line, leaving you stranded out wide.
    
    *Use the sliders above to see these rules in action!*
    """)
st.write("---")  # Adds a clean visual divider line
st.set_page_config(page_title="Golf Shot Shape Simulator", layout="wide")
col1, col2, col3 = st.columns([1, 2, 3])
with col1:
    st.subheader("Swing Inputs")
    club_path = st.slider("Club Path (°)", min_value=-15.0, max_value=15.0, value=0.0, step=0.05)
    face_angle = st.slider("Face Angle (°)", min_value=-15.0, max_value=15.0, value=0.0, step=0.05)
with col3:
    st.subheader("Adjust Image Size")
    scale_width = st.slider("Image Width (px)", min_value = 100, max_value = 1080, value = 500, step =10)
    





# rsz_fac = st.slider("Scale", min_value=0.1, max_value=1.0, value=0.3, step=0.01)

class GolfTrajectoryGenerator:
    def __init__(self, theta_path, theta_face,re_init=False, wk=0):
        self.theta_path = theta_path
        self.wk = wk
        self.theta_face = theta_face
        
        self.init_mpp = False
        self.mpp = 0
        
        
        self.drag_accel = 0.0
        self.lift_accel = 0.0
        self.vert_lift_accel = 0.0
        self.side_curve_accel = 0.0
        self.vx, self.vy, self.vz = 0.0, 0.0, 0.0
        self.vx0, self.vy0, self.vz0 = 0.0, 0.0, 0.0
        self.lift__accel = 0.0
        self.hld = 0.0
        self.vla = 0.0
        self.phi_axis = 0.0
        
        self.const_multiplier = -1.0 * air_density * ball_cross_section / (2.0 * ball_mass)
        
        
        self.dyn_loft = 0
        
        if not re_init:
            self.data_start = False
            self.shot_data = {}
            self.club_name = "7I"
            self.swing_speed = 85
            self.club_speed = self.swing_speed * 0.44704
        self.smash_factor =  smash_factor_lookup[self.club_name]
        self.v0 =  self.smash_factor * self.club_speed
        
        
        
    def calc_flight(self):
        self.xpos_arr = [0.0]
        self.ypos_arr = [0.0]
        self.zpos_arr = [0.0]
        self.shape_name = ""
        self.theta_path = copy(club_path)
        self.theta_face = copy(face_angle)
        self.calc_params()
        self.calc_initial_velocities()
        self.calculate_trajectory()
        
        
        
    def re_init(self, theta_path, theta_face,wk=0):
        # data_started = self.data_start
        # shot_data = {k:v for k,v in self.shot_data.items()}
        self.__init__(theta_path, theta_face,True, wk)
        self.swing_speed = 97 if self.club_name != "D" else 110
        self.club_speed = self.swing_speed * 0.44704
        
        # self.data_start = data_started
        
    def max_dev(self):
        if self.hld < 0:
            abs_arr = [abs(x) if x <= 0 else 0 for x in self.xpos_arr]
            idx = abs_arr.index(max(abs_arr))
            return self.xpos_arr[idx]
        else:
            return max(self.xpos_arr)
    
    def calc_params(self):
        self.dyn_loft = 0.75 * club_loft_lookup[self.club_name] if self.club_name != "D" else 1.45 * club_loft_lookup[self.club_name]
        
        if self.club_name == "D":
            
            self.hld = 0.80 * self.theta_face + 0.2 * self.theta_path
        else:
            self.hld = 0.75 * self.theta_face + 0.25 * self.theta_path
        
        tan_phi = tan(radians(self.theta_face - self.theta_path)) / sin(radians(self.dyn_loft))
        self.phi_axis = degrees(atan(tan_phi))
        
        if self.club_name == "D":
            self.vla = 0.20 * 4
            print("Driver:", self.dyn_loft)
        else:
            self.vla = 0.20 * -4
            
        
        
        self.vla += 0.80 * self.dyn_loft
        print(self.vla)
        
        # print(f"Horizontal Launch Direction: {self.hld:.4f}°")
        # print(f"Vertical Launch Angle: {self.vla:.4f}°")
        # print(f"Initial Ball Speed: {self.v0:.4f} m/s")
        
    
    
    def calc_initial_velocities(self):
        self.vx0 = self.v0 * cos(radians(self.vla)) * sin(radians(self.hld))
        self.vy0 = self.v0 * cos(radians(self.vla)) * cos(radians(self.hld))
        self.vz0 = self.v0 * sin(radians(self.vla))
        
        self.vx = copy(self.vx0)
        self.vy = copy(self.vy0)
        self.vz = copy(self.vz0)
        # print(f"Vx0: {self.vx:.4f} m/s")
        # print(f"Vy0: {self.vy:.4f} m/s")
        # print(f"Vz0: {self.vz:.4f} m/s")
        # print(f"Spin Axis: {self.phi_axis:.4f}°")
    
    
    def classify_shot_shape(self):
        
        face_to_path = self.theta_face - self.theta_path
        corridor = 0.03 * self.ypos_arr[-1]
        
        # no/low side-spin case
        # push/block means 
        if self.theta_face > 1.5 and abs(face_to_path) < 1.0:
            self.shape_name = "Push/Block"
        elif abs(self.theta_face) <= 1.5 and (abs(face_to_path) <= 1.0 or abs(self.xpos_arr[-1]) < corridor):
            self.shape_name = "Straight"
        elif self.theta_face < 1.5 and abs(face_to_path) < 1.0:
            self.shape_name = "Pull"
        
        # The really bad shots
        elif (abs(self.theta_face) < 1.5 and face_to_path > 1.0) or abs(self.hld) < 1.5 and face_to_path > 1.5:
            self.shape_name = "Slice"
        elif abs(self.theta_face) < 1.5 and face_to_path < -1.0 or abs(self.hld) < 1.5 and face_to_path < -1.5:
            self.shape_name = "Hook"
            
        elif self.max_dev() < 0 and face_to_path < 0:
            self.shape_name = "Pull-Hook"
        elif self.max_dev() > 0 and face_to_path > 0:
            self.shape_name = "Push-Slice"
       
        # draw - starts right, lands in the corridor
        elif self.max_dev() > 0 and abs(self.xpos_arr[-1]) < corridor:
            self.shape_name = "Draw"
        elif self.max_dev() > 0 and self.xpos_arr[-1] < -corridor:
            self.shape_name = "Push-Hook"
        elif self.max_dev() > 0 and self.xpos_arr[-1] > corridor:
            self.shape_name = "Push-Draw"
        
        # fade start left, lands in the corridor
        
        elif self.max_dev() < 0 and abs(self.xpos_arr[-1]) < corridor: 
            self.shape_name = "Fade"
        elif self.max_dev() < 0 and self.xpos_arr[-1] < -corridor: 
            self.shape_name = "Pull-Fade"
        elif self.max_dev() < 0 and self.xpos_arr[-1] > corridor: 
            self.shape_name = "Pull-Slice"
        
        # elif face_to_path > 0:
        #     self.shape_name = "Slice"
        # elif face_to_path < 0:
        #     self.shape_name = "Hook"
        
        # print(self.hld)
        
        
        
        
    
    def append_data(self):
        if not self.data_start:
            for x in ["Shot Type", "Club Path (°)", "Face Angle (°)", "Face To Path (°)", "Carry (yd)", "Off-Center (yd)", "Off-Center Ratio (%)", "Apex (ft)"]:        
                self.shot_data[x] = []
            print("initialized")
        self.data_start = True
        
        
        self.shot_data["Shot Type"].append(self.shape_name)
        self.shot_data["Club Path (°)"].append(self.theta_path)
        self.shot_data["Face Angle (°)"].append(self.theta_face)
        self.shot_data["Face To Path (°)"].append(self.theta_face - self.theta_path)
        self.shot_data["Carry (yd)"].append(self.ypos_arr[-1])
        self.shot_data["Off-Center (yd)"].append(m2y(self.xpos_arr[-1]))
        self.shot_data["Off-Center Ratio (%)"].append(100*self.xpos_arr[-1]/self.ypos_arr[-1])
        self.shot_data["Apex (ft)"].append(m2f(max(self.zpos_arr)))
        
            
            
        
        
        
        
        
        
        
        
        
    
    
    def draw_trajectory(self):
        
        
        
        
        
        self.classify_shot_shape()
        # if not self.data_start or self.wk == 0:
        wd = 1080
        ht = 1920
        im = np.zeros((ht,wd,3)).astype("uint8")
        
        # color the whole image dark green to start
        im[:,:] = DKGRN
        
        field_wd = toint(0.85 * wd)
        field_ht = toint(0.8 * ht)
        
        og_x = toint(wd/2)
        og_y = ht - toint(ht * 0.075)
        
        if not self.init_mpp:
            # calculate meters per pixel based on 
            mpp_wd = self.max_dev() * 2 / field_wd
            mpp_ht = max(self.ypos_arr) / field_ht
            self.mpp = max([mpp_wd, mpp_ht])
            self.pin_pos = (og_x, og_y - toint( self.ypos_arr[-1] / self.mpp))
            self.init_mpp = True
            
        
        
        
        
        
        # draw the tee box
        # it's absurdly big, get over it
        tee_box_width = 25 # meters
        tee_box_x1 = og_x - toint((tee_box_width/self.mpp))
        tee_box_x2 = og_x + toint(tee_box_width/self.mpp)
        
        tee_box_len = 15 # meters
        tee_box_y1 = og_y - toint((tee_box_len/self.mpp))
        tee_box_y2 = og_y + toint(tee_box_len/self.mpp)
        
        # change the teebox area to a lighter green
        tee_roi = im[tee_box_y1:tee_box_y2, tee_box_x1: tee_box_x2]
        tee_roi[:,:] = LTGRN
        
        # draw the tee markers
        tee_ht = len(tee_roi)
        tee_wd = len(tee_roi[0])
        
        tee_marker_pos = (toint( tee_wd *0.1), toint( tee_ht/2))
        cv2.circle(tee_roi,tee_marker_pos,15,BLUE, FILLED)
        tee_marker_pos = (toint( tee_wd *0.9), toint( tee_ht/2))
        cv2.circle(tee_roi,tee_marker_pos,15,BLUE, FILLED)
        
        
        
        # calculate the end point of the horizontal launch arrow
        # og_x and og_y will change before we do this, just save them here
        hld_x = toint( og_x + sin(radians(self.hld))*0.25 * ht)
        hld_y = toint( og_y - cos(radians(self.hld))*0.25 * ht) 
        
        
        # draw the green
        # green and cup are also absurdly large - to show texture
        green_radius = 28 # meters
        green_pxrad = toint(green_radius/self.mpp)    
        cv2.circle(im,self.pin_pos,green_pxrad,LTGRN,-1)
        cup_radius = 1.5 # meters
        cup_pxrad = toint(cup_radius/self.mpp)
        cv2.circle(im,self.pin_pos,cup_pxrad,BLK,-1)
        
        # draw the pin
        pin_top = (self.pin_pos[0]-toint(0.025*ht),self.pin_pos[1]-toint(0.05*ht))
        cv2.line(im,self.pin_pos,pin_top,MDYLW,5)
        
        # draw the flag
        flag_tip = (self.pin_pos[0]-toint(0.04*ht),self.pin_pos[1]-toint(0.012*ht))
        for i in range(33,46):
            pt = (self.pin_pos[0]-i,self.pin_pos[1]-(i*2))
            cv2.line(im,flag_tip,pt,(0,0,255),5)

        
        # draw a dashed line down the center of the field
        y_pos = self.pin_pos[1] - toint(ht * .110)
        jump = 100
        while True:
            if y_pos+150 >= ht:
                cv2.line(im,(og_x,y_pos),(og_x,ht),(125,125,125),4)
                break
            else:
                cv2.line(im,(og_x,y_pos),(og_x,y_pos+jump),(125,125,125),4)
            y_pos += 2 * jump
        
        
        cv2.line(im,(og_x,og_y),(hld_x,hld_y),(0,255,255),2)
        
        ldr_x = toint( hld_x + cos(radians(self.hld + 60))*30)
        ldr_y = toint( hld_y + sin(radians(self.hld + 60))*30)
        
        cv2.line(im,(hld_x,hld_y),(ldr_x,ldr_y),(0,255,255),2)
        ldr_x = toint( hld_x - cos(radians(self.hld - 60))*30)
        ldr_y = toint( hld_y + sin(radians(self.hld + 60))*30)
        
        cv2.line(im,(hld_x,hld_y),(ldr_x,ldr_y),(0,255,255),2)
        
        
        
        
        py, px = None, None
        for i in range(len(self.xpos_arr)):
            x = og_x + toint(self.xpos_arr[i]/self.mpp)
            y = og_y - toint(self.ypos_arr[i]/self.mpp)
            
            if i > 0:
                cv2.line(im,(x,y),(px,py),MGTA,3)
            
            px = copy(x)
            py = copy(y)

        
        # # maybe later - draw the side-view to see the apex
        # apex_mpp = self.mpp*2
        # apex_og_x = toint( (wd - field_wd)/2)
        # apex_og_y = self.pin_pos[1] - toint(0.015*ht)
        
        # end_apex_x = apex_og_x + toint( self.ypos_arr[-1] / apex_mpp)
        # while end_apex_x > toint( wd - (wd-field_wd) / 2):
        #     apex_mpp *=1.01
        #     end_apex_x = apex_og_x + toint( self.ypos_arr[-1] / apex_mpp)            
        
        
        
        # for i in range(len(self.xpos_arr)):
        #     x = apex_og_x + toint( self.ypos_arr[i] / apex_mpp)
        #     y = apex_og_y - toint( self.zpos_arr[i] / apex_mpp)
        #     if i > 0:
        #         cv2.line(im,(x,y),(px,py),MGTA,3)
        #     px = copy(x)
        #     py = copy(y)
        # cv2.line(im,(apex_og_x,apex_og_y),(px,py),BLK,2)
        
        
        # print(hld_x, hld_y)
        
        
        
        
        ft = cv2.FONT_HERSHEY_SIMPLEX
        clr = WHT
        sz = 1.2
        ypos = toint(0.35*ht)
        jump = 45
        face_to_path = self.theta_face - self.theta_path
        pct = 100 * self.xpos_arr[-1] / self.ypos_arr[-1]
        face_path_pct = 100 * self.theta_face / self.theta_path if abs(self.theta_path) >= 0.04 else "?"
        texts = [
            # f"Club: {self.club_name}",
            f"Shot: {self.shape_name}",
            f"Path: {self.theta_path:.2f} dg",
            f"Face To Path: {face_to_path:.2f} dg",
            f"Face: {self.theta_face:.2f} dg",
            f"Face:Path: {face_path_pct:.2f} %" if abs(self.theta_path) >= 0.04 else "Face:Path: N/A %",
            # f"Apex: {m2f(max(self.zpos_arr)):.2f} ft",
            f"Carry: {m2y(max(self.ypos_arr)):.2f} yd",
            f"Off-Center Pct: {pct:.2f}%",
            f"Off-Center: {m2y(self.xpos_arr[-1]):.2f} yd",
            f"Max Dev.: {self.max_dev():.2f} yd",
            # f"Dyn. Loft: {self.dyn_loft:.2f} dg",
            # f"Launch Ang: {self.vla:.2f} dg",
            # f"Swing Speed: {self.swing_speed:.0f} mph",
            # f"Smash Fac.: {self.smash_factor:.2f}",
            # f"Ball Speed: {self.swing_speed * self.smash_factor:.0f} mph"
            
            
        ]
        for t in texts:
            cv2.putText(im,t, (10,ypos),ft,sz,clr,2)
            
            ypos += jump
        
        
        
        
        path_rad = toint(0.125*ht)
        
        path_pt = [toint( 0.05*wd), toint( 0.8 * ht)]
        cv2.circle(im,path_pt,path_rad,(0,0,255),4)
        cv2.line(im,path_pt,(path_pt[0]+path_rad,path_pt[1]),(0,0,255),4)
        roi = im[toint( 0.8*ht)-(path_rad+10):toint( 0.8*ht)+(path_rad+10), :toint( 0.05*wd)-1]
        roi[:,:] = (30,125,10)
        
        
        
        
        ang = radians(self.theta_path)
        
        path_x_pos = toint( (path_pt[0] + cos(ang)*path_rad))
        path_y_pos = toint( (path_pt[1] + sin(ang)*path_rad))
        
        path_pos = (path_x_pos,path_y_pos)
        cv2.circle(im,path_pos,15,(255,255,255),-1)
        
        path_pt[0] -= 40
        path_pt[1] -= 20
        cv2.putText(im,"Club Path", path_pt,ft,1.4,(0,0,255),2)
        path_pt[1] += 70
        cv2.putText(im,"Club Face", path_pt,ft,1.4,(0,0,0),2)
        
        
        
        
        
        path_pos = (path_x_pos,path_y_pos)
        cv2.putText(im,"Hztl Launch Dir", (675,path_pt[1]-55),ft,1.4,(0,255,255),3)
        cv2.putText(im,f"{self.hld:.2f} dg", (675,path_pt[1]-10),ft,1.4,(0,255,255),3)
        
        
        ang = radians(self.theta_face+90)
        # face_pos_x = toint( (path_pos[0] + cos(ang)*30))
        # face_pos_y = toint( (path_pos[1] + sin(ang)*30))
        # face_pos = (face_pos_x, face_pos_y)
        
        ang = radians(self.theta_face)
        pt1_x = toint( path_x_pos + cos(ang)* 50)
        pt1_y = toint( path_y_pos + sin(ang)* 50)
        pt2_x = toint( path_x_pos - cos(ang)* 50)
        pt2_y = toint( path_y_pos - sin(ang)* 50)
        pt1 = (pt1_x, pt1_y)
        pt2 = (pt2_x, pt2_y)
        cv2.line(im,pt1,pt2,(0,0,0),4)
        
       
        ang = radians(self.theta_path - 90)
        pt1_x = toint( path_pos[0] + cos(ang) * 300)        
        pt1_y = toint( path_pos[1] + sin(ang) * 300)        
        pt = (pt1_x, pt1_y)
        cv2.line(im,path_pos,pt,(0,0,255),4)
        
        ang = radians(self.theta_path+115)
        ldr_pt_x = toint( pt[0] + cos(ang)*30)
        ldr_pt_y = toint( pt[1] + sin(ang)*30)
        ldr_pt = (ldr_pt_x,ldr_pt_y)
        cv2.line(im,ldr_pt,pt,(0,0,255),4)
        ang = radians(self.theta_path+60)
        ldr_pt_x = toint( pt[0] + cos(ang)*30)
        ldr_pt_y = toint( pt[1] + sin(ang)*30)
        ldr_pt = (ldr_pt_x,ldr_pt_y)
        cv2.line(im,ldr_pt,pt,(0,0,255),4)
        
        
        
        
        ang = radians(self.theta_face - 90)
        
        pt1_x = toint( path_pos[0] + cos(ang) * 200)        
        pt1_y = toint( path_pos[1] + sin(ang) * 200)        
        pt = (pt1_x, pt1_y)
        cv2.line(im,path_pos,pt,(0,0,0),4)
        
        ang = radians(self.theta_face+115)
        ldr_pt_x = toint( pt[0] + cos(ang)*30)
        ldr_pt_y = toint( pt[1] + sin(ang)*30)
        ldr_pt = (ldr_pt_x,ldr_pt_y)
        cv2.line(im,ldr_pt,pt,(0,0,0),4)
        ang = radians(self.theta_face+60)
        ldr_pt_x = toint( pt[0] + cos(ang)*30)
        ldr_pt_y = toint( pt[1] + sin(ang)*30)
        ldr_pt = (ldr_pt_x,ldr_pt_y)
        cv2.line(im,ldr_pt,pt,(0,0,0),4)
        
        
        
        
        
        
        
        
        # im = cv2.resize(im,None, None, fx = rsz_fac, fy=rsz_fac)
        with col2:
            final_rgb_image = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            
            st.image(final_rgb_image, width=scale_width)    
        
        # cv2.imshow("Trajectory",im)
        # x = cv2.waitKey(self.wk)
        
        # rv = True
            
        # else:
        #     rv = True
        
        # self.append_data()
        # return rv
            
            
    def data_to_csv(self):
        with open(Path(__file__).resolve().parent.joinpath("data.csv"),'w') as f:
            titles = list(self.shot_data.keys())
            f.write("\t".join(titles) + "\n")
            for i in range(len(self.shot_data[titles[0]])):
                f.write("\t".join([str(self.shot_data[t][i]) for t in titles]) + "\n")
    
            
            
        
    def calculate_trajectory(self):
        
        
        end = False
        
        
        phi_cos = cos(radians(self.phi_axis))
        phi_sin = sin(radians(self.phi_axis))
        delta_t = 0.005 # five millisecond time delta
        while not end:
            v_total = pow(pow(self.vx,2) + pow(self.vy,2) + pow(self.vz,2),0.5)
            ax_drag = self.const_multiplier * drag_coeff * v_total * self.vx
            ay_drag = self.const_multiplier * drag_coeff * v_total * self.vy
            az_drag = self.const_multiplier * drag_coeff * v_total * self.vz
            a_lift_total = -1.0 * self.const_multiplier * lift_coeff * pow(v_total,2)
            ax_lift = a_lift_total * phi_sin
            az_lift = a_lift_total * phi_cos
            accel_x = ax_drag + ax_lift
            accel_y = ay_drag
            accel_z = az_drag + az_lift - gravity
            
            self.vx = self.vx + accel_x * delta_t
            self.vy = self.vy + accel_y * delta_t
            self.vz = self.vz + accel_z * delta_t
            
            x = self.xpos_arr[-1] + self.vx * delta_t
            y = self.ypos_arr[-1] + self.vy * delta_t
            z = self.zpos_arr[-1] + self.vz * delta_t
            self.xpos_arr.append(x)
            self.ypos_arr.append(y)
            self.zpos_arr.append(z)
            
            if self.zpos_arr[-1] < 0.0:
                end = True
                del self.xpos_arr[-1]
                del self.ypos_arr[-1]
                del self.zpos_arr[-1]
        
        # print(f"Apex: {m2f(max(self.zpos_arr)):.1f} ft")
        # print(f"Carry: {m2y(max(self.ypos_arr)):.1f} yd")
        # print(f"Offline: {m2y(self.xpos_arr[-1]):.1f} yd")
        # print(f"Trajectory deviation: {m2y(self.max_dev()):.1f} yd")
        ft = len(self.xpos_arr) * delta_t
        # print(f"Flight Time: {ft:.1f} s")
        
        
        
        
        
        
        
        
        
            
            
if __name__ == "__main__":
    
    # if 0:
    #     theta_path = -11.5
    #     theta_face = 3.8
    #     gtg = GolfTrajectoryGenerator(theta_path, theta_face,False,0)
    #     gtg.calculate_trajectory()
    #     gtg.draw_trajectory()
    #     # cv2.destroyAllWindows()     
        
    # else:
    # end = False
    # i_end = 241
    # j_end = 161
    # gtg = None
    # theta_path = 0.0
    # theta_face = 0.0
    # while True:
    
    #     if end:
    #         break
        
    #     if gtg is None:
    gtg = GolfTrajectoryGenerator(0.0, 0.0,False,0)
    # else:
    #     gtg.re_init(theta_path, theta_face,0)
    gtg.calc_flight()
    gtg.draw_trajectory()
        # if not res:
        #     end = True
        # if end:
        #     break
        # cv2.destroyAllWindows()
        # gtg.data_to_csv()        
    
    
    
    














