'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''


from pid import PIDAgent
from keyframes import hello
from keyframes import wipe_forehead
from keyframes import leftBackToStand
import numpy as np
import matplotlib.pyplot as plt
from spark_agent import INVERSED_JOINTS

epsilon = 1e-6 #error margin for x to t conversion

#data structures for debug plotting
interpolatedPoints = [[],[]]
keyframePoints = [[],[]]
testJoint = "LElbowYaw"


class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])
        self.myTime = self.perception.time
        self.done = 0 # only relevant for debug plotting
        

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes)
        self.target_joints.update(target_joints)
        print "Im Player number:", self.player_id
        return super(AngleInterpolationAgent, self).think(perception)
      

    def angle_interpolation(self, keyframes):
        target_joints = {}
        
        # YOUR CODE HERE
        time = self.perception.time - self.myTime
        (names, times, keys) = keyframes
        
        for i, name in enumerate(names):
	  curTimes = times[i]
	  
	  if curTimes[-1]<time or time<curTimes[0]: #skip if time is not in time frame
	    
	    #plot interpolated data for testing
	    if (not self.done) and curTimes[-1]<time:
	      self.done = 1
	      plt.plot(interpolatedPoints[0],interpolatedPoints[1],"r")
	      plt.plot(keyframePoints[0],keyframePoints[1],"bo")
	      plt.title(testJoint)
	      plt.show()
	    
	    continue
	  
	  #getting relevant Indices
	  eIndex = len([x for x in curTimes if x<time])
	  sIndex = eIndex - 1
	  
	  #get interpolation points
	  skey = keys[i][sIndex]
	  (p0x, p0y) = (curTimes[sIndex], skey[0])
	  (p1x, p1y) = (p0x + skey[2][1], p0y + skey[2][2])
	  ekey = keys[i][eIndex]
	  (p3x, p3y) = (curTimes[eIndex], ekey[0])
	  (p2x, p2y) = (p3x + ekey[1][1], p3y + ekey[1][2])
	  
	  #calculating bezier polynomial as described in http://pomax.github.io/bezierinfo/
	  bezierMatrix = np.array([[1,0,0,0],[-3,3,0,0],[3,-6,3,0],[-1,3,-3,1]])
	  x = np.array([p0x,p1x,p2x,p3x])
	  y = np.array([p0y,p1y,p2y,p3y])
	  
	  #getting t value candidates (solutions for the polynomial) for curTime
	  coefficientsX = np.dot(bezierMatrix, x)
	  coefficientsX[0] -= time
	  candidates = np.polynomial.polynomial.polyroots(coefficientsX)
	  
	  #finding correct candidate for t (t has to be in [0,1])
	  candidates = [x.real for x in candidates if -(epsilon)<=x.real<=1+(epsilon) and x.imag == 0] #error margin uncertain
	   
	  ''' #testing t values close to boundaries
	  if not candidates:
	    print "candidates was empty"
	  for x in candidates:
	    if x >1:
	      f = open("workfile","a+")
	      f.write("[")
	      for y in t:
		f.write("%.32f,"%y)
	      f.write("]")
	      f.close()
	  '''
	  #solution should be unique but due to error margin solutions actually marginally larger than 1 (or actually marginally smaller than 0) might get chosen
	  if len(candidates) > 1: # if thats the case, there must also exist a correct solution -> choose the one closer to 0.5
	  	candidates = np.asarray([(x,np.abs(x-0.5)) for x in candidates],dtype = [("value", float),("distance", float)]) 
	  	candidates = np.sort(candidates, order="distance")
	  	t = candidates[0][0]
	  else: #if there's only one solution it must be the right one
	  	t = candidates[0]
	  
	  if t < 0.: #clip values marginally smaller than 0 to 0
	    t = 0.
	  if t > 1.: #clip values marginally larger than 1 to 1
	    t = 1.
	    
	  #getting y values
	  coefficientsY = np.dot(bezierMatrix, y)
	  result = np.dot(np.array([1, t, t**2, t**3]),coefficientsY)
	  target_joints[name] = result
	  
	  
	  #collecting plot data
	  if name == testJoint:
	    interpolatedPoints[0].append(time)
	    interpolatedPoints[1].append(result)
	    keyframePoints[0].append(p0x)
	    keyframePoints[1].append(p0y)
	    keyframePoints[0].append(p3x)
	    keyframePoints[1].append(p3y)
	  
        
        return target_joints

if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = leftBackToStand()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
