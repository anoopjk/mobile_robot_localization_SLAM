# The full Kalman filter, consisting of prediction and correction step.
#
# slam_07_f_kalman_filter
# Claus Brenner, 12.12.2012
from lego_robot import *
from math import sin, cos, pi, atan2, sqrt
from numpy import *
from slam_d_library import get_observations, write_cylinders


class ExtendedKalmanFilter:
    def __init__(self, state, covariance,
                 robot_width, scanner_displacement,
                 control_motion_factor, control_turn_factor,
                 measurement_distance_stddev, measurement_angle_stddev):
        # The state. This is the core data of the Kalman filter.
        self.state = state
        self.covariance = covariance

        # Some constants.
        self.robot_width = robot_width
        self.scanner_displacement = scanner_displacement
        self.control_motion_factor = control_motion_factor
        self.control_turn_factor = control_turn_factor
        self.measurement_distance_stddev = measurement_distance_stddev
        self.measurement_angle_stddev = measurement_angle_stddev

    @staticmethod
    def g(state, control, w):
        x, y, theta = state
        l, r = control
        if r != l:
            alpha = (r - l) / w
            rad = l/alpha
            g1 = x + (rad + w/2.)*(sin(theta+alpha) - sin(theta))
            g2 = y + (rad + w/2.)*(-cos(theta+alpha) + cos(theta))
            g3 = (theta + alpha + pi) % (2*pi) - pi
        else:
            g1 = x + l * cos(theta)
            g2 = y + l * sin(theta)
            g3 = theta

        return array([g1, g2, g3])

    @staticmethod
    def dg_dstate(state, control, w):

        # --->>> Put your method from 07_d_kalman_predict here.
        theta = state[2]
        l, r = control
        alpha = (r-l)/w 
#        print "control is as follows"
#        print control
#        print "state is as follows"
#        print state
#        print "w is as follows"
#        print w 
        if r != l:

            # --->>> Put your code here.
            # This is for the case r != l.
            # g has 3 components and the state has 3 components, so the
            # derivative of g with respect to all state variables is a
            # 3x3 matrix. To construct such a matrix in Python/Numpy,
            # use: m = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            # where 1, 2, 3 are the values of the first row of the matrix.
            # Don't forget to return this matrix.
            R = l/alpha

            m = array([[1, 0, (R+w/2)*(cos(theta+alpha)- cos(theta))], [0, 1, (R+w/2)*(sin(theta+alpha)-sin(theta))], [0, 0, 1]])  # Replace this.

        else:

            # --->>> Put your code here.
            # This is for the special case r == l.
            m = array([[1, 0, -l*sin(theta) ], [0, 1, l*cos(theta)], [0, 0, 1]])  # Replace this.

        return m

       

    @staticmethod
    def dg_dcontrol(state, control, w):

        # --->>> Put your method from 07_d_kalman_predict here.
        theta = state[2]
        l, r = tuple(control)
                
        
        if r != l:
            #print "r is not equal to l"
            
            # --->>> Put your code here.
            # This is for the case l != r.
            # Note g has 3 components and control has 2, so the result
            # will be a 3x2 (rows x columns) matrix.
            alpha = (r-l)/w
            A1 = w*r/(r-l)**2
            B1 = (r+l)/(2*(r-l))
            dg1 = A1*(sin(theta+alpha)- sin(theta)) - B1*cos(theta+alpha)
            dg3 = A1*(-cos(theta+alpha)+ cos(theta)) - B1*sin(theta+alpha)
            dg5 = -1/w
            A2 = -w*l/(r-l)**2
            B2 = (r+l)/(2*(r-l))
            dg2 = A2*(sin(theta+alpha)- sin(theta)) + B2*cos(theta+alpha)
            dg4 = A2*(-cos(theta+alpha)+ cos(theta)) + B2*sin(theta+alpha)
            dg6 = 1/w
            
            
            
        else:

            # --->>> Put your code here.
            # This is for the special case l == r.
            dg1 = 1/2*(cos(theta)+ l/w*sin(theta))
            dg3 = 1/2*(sin(theta)- l/w*cos(theta))
            dg5 = -1/w
            dg2 = 1/2*(cos(theta) - l/w*sin(theta))
            dg4 = 1/2*(sin(theta)+ l/w*cos(theta))
            dg6 = 1/w          

        m = array([[dg1, dg2], [dg3, dg4], [dg5, dg6]])  # Remove this.
            
        return m


    @staticmethod
    def get_error_ellipse(covariance):
        """Return the position covariance (which is the upper 2x2 submatrix)
           as a triple: (main_axis_angle, stddev_1, stddev_2), where
           main_axis_angle is the angle (pointing direction) of the main axis,
           along which the standard deviation is stddev_1, and stddev_2 is the
           standard deviation along the other (orthogonal) axis."""
        eigenvals, eigenvects = linalg.eig(covariance[0:2,0:2])
        angle = atan2(eigenvects[1,0], eigenvects[0,0])
        return (angle, sqrt(eigenvals[0]), sqrt(eigenvals[1]))        

    def predict(self, control):

        # --->>> Put your method from 07_d_kalman_predict here.
        """The prediction step of the Kalman filter."""
        # covariance' = G * covariance * GT + R
        # where R = V * (covariance in control space) * VT.
        # Covariance in control space depends on move distance.
        left, right = control

        # --->>> Put your code to compute the new self.covariance here.
        # First, construct the control_covariance, which is a diagonal matrix.
        # In Python/Numpy, you may use diag([a, b]) to get
        # [[ a, 0 ],
        #  [ 0, b ]].
        alpha1 = self.control_motion_factor
        alpha2 = self.control_turn_factor
        varl = (alpha1*left)**2 + (alpha2*(left-right))**2
        varr = (alpha1*right)**2 + (alpha2*(left-right))**2
        control_covariance = diag([varl,varr])
        # Then, compute G using dg_dstate and V using dg_dcontrol.
        G = self.dg_dstate(self.state, control, self.robot_width) 
        V = self.dg_dcontrol(self.state, control, self.robot_width)
        # Then, compute the new self.covariance.
        self.covariance = dot(G,dot(self.covariance,G.T)) + dot(V,dot(control_covariance,V.T))
        # Note that the transpose of a Numpy array G is expressed as G.T,
        # and the matrix product of A and B is written as dot(A, B).
        # Writing A*B instead will give you the element-wise product, which
        # is not intended here.

        # state' = g(state, control)
        self.state = self.g(self.state, control, self.robot_width)

    @staticmethod
    def h(state, landmark, scanner_displacement):
        """Takes a (x, y, theta) state and a (x, y) landmark, and returns the
           measurement (range, bearing)."""
        dx = landmark[0] - (state[0] + scanner_displacement * cos(state[2]))
        dy = landmark[1] - (state[1] + scanner_displacement * sin(state[2]))
        r = sqrt(dx * dx + dy * dy)
        alpha = (atan2(dy, dx) - state[2] + pi) % (2*pi) - pi

        return array([r, alpha])

    @staticmethod
    def dh_dstate(state, landmark, scanner_displacement):

        # --->>> Put your method from 07_e_measurement derivative here.
                # --->>> Insert your code here.
        # Note that:
        # x y theta is state[0] state[1] state[2]
        x = state[0]
        y = state[1]
        theta = state[2]    
        # x_m y_m is landmark[0] landmark[1]
        x_m =  landmark[0]
        y_m = landmark[1]
        # The Jacobian of h is a 2x3 matrix.
        x_l = x+ scanner_displacement*cos(theta)
        y_l = y+ scanner_displacement*sin(theta)
        
        dx = x_m - x_l
        dy = y_m - y_l
        
        q = dx*dx + dy*dy 
        
        drdx = -dx/sqrt(q)
        drdy = -dy/sqrt(q)
        drdtheta = (scanner_displacement/sqrt(q))*(dx*sin(theta)- dy*cos(theta))
        
        dalphadx = dy/q
        dalphady = -dx/q
        dalphadtheta = -(scanner_displacement/q)*(dx*cos(theta) + dy*sin(theta)) - 1
        
        
        return array([[drdx, drdy, drdtheta], [dalphadx, dalphady, dalphadtheta]]) 

    def correct(self,  measurement,  landmark):
        """The correction step of the Kalman filter."""

        # --->>> Put your new code here.
        #
        # You will have to compute:
        # H, using dh_dstate(...).
        H = self.dh_dstate(self.state, landmark, self.scanner_displacement)
        # Q, a diagonal matrix, from self.measurement_distance_stddev and
        #  self.measurement_angle_stddev (remember: Q contains variances).
        Q = diag([self.measurement_distance_stddev**2, self.measurement_angle_stddev**2 ])
        # K, from self.covariance, H, and Q.
        #  Use linalg.inv(...) to compute the inverse of a matrix.
        #residual covariance
        S = linalg.inv(dot(H,dot(self.covariance,H.T)) + Q)
       
        K = dot(self.covariance,dot(H.T,S))
        
        # The innovation: it is easy to make an error here, because the
        #  predicted measurement and the actual measurement of theta may have
        #  an offset of +/- 2 pi. So here is a suggestion:
        #   innovation = array(measurement) -\
        #                self.h(self.state, landmark, self.scanner_displacement)
        #   innovation[1] = (innovation[1] + pi) % (2*pi) - pi
        #residual or innovation
        #innovation = array(measurement) - self.h(self.state, landmark, self.scanner_displacement)

        innovation = array(measurement) - \
                        self.h(self.state, landmark, self.scanner_displacement)
        
        innovation[1] = (innovation[1]+pi)%(2*pi) - pi
        # Then, you'll have to compute the new self.state.
        self.state = self.state + dot(K,innovation)
        
        # And finally, compute the new self.covariance. Use eye(3) to get a 3x3
        #  identity matrix.
        self.covariance = dot((eye(3)- dot(K,H)),self.covariance)        
        
        
        # Hints:
        # dot(A, B) is the 'normal' matrix product (do not use: A*B).
        # A.T is the transposed of a matrix A (A itself is not modified).
        # linalg.inv(A) returns the inverse of A (A itself is not modified).
        # eye(3) returns a 3x3 identity matrix.

        

if __name__ == '__main__':
    # Robot constants.
    scanner_displacement = 30.0
    ticks_to_mm = 0.349
    robot_width = 155.0

    # Cylinder extraction and matching constants.
    minimum_valid_distance = 20.0
    depth_jump = 100.0
    cylinder_offset = 90.0
    max_cylinder_distance = 300.0

    # Filter constants.
    control_motion_factor = 0.35  # Error in motor control.
    control_turn_factor = 0.6  # Additional error due to slip when turning.
    measurement_distance_stddev = 200.0  # Distance measurement error of cylinders.
    measurement_angle_stddev = 15.0 / 180.0 * pi  # Angle measurement error.

    # Measured start position.
    initial_state = array([1850.0, 1897.0, 213.0 / 180.0 * pi])
    # Covariance at start position.
    initial_covariance = diag([100.0**2, 100.0**2, (10.0 / 180.0 * pi) ** 2])
    # Setup filter.
    kf = ExtendedKalmanFilter(initial_state, initial_covariance,
                              robot_width, scanner_displacement,
                              control_motion_factor, control_turn_factor,
                              measurement_distance_stddev,
                              measurement_angle_stddev)

    # Read data.
    logfile = LegoLogfile()
    logfile.read("robot4_motors.txt")
    logfile.read("robot4_scan.txt")
    logfile.read("robot_arena_landmarks.txt")
    reference_cylinders = [l[1:3] for l in logfile.landmarks]

    # Loop over all motor tick records and all measurements and generate
    # filtered positions and covariances.
    # This is the Kalman filter loop, with prediction and correction.
    states = []
    covariances = []
    matched_ref_cylinders = []
    for i in xrange(len(logfile.motor_ticks)):
        # Prediction.
        control = array(logfile.motor_ticks[i]) * ticks_to_mm
        kf.predict(control)

        # Correction.
        observations = get_observations(
            logfile.scan_data[i],
            depth_jump, minimum_valid_distance, cylinder_offset,
            kf.state, scanner_displacement,
            reference_cylinders, max_cylinder_distance)
        
        #observations = observations[-1:]
        for j in xrange(len(observations)):
            kf.correct(*observations[j])

        # Log state, covariance, and matched cylinders for later output.
        states.append(kf.state)
        covariances.append(kf.covariance)
        matched_ref_cylinders.append([m[1] for m in observations])

    # Write all states, all state covariances, and matched cylinders to file.
    f = open("kalman_prediction_and_correction.txt", "w")
    for i in xrange(len(states)):
        # Output the center of the scanner, not the center of the robot.
        print >> f, "F %f %f %f" % \
            tuple(states[i] + [scanner_displacement * cos(states[i][2]),
                               scanner_displacement * sin(states[i][2]),
                               0.0])
        # Convert covariance matrix to angle stddev1 stddev2 stddev-heading form
        e = ExtendedKalmanFilter.get_error_ellipse(covariances[i])
        print >> f, "E %f %f %f %f" % (e + (sqrt(covariances[i][2,2]),))
        # Also, write matched cylinders.
        write_cylinders(f, "W C", matched_ref_cylinders[i])        

    f.close()
