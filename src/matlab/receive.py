import sys
import matlab.engine
import sched, time

s = sched.scheduler(time.time, time.sleep)
eng = matlab.engine.start_matlab()

def receive(sc): 
    eng.Receiver_Signal_FMCW(sys.argv[1], nargout=0)
    s.enter(0.1, 1, receive, (sc,))


s.enter(60, 1, receive, (s,))
s.run()
    
