import sys
import os 
import time
import matlab.engine
import subprocess

eng = matlab.engine.start_matlab()

print("Starting transmission")
child = subprocess.Popen(["python3", "transmit.py"])
time.sleep(2)

i = 0 
while True: 
    try: 
        time.sleep(1)
        print("Recording now!")
        print("Directory: \t", sys.argv[1])
        print("Gesture: \t", sys.argv[2])
        print("#: \t\t", i)
        
        path = os.path.join(sys.argv[1] + "/" + sys.argv[2], str(i))
        print("Saving to: ", path)
        
        eng.Receiver_Signal_FMCW(path, nargout=0)
        i += 1
        input("Press enter to continue, or Ctrl-C to quit")
    
    except KeyboardInterrupt:
        child.terminate()
        eng.quit()
        sys.exit(0)
    