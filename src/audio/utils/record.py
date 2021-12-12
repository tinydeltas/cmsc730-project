import sys
import os 
import time
from datetime import datetime
import matlab.engine
import subprocess

eng = matlab.engine.start_matlab()
run = datetime.now().strftime("%m_%d_%Y_%H:%M:%S")

directory_default = "audio_new_microphone_sync_4s"
gesture_label_default = "test"

def record(directory, gesture_label, i): 
    # print("Starting transmission")
    # child = subprocess.Popen(["python3", "transmit.py"])
    print("Saving to: ", directory)
    if not os.path.exists(directory): 
        print("Making folder")
        os.makedirs(directory)
        
    print("Gesture label", gesture_label)
    fullpath = os.path.join(directory, gesture_label + "/" + run)
    if not os.path.exists(fullpath): 
        print("Making folder ", fullpath)
        os.makedirs(fullpath)

    while True: 
        try: 
            print("Recording now!")
            print("#: t", i)
            
            path = os.path.join(fullpath, str(i))
            print("Saving to: ", path)
        
            eng.Receiver_Signal_FMCW(path, nargout=0)
            i += 1
            
            # input("Press enter to continue, or Ctrl-C to quit")
            print("Finished recording, sleeping for 3")
            time.sleep(3)
        
        except KeyboardInterrupt:
            # child.kill()
            eng.quit()
            sys.exit(0)
    
def main():
    directory = directory_default
    gesture_label = gesture_label_default
    
    if len(sys.argv) > 1: 
        directory = sys.argv[1]

    if len(sys.argv) > 2: 
        gesture_label = sys.argv[2]
    
    record(directory, gesture_label, 0) 

main()