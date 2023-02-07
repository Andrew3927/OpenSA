def CL_green_print(string):
    print("\033[32m" + string + "\033[0m")

def CL_red_print(string):
    print( "\033[1;31;40m" + string + "\033[0m")

def __printConfiguration(EPOCH, acti_func, cnn_depth, loss, optim, is_autoTune):
    print("Training configuration:  " + "EPOCH=" + str(EPOCH) + ", activation function=" +
          acti_func + ", cnn_depth=" + str(cnn_depth) + ", loss function=" + loss + ", optimizer=" +
          optim + ", is_autoTune=" + str(is_autoTune) + "\n")

