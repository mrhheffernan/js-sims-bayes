import matplotlib.pyplot as plt
import dill
import numpy as np

from configurations import *

def main():
    for s in systems:
        #load the dill'ed emulator from emulator file
        system_str = s[0]+"-"+s[1]+"-"+str(s[2])
        emu = dill.load(open('emulator/emu-' + system_str + '.dill', "rb"))

        #try to get some properties of the emulator
        print( "emu.npc = " +str(emu.npc) )

main()
