from time import sleep
from os import system, name
import argparse

# Reading user inputs
parser = argparse.ArgumentParser( description='Application to start a timer')
parser.add_argument('-t', dest='time', type=int, required=True,
        help='Time in minutes')
parser.add_argument('-c', dest='color', type=str, required=True,
        help='Color of the clock')
inps =  parser.parse_args()
mins = inps.time
chosen_color = inps.color

# Assert: time
assert mins<100 and mins>0, '\nChoose a number between 0 and 100'

# Color scheme of the clock
colors = {}
colors ['red'] = ("\033[91m", "\033[00m")
colors ['green'] = ("\033[92m", "\033[00m")
colors ['yellow'] = ("\033[93m", "\033[00m")
colors ['lightpurple'] = ("\033[94m", "\033[00m")
colors ['purple'] = ("\033[95m", "\033[00m")
colors ['cyan'] =  ("\033[96m", "\033[00m")
colors ['gray'] = ("\033[97m", "\033[00m")
colors ['black'] = ("\033[98m", "\033[00m")

# Assert: color
assert chosen_color in list(colors.keys()), '\nPlease choose a color from red, green, yellow, lightpurple, purple, cyan, gray, or black'

# Clear Screen
def clearScreen():
    if name == 'nt':
        system('cls')
    else:
        system('clear')

## Digital numbers
numbers = []
# zero
numbers.append(('@@@@@@@', '||   ||', '@@   @@', '||   ||', '@@@@@@@'))
# one
numbers.append((' @@@@  ', '   ||  ', '   @@  ', '   ||  ', '@@@@@@@'))
# two
numbers.append(('@@@@@@@', '     ||', '@@@@@@@', '||     ', '@@@@@@@'))
# three
numbers.append(('@@@@@@@', '     ||', '@@@@@@@', '     ||', '@@@@@@@'))
# four
numbers.append(('@@   @@', '||   ||', '@@@@@@@', '     ||', '     @@'))
# five
numbers.append(('@@@@@@@', '||     ', '@@@@@@@', '     ||', '@@@@@@@'))
# six
numbers.append(('@@@@@@@', '||     ', '@@@@@@@', '||   ||', '@@@@@@@'))
# seven
numbers.append(('@@@@@@@', '     ||', '     @@', '     ||', '     @@'))
# eight
numbers.append(('@@@@@@@', '||   ||', '@@@@@@@', '||   ||', '@@@@@@@'))
# nine
numbers.append(('@@@@@@@', '||   ||', '@@@@@@@', '     ||', '@@@@@@@'))
# Breaker
numbers.append(('    ', ' @@ ', '    ', ' @@ ', '    '))

# Plot the clock
def plot_digit():
    tex = ''
    for y in range(5):
        for x in [c1, c2, 10, c3, c4]:
            tex += numbers[x][y]+ "   "                
        tex += "\n"
    clearScreen()
    print('\n'+colors[chosen_color][0]+tex+colors[chosen_color][1])

# Finding minute digits
def find_c1c2():
    str_min = str(mins)
    if len(str_min)==1:
        c1, c2 = 0, int(str_min)
    else:
        c1, c2 = [int(x) for x in [*str_min]]
    return c1, c2

# Finding second digits
def find_c3c4():
    count_str = str(count)
    if len(count_str)==1:
        c3, c4 = 0, int(count_str)
    else:
        c3, c4 = [int(x) for x in [*count_str]]
    return c3, c4

while mins!=0:
    c3 = c4 = 0
    c1, c2 = find_c1c2()
    plot_digit()
    sleep(1)
    mins += -1
    c1, c2 = find_c1c2()
    for count in range(59,0,-1):
        c3, c4 = find_c3c4()
        plot_digit()
        sleep(1)

clearScreen()
print(colors[chosen_color][0]+
    '   *******     **      **   ********   *******  \n'+
    '  **/////**   /**     /**  /**/////   /**////** \n'+
    ' **     //**  /**     /**  /**        /**   /** \n'+
    '/**      /**  //**    **   /*******   /*******  \n'+
    '/**      /**   //**  **    /**////    /**///**  \n'+
    '//**     **     //****     /**        /**  //** \n'+
    ' //*******       //**      /********  /**   //**\n'+
    '  ///////         //       ////////   //     // '+
    colors[chosen_color][1]
    )

# A system notification
system('notify-send --hint int:transient:1 "Timer finished!" "Time to stretch a bit"')
