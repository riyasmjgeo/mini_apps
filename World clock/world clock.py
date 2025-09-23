#importing module
from tkinter import *
from tkinter.ttk import *
# from time import strftime
from datetime import datetime
from os import name, system
from time import sleep
import pytz

# Define the time zones
time_zones = {
    "CGY ": "America/Edmonton",
    "ABD ": "Europe/London",
    "HYD ": "Asia/Kolkata"  
}

#creating tkinter window
root = Tk()
root.title("WorldClock")

#Display time on Label
def time():
    string = ''
    for city, tz in time_zones.items():
        local_time = datetime.now(pytz.timezone(tz))
        string += city + local_time.strftime('%I:%M %p') + '\n'
    lbl.config(text=string[:-1])
    lbl.after(1000 * 60, time)

#styling label
lbl = Label (root, font=("lucidia", 45, "bold"),
             background="White",
             foreground="Black")
        
#center Placing
lbl.pack(anchor="center")
time()
mainloop()