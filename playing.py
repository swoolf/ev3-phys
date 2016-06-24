#!/usr/bin/env python

import ev3dev.ev3 as ev3
from ev3dev.auto import *
import draw
import webhost as webb
import threading

from time import sleep

#Set Up
mR = ev3.LargeMotor('outA')
mL = ev3.LargeMotor('outB')
cs = ColorSensor()
cs.mode = 'COL-COLOR'

numIfThen = 2

def checkInputs(task):
    oldColor = None
    state = None
    lastState = ''
    stateBuffer = 0
    send = False
    toSend = None

    if(task == 'if'):
        while True:
            Rspeed = mR.speed
            Lspeed = mL.speed
            color = cs.value()
            if Rspeed < -20:
                state = 'Back Right'
            elif Rspeed > 20:
                state = 'Turn Left'
            elif Lspeed < -20:
                state = 'Back Left'
            elif Lspeed > 20:
                state = 'Turn Right'
            elif color == 6:
                state = 'seeWhite'
            elif color == 1:
                state = 'seeBlack'
            else:
                stateBuffer=0

            if state == lastState and state != None:
                stateBuffer +=1
                if stateBuffer == 3:
                    toSend = state
                    send =True
                    ev3.Sound.speak(toSend).wait()

            if send and state != toSend: return toSend

            lastState = state
            state = None
            sleep(.1)
    elif(task == 'then'):
        while True:
            Rspeed = mR.speed
            Lspeed = mL.speed
            if Rspeed < -20:
                state = 'Back Left'
            elif Rspeed > 20:
                state = 'Turn Left'
            elif Lspeed < -20:
                state = 'Back Right'
            elif Lspeed > 20:
                state = 'Turn Right'
            else:
                stateBuffer=0
        
            if state == lastState and state != None:
                stateBuffer +=1
            if stateBuffer == 3:
                toSend = state
                send =True
                ev3.Sound.speak(toSend).wait()

            if send and state != toSend: return toSend
        
            lastState = state
            state = None
            sleep(.1)

def runPgm(program, site, btn):
    print 'Running....'
    site.post(prog2string(program) + "<br><br>Now Running Program... <br><br>Press 'DOWN' to restart..." , 'index.html')
    ev3.Sound.speak('Now running program').wait()
    while not btn.any():
        val = cs.value()
        if val == 6:
            for i in (0,2):
                if program[i]=='seeWhite' :
                    doAction(program[i+1])
        if val == 1:
            for i in (0,2):
                if program[i]=='seeBlack' :
                    doAction(program[i+1])
    ev3.Sound.beep()

def doAction(action):
    if action == 'Turn Left':
        mR.run_timed(time_sp=500, duty_cycle_sp=30)
        mL.run_timed(time_sp=500, duty_cycle_sp=0)
    if action == 'Back Left':
        mR.run_timed(time_sp=500, duty_cycle_sp=-30)
        mL.run_timed(time_sp=500, duty_cycle_sp=0)
    if action == 'Turn Right':
        mL.run_timed(time_sp=500, duty_cycle_sp=30)
        mR.run_timed(time_sp=500, duty_cycle_sp=0)
    if action == 'Back Right':
        mL.run_timed(time_sp=500, duty_cycle_sp=-30)
        mR.run_timed(time_sp=500, duty_cycle_sp=0)

def post2site(program,site):
    site.post(prog2string(program), 'index.html')

def prog2string(program):
    msg = 'If: '
    for i in range(0, len(program)):
        if (i % 2 == 0):
            msg +=  program[i] + '&nbsp; &nbsp; &nbsp; &nbsp; Then:  '
        else:
            msg +=  program[i] + '<br>'
            if (i !=numIfThen*2-1): msg+= 'If:  '
    return msg

def getIfThen(program,site):
    ev3.Sound.speak('if').wait()
    post2site(program, site)
    program.append(checkInputs('if'))
    post2site(program, site)
    sleep(.5)
    ev3.Sound.speak('then').wait()
    program.append( checkInputs('then'))
    post2site(program, site)
    return program

def main():
    btn = Button()
    site = webb.webScreen()
    ev3.Sound.tone([(300,100,100)]).wait()
    ev3.Sound.tone([(500,100,100)]).wait()
    ev3.Sound.tone([(600,300,0)]).wait()
    site.post( "<br><br>Press 'DOWN' to begin.." , 'index.html')
    while not btn.any():
        pass
    while True:
        program = []
        site.post("", "index.html")
        sleep(1)
        for x in range(0,numIfThen):
            getIfThen(program,site)
            print program
        site.post(prog2string(program) + "<br><br>Press 'DOWN' to run program..." , 'index.html')
        while not btn.any():
                pass
        ev3.Sound.beep()
        sleep(1)
        runPgm(program, site, btn)

def startServer(tM):
    Gsite = webb.webScreen()
    Gsite.post("", "index.html")
    Gsite.serve()


tM = threading.Thread(target=main)
tM.start()



